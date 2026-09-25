"""
zones.py

Defines named polygon zones on a camera's frame (e.g. "restricted_area",
"pedestrian_lane") and tracks, per object track_id, which zones it is
currently inside, when it entered each zone, and how long it has
remained ("dwell time").

Zone polygons are stored as NORMALIZED coordinates (0.0-1.0 for both x
and y) so the same zone definition works regardless of the actual frame
resolution of a given camera/video. They are scaled to pixel coordinates
at check time using the current frame's width/height.

Point-in-polygon uses a standard ray-casting algorithm; no extra
dependency (e.g. shapely) is required for this.
"""

import time
from collections import defaultdict


def point_in_polygon(x, y, polygon):
    """
    Ray-casting point-in-polygon test.
    `polygon` is a list of (x, y) pixel-coordinate tuples.
    """
    n = len(polygon)
    inside = False
    if n < 3:
        return False

    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    else:
                        x_intersect = x1
                    if x1 == x2 or x <= x_intersect:
                        inside = not inside
        x1, y1 = x2, y2

    return inside


class Zone:
    """A single named polygon zone with a semantic type (restricted, monitored, etc)."""

    def __init__(self, name, zone_type, polygon_norm=None, allowed_classes=None,
                 severity=None, max_dwell_seconds=None, max_objects=None,
                 points=None, dwell_warning_seconds=None, dwell_critical_seconds=None):
        self.name = name
        self.zone_type = zone_type
        self.polygon_norm = polygon_norm if polygon_norm is not None else (points or [])
        # If set, only these classes are considered "in violation" when in
        # this zone (e.g. a "pedestrian_only" zone flags vehicles).
        self.allowed_classes = set(allowed_classes) if allowed_classes else None
        # Baseline display severity for this zone (LOW/MEDIUM/HIGH). Purely
        # informational - the risk engine's rules still decide actual risk.
        self.severity = severity or ("HIGH" if zone_type == "restricted" else "LOW")
        # Optional behavioral thresholds (Phase 5/3). None disables the
        # corresponding check, preserving old behavior for zones/tests that
        # don't set them.
        self.max_dwell_seconds = max_dwell_seconds
        self.max_objects = max_objects
        self.dwell_warning_seconds = dwell_warning_seconds
        self.dwell_critical_seconds = dwell_critical_seconds

    def pixel_polygon(self, frame_w, frame_h):
        return [(px * frame_w, py * frame_h) for px, py in self.polygon_norm]

    def contains(self, x, y, frame_w, frame_h):
        return point_in_polygon(x, y, self.pixel_polygon(frame_w, frame_h))


class ZoneManager:
    """Tracks zone membership / dwell time for every active track on a camera."""

    # Loitering is reported once a track's dwell passes
    # LOITERING_MULTIPLIER * zone.max_dwell_seconds, giving a distinct,
    # more severe event beyond the initial EXTENDED_DWELL notice.
    LOITERING_MULTIPLIER = 3.0

    def __init__(self, zones, dwell_warning_seconds=5.0, dwell_critical_seconds=15.0):
        self.zones = zones  # list[Zone]
        self.dwell_warning_seconds = float(dwell_warning_seconds)
        self.dwell_critical_seconds = float(dwell_critical_seconds)
        # track_id -> {zone_name: entry_timestamp}
        self._track_zone_entry = {}
        # track_id -> {zone_name: {"extended_dwell", "loitering"}} - which
        # one-shot dwell flags have already fired for this occupancy, so we
        # don't emit the same behavioral event every single frame.
        self._track_zone_flags = {}
        # zone_name -> set(track_id) currently inside, for this frame only.
        # Populated by update(); read by zone_violations() after all tracks
        # in the frame have been processed.
        self._frame_occupancy = defaultdict(set)

    def begin_frame(self):
        """Call once per frame before update()-ing each track."""
        self._frame_occupancy = defaultdict(set)

    def update(self, track_id, class_name, center, frame_w, frame_h, timestamp=None):
        """
        Check the given track's current position against all zones.

        Returns a list of event dicts:
            {"type": "ZONE_ENTER"|"ZONE_EXIT"|"ZONE_DWELL"|"EXTENDED_DWELL"|"LOITERING",
             "zone": name, "zone_type": ..., "zone_severity": ...,
             "dwell_seconds": float, "class_name": ..., "violation": bool,
             "entry_time": float, "track_id": str, "threshold_crossed": str}
        and the set of zone names currently containing the track.
        """
        timestamp = timestamp if timestamp is not None else time.time()
        x, y = center

        current_zones = {}
        for zone in self.zones:
            if zone.contains(x, y, frame_w, frame_h):
                current_zones[zone.name] = zone
                self._frame_occupancy[zone.name].add(track_id)

        prev_entry = self._track_zone_entry.get(track_id, {})
        prev_names = set(prev_entry.keys())
        current_names = set(current_zones.keys())
        flags = self._track_zone_flags.setdefault(track_id, {})

        events = []

        # New entries
        for name in current_names - prev_names:
            prev_entry[name] = timestamp
            zone = current_zones[name]
            events.append(self._make_event(
                "ZONE_ENTER", zone, class_name, 0.0,
                entry_time=timestamp, track_id=track_id,
            ))

        # Exits
        for name in prev_names - current_names:
            entry_time = prev_entry.pop(name)
            zone = self._find_zone(name)
            events.append(self._make_event(
                "ZONE_EXIT", zone, class_name, timestamp - entry_time,
                entry_time=entry_time, track_id=track_id,
            ))
            flags.pop(name, None)

        # Still inside -> dwell update (+ one-shot behavioral events)
        for name in current_names & prev_names:
            zone = current_zones[name]
            entry_t = prev_entry[name]
            dwell = timestamp - entry_t
            events.append(self._make_event(
                "ZONE_DWELL", zone, class_name, dwell,
                entry_time=entry_t, track_id=track_id,
            ))
            events.extend(self._dwell_behavior_events(
                zone, class_name, dwell, flags,
                entry_time=entry_t, track_id=track_id,
            ))

        self._track_zone_entry[track_id] = prev_entry

        return events, current_names

    def _dwell_behavior_events(self, zone, class_name, dwell, flags, entry_time=None, track_id=None):
        """One-shot EXTENDED_DWELL / LOITERING events with configurable warning and critical thresholds."""
        events = []
        warn_threshold = (
            zone.dwell_warning_seconds if getattr(zone, "dwell_warning_seconds", None) is not None
            else (zone.max_dwell_seconds if zone.max_dwell_seconds else (
                self.dwell_warning_seconds if zone.zone_type == "restricted" else None
            ))
        )
        if not warn_threshold:
            return events

        crit_threshold = (
            zone.dwell_critical_seconds if getattr(zone, "dwell_critical_seconds", None) is not None
            else (
                zone.max_dwell_seconds * self.LOITERING_MULTIPLIER if zone.max_dwell_seconds
                else self.dwell_critical_seconds
            )
        )

        zone_flags = flags.setdefault(zone.name, set())

        if dwell >= warn_threshold and "extended_dwell" not in zone_flags:
            zone_flags.add("extended_dwell")
            events.append(self._make_event(
                "EXTENDED_DWELL", zone, class_name, dwell,
                entry_time=entry_time, track_id=track_id,
                threshold_crossed="WARNING", threshold_seconds=warn_threshold,
            ))

        if dwell >= crit_threshold and "loitering" not in zone_flags:
            zone_flags.add("loitering")
            events.append(self._make_event(
                "LOITERING", zone, class_name, dwell,
                entry_time=entry_time, track_id=track_id,
                threshold_crossed="CRITICAL", threshold_seconds=crit_threshold,
            ))

        return events

    def zone_violations(self):
        """
        Call once per frame, after every track has been passed through
        update(), to detect zone-level (rather than per-track) violations:
        too many simultaneous objects in a zone that defines max_objects.

        Returns a list of event dicts:
            {"type": "GROUP_INCURSION"|"UNAUTHORIZED_ZONE_ENTRY",
             "zone": name, "zone_type": ..., "zone_severity": ...,
             "count": int, "max_objects": int}
        """
        events = []
        for zone in self.zones:
            if not zone.max_objects:
                continue
            occupants = self._frame_occupancy.get(zone.name, set())
            if len(occupants) >= zone.max_objects:
                events.append({
                    "type": "GROUP_INCURSION",
                    "zone": zone.name,
                    "zone_type": zone.zone_type,
                    "zone_severity": zone.severity,
                    "count": len(occupants),
                    "max_objects": zone.max_objects,
                })
        return events

    def _make_event(self, event_type, zone, class_name, dwell_seconds,
                    entry_time=None, track_id=None, threshold_crossed=None, threshold_seconds=None):
        violation = bool(
            zone.allowed_classes and class_name not in zone.allowed_classes
        )
        ev = {
            "type": event_type,
            "zone": zone.name,
            "zone_type": zone.zone_type,
            "zone_severity": zone.severity,
            "dwell_seconds": round(dwell_seconds, 1),
            "class_name": class_name,
            "violation": violation,
        }
        if entry_time is not None:
            ev["entry_time"] = float(entry_time)
        if track_id is not None:
            ev["track_id"] = str(track_id)
        if threshold_crossed is not None:
            ev["threshold_crossed"] = threshold_crossed
        if threshold_seconds is not None:
            ev["threshold_seconds"] = float(threshold_seconds)
        return ev

    def _find_zone(self, name):
        for zone in self.zones:
            if zone.name == name:
                return zone
        # Zone config changed mid-run; fall back to a generic placeholder
        return Zone(name, "unknown", [])

    def forget_track(self, track_id):
        self._track_zone_entry.pop(track_id, None)
        self._track_zone_flags.pop(track_id, None)

    def cleanup(self, active_track_ids):
        """Drop bookkeeping for tracks that no longer exist."""
        stale = [tid for tid in self._track_zone_entry if tid not in active_track_ids]
        for tid in stale:
            del self._track_zone_entry[tid]
        stale_flags = [tid for tid in self._track_zone_flags if tid not in active_track_ids]
        for tid in stale_flags:
            del self._track_zone_flags[tid]


def load_zones_for_camera(zone_config, camera_id):
    """
    Build Zone objects for a given camera from the parsed zones.json config.

    Supports a per-camera key, falling back to a "default" key so a
    single zone layout can be shared across cameras that weren't given
    an explicit override.
    """
    camera_zones = zone_config.get(camera_id, zone_config.get("default", []))
    zones = []
    for z in camera_zones:
        zones.append(Zone(
            name=z["name"],
            zone_type=z.get("type", "monitored"),
            polygon_norm=z["polygon"],
            allowed_classes=z.get("allowed_classes"),
            severity=z.get("severity"),
            max_dwell_seconds=z.get("max_dwell_seconds"),
            max_objects=z.get("max_objects"),
        ))
    return zones
