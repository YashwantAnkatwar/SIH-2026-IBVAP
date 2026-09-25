"""
lines.py

Configurable virtual line-crossing detection (Phase 4).

A Line is defined by two points (NORMALIZED 0.0-1.0, same convention as
zones.py) which split the frame into "side A" and "side B" using the
sign of the 2D cross product of the line direction vs. the vector to the
object's center point. When a track's side flips between two
consecutive frames, that is a crossing event, and the direction
(A->B or B->A) is reported using the line's configured human-readable
side labels.

This intentionally does not try to infer compass directions like
"LEFT -> RIGHT" from an arbitrary polygon; instead it reports crossings
in terms of the two sides *as configured* (e.g. "APPROACH SIDE" ->
"RESTRICTED SIDE"), which is both simpler and more meaningful for an
arbitrarily-oriented checkpoint line. For the common axis-aligned case
(a vertical or horizontal line) this naturally corresponds to
LEFT/RIGHT or TOP/BOTTOM crossings.
"""


def _side(point_a, point_b, x, y):
    """
    Sign of the cross product of (point_b - point_a) x (p - point_a).
    Positive/negative indicates which side of the line p falls on;
    0 means exactly on the line (treated as no crossing, to avoid
    div-by-zero-style flapping on an exact boundary hit).
    """
    ax, ay = point_a
    bx, by = point_b
    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


class Line:
    """A single named virtual crossing line."""

    def __init__(self, name, point_a_norm, point_b_norm,
                 side_a_label="SIDE A", side_b_label="SIDE B"):
        self.name = name
        self.point_a_norm = point_a_norm
        self.point_b_norm = point_b_norm
        self.side_a_label = side_a_label
        self.side_b_label = side_b_label

    def pixel_points(self, frame_w, frame_h):
        ax, ay = self.point_a_norm
        bx, by = self.point_b_norm
        return (ax * frame_w, ay * frame_h), (bx * frame_w, by * frame_h)

    def side_of(self, x, y, frame_w, frame_h):
        """Returns +1, -1, or 0 (exactly on the line) for point (x, y)."""
        point_a, point_b = self.pixel_points(frame_w, frame_h)
        return _side(point_a, point_b, x, y)

    def label_for_side(self, side):
        if side > 0:
            return self.side_a_label
        if side < 0:
            return self.side_b_label
        return "ON LINE"


class LineCrossingDetector:
    """
    Tracks which side of each configured Line every active track_id was
    last seen on, and reports a crossing event the frame a track's side
    flips. Includes debouncing / cooldown per track to avoid boundary flapping.
    """

    def __init__(self, lines, debounce_seconds=2.0):
        self.lines = lines  # list[Line]
        self.debounce_seconds = debounce_seconds
        # track_id -> {line_name: last_side (+1/-1)}
        self._last_side = {}
        # (track_id, line_name) -> last_crossing_time
        self._last_crossing_time = {}

    def update(self, track_id, center, frame_w, frame_h, timestamp=None):
        """
        Check the given track's current position against all configured
        lines. Returns a list of event dicts:
            {"type": "LINE_CROSSING", "line": name,
             "direction": "A_TO_B"|"B_TO_A",
             "crossing_direction": "OUTSIDE -> INSIDE"|"INSIDE -> OUTSIDE",
             "from_label": ..., "to_label": ...}
        """
        x, y = center
        events = []
        track_sides = self._last_side.setdefault(track_id, {})

        for line in self.lines:
            side = line.side_of(x, y, frame_w, frame_h)
            if side == 0:
                continue  # exactly on the line this frame; wait for a clear side

            prev_side = track_sides.get(line.name)
            if prev_side is not None and side != prev_side:
                # Check debouncing if timestamp is provided
                debounced = False
                if timestamp is not None and self.debounce_seconds > 0:
                    last_time = self._last_crossing_time.get((track_id, line.name))
                    if last_time is not None and (timestamp - last_time) < self.debounce_seconds:
                        debounced = True

                if not debounced:
                    if timestamp is not None:
                        self._last_crossing_time[(track_id, line.name)] = timestamp

                    from_label = line.label_for_side(prev_side)
                    to_label = line.label_for_side(side)
                    direction = "A_TO_B" if side < 0 else "B_TO_A"
                    crossing_direction = f"{from_label} -> {to_label}"
                    events.append({
                        "type": "LINE_CROSSING",
                        "line": line.name,
                        "direction": direction,
                        "crossing_direction": crossing_direction,
                        "from_label": from_label,
                        "to_label": to_label,
                    })

            track_sides[line.name] = side

        return events

    def forget_track(self, track_id):
        self._last_side.pop(track_id, None)
        to_del = [k for k in self._last_crossing_time if k[0] == track_id]
        for k in to_del:
            del self._last_crossing_time[k]

    def cleanup(self, active_track_ids):
        stale = [tid for tid in self._last_side if tid not in active_track_ids]
        for tid in stale:
            del self._last_side[tid]
        stale_time_keys = [k for k in self._last_crossing_time if k[0] not in active_track_ids]
        for k in stale_time_keys:
            del self._last_crossing_time[k]


def load_lines_for_camera(line_config, camera_id):
    """
    Build Line objects for a given camera from the parsed lines.json
    config. Supports a per-camera key, falling back to "default".
    """
    camera_lines = line_config.get(camera_id, line_config.get("default", []))
    lines = []
    for entry in camera_lines:
        lines.append(Line(
            name=entry["name"],
            point_a_norm=entry["point_a"],
            point_b_norm=entry["point_b"],
            side_a_label=entry.get("side_a_label", "SIDE A"),
            side_b_label=entry.get("side_b_label", "SIDE B"),
        ))
    return lines
