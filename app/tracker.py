"""
tracker.py

The Detector already produces per-frame track IDs via ByteTrack (through
ultralytics' model.track()). This module builds on top of those raw IDs
to maintain higher-level, time-aware track state that the raw per-frame
IDs alone don't give us:

    - how long a track has existed
    - its recent position history
    - its estimated speed (pixels/second), used to flag rapid/suspicious
      movement
    - automatic cleanup of tracks that haven't been seen for a while
      (so memory doesn't grow unbounded across a long-running stream)

This keeps ID association logic (ByteTrack itself) separate from
track-lifecycle bookkeeping (this file), which is where most of the
"suspicious movement" / dwell-adjacent signals come from.
"""

from collections import deque


class TrackInfo:
    """Bookkeeping for a single tracked object across frames."""

    def __init__(self, track_id, class_name, timestamp, center):
        self.track_id = track_id
        self.class_name = class_name
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.frames_seen = 1
        self.history = deque(maxlen=30)  # recent (timestamp, x, y)
        self.history.append((timestamp, center[0], center[1]))
        self.speed = 0.0  # pixels / second, smoothed
        self.prev_speed = 0.0
        self.instant_speed = 0.0
        self.acceleration = 0.0  # (speed - prev_speed) / dt, px/s^2
        self.velocity_vector = (0.0, 0.0)  # (vx, vy) in px/s
        self.prev_velocity_vector = (0.0, 0.0)
        self.direction = "STATIONARY"
        self.movement_state = "STATIONARY"

    @property
    def age_seconds(self):
        return self.last_seen - self.first_seen

    def update(self, timestamp, center):
        if self.history:
            prev_t, prev_x, prev_y = self.history[-1]
            dt = timestamp - prev_t
            if dt > 0:
                dist = ((center[0] - prev_x) ** 2 + (center[1] - prev_y) ** 2) ** 0.5
                instant_speed = dist / dt
                vx_inst = (center[0] - prev_x) / dt
                vy_inst = (center[1] - prev_y) / dt

                self.prev_speed = self.speed
                self.prev_velocity_vector = self.velocity_vector
                self.instant_speed = round(instant_speed, 1)

                # Exponential smoothing to avoid single-frame camera jitter
                self.speed = (0.6 * self.speed) + (0.4 * instant_speed)
                vx_smooth = (0.6 * self.velocity_vector[0]) + (0.4 * vx_inst)
                vy_smooth = (0.6 * self.velocity_vector[1]) + (0.4 * vy_inst)
                self.velocity_vector = (round(vx_smooth, 1), round(vy_smooth, 1))

                # Acceleration calculation (rate of speed change)
                self.acceleration = round((self.speed - self.prev_speed) / dt, 1)

        # Classify movement state and cardinal direction
        if self.speed < 8.0:
            self.movement_state = "STATIONARY"
            self.direction = "STATIONARY"
        else:
            self.movement_state = "MOVING"
            vx, vy = self.velocity_vector
            abs_vx, abs_vy = abs(vx), abs(vy)
            if abs_vx < 0.414 * abs_vy:
                self.direction = "SOUTH" if vy > 0 else "NORTH"
            elif abs_vy < 0.414 * abs_vx:
                self.direction = "EAST" if vx > 0 else "WEST"
            else:
                if vx > 0 and vy > 0:
                    self.direction = "SOUTH_EAST"
                elif vx > 0 and vy < 0:
                    self.direction = "NORTH_EAST"
                elif vx < 0 and vy > 0:
                    self.direction = "SOUTH_WEST"
                else:
                    self.direction = "NORTH_WEST"

        self.history.append((timestamp, center[0], center[1]))
        self.last_seen = timestamp
        self.frames_seen += 1


class Tracker:
    """Maintains TrackInfo state for every track_id seen on one camera."""

    def __init__(self, stale_after_seconds=5.0):
        self.tracks = {}
        self.stale_after_seconds = stale_after_seconds

    def update(self, detections, timestamp):
        """
        Update track state from this frame's detections.

        `detections` is the list produced by Detector.infer().
        Returns the same list, with each dict enriched with:
            "age_seconds", "speed", "velocity", "velocity_vector",
            "direction", "movement_state", "frames_seen", "is_new"
        """
        enriched = []

        for det in detections:
            track_id = det["track_id"]
            is_new = track_id not in self.tracks

            if is_new:
                self.tracks[track_id] = TrackInfo(
                    track_id, det["class_name"], timestamp, det["center"]
                )
            else:
                self.tracks[track_id].update(timestamp, det["center"])

            info = self.tracks[track_id]

            enriched_det = dict(det)
            enriched_det["age_seconds"] = round(info.age_seconds, 2)
            enriched_det["speed"] = round(info.speed, 1)
            enriched_det["velocity"] = round(info.speed, 1)
            enriched_det["prev_velocity"] = round(info.prev_speed, 1)
            enriched_det["instant_speed"] = round(info.instant_speed, 1)
            enriched_det["acceleration"] = round(info.acceleration, 1)
            enriched_det["velocity_vector"] = info.velocity_vector
            enriched_det["prev_velocity_vector"] = info.prev_velocity_vector
            enriched_det["direction"] = info.direction
            enriched_det["movement_state"] = info.movement_state
            enriched_det["frames_seen"] = info.frames_seen
            enriched_det["history"] = list(info.history)
            enriched_det["is_new"] = is_new
            enriched.append(enriched_det)

        return enriched

    def prune_stale(self, timestamp):
        """Drop tracks that haven't been updated recently to bound memory use."""
        stale_ids = [
            tid for tid, info in self.tracks.items()
            if (timestamp - info.last_seen) > self.stale_after_seconds
        ]
        for tid in stale_ids:
            del self.tracks[tid]
        return stale_ids

    def active_track_ids(self):
        return set(self.tracks.keys())
