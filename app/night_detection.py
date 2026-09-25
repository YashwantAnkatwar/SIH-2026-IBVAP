"""
night_detection.py

A real, defensible night/low-light detector based on scene luminance —
not a per-video label and not guesswork. Computes the mean brightness
of the (grayscale) frame and compares it to a configurable threshold.
This is a genuine, if simple, computer-vision measurement: a real
nighttime CCTV frame is measurably darker than a real daytime one, and
this class measures exactly that, on every frame, live.

Includes light temporal smoothing (a short rolling window) so a single
transient bright/dark frame — a headlight sweep, a cloud passing, mild
sensor noise — doesn't flip the scene's night/day classification back
and forth every frame, which would make "night + movement" alerts
noisy rather than meaningful.
"""

from collections import deque

import cv2
import numpy as np

DEFAULT_LUMINANCE_THRESHOLD = 70.0  # 0-255 grayscale mean; tune per camera/site in config
SMOOTHING_WINDOW = 15


class NightDetector:
    def __init__(self, threshold: float = DEFAULT_LUMINANCE_THRESHOLD, window: int = SMOOTHING_WINDOW):
        self.threshold = threshold
        self._recent_luminance = deque(maxlen=window)

    def update(self, frame) -> dict:
        """Call once per frame. Returns a dict with the raw and smoothed
        luminance and the current is_night classification, so callers
        (and the dashboard) can show *why* a frame was classified as
        night rather than just a bare boolean."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_luminance = float(np.mean(gray))
        self._recent_luminance.append(raw_luminance)
        smoothed = sum(self._recent_luminance) / len(self._recent_luminance)

        return {
            "raw_luminance": raw_luminance,
            "smoothed_luminance": smoothed,
            "is_night": smoothed < self.threshold,
            "threshold": self.threshold,
        }
