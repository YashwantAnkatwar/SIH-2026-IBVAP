"""
Unit tests for night_detection.py — luminance-based night/day
classification and temporal smoothing. Uses synthetic bright/dark
frames since none of the bundled demo footage is actual nighttime
footage (worth being honest about: this validates the MECHANISM, not
accuracy against real night CCTV footage, which needs real night
footage to test).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import numpy as np

from night_detection import NightDetector


def _flat_frame(brightness):
    return np.full((100, 100, 3), brightness, dtype=np.uint8)


def test_bright_frame_is_not_night():
    detector = NightDetector(threshold=70.0)
    state = detector.update(_flat_frame(200))
    assert state["is_night"] is False
    assert state["raw_luminance"] > 150


def test_dark_frame_is_night():
    detector = NightDetector(threshold=70.0)
    state = detector.update(_flat_frame(20))
    assert state["is_night"] is True
    assert state["raw_luminance"] < 30


def test_smoothing_prevents_single_frame_flicker():
    detector = NightDetector(threshold=70.0, window=10)
    # Establish a stable bright (day) baseline.
    for _ in range(10):
        detector.update(_flat_frame(200))
    # One single dark frame (e.g. a passing shadow) should NOT flip the
    # smoothed classification to night on its own.
    state = detector.update(_flat_frame(0))
    assert state["is_night"] is False


def test_sustained_darkness_does_flip_classification():
    detector = NightDetector(threshold=70.0, window=10)
    for _ in range(10):
        detector.update(_flat_frame(200))
    # Many consecutive dark frames (sustained low light) SHOULD flip it.
    state = None
    for _ in range(10):
        state = detector.update(_flat_frame(10))
    assert state["is_night"] is True


def test_threshold_is_reported_for_operator_transparency():
    detector = NightDetector(threshold=55.0)
    state = detector.update(_flat_frame(100))
    assert state["threshold"] == 55.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} night detection tests passed.")
