"""
Unit tests for lines.py — virtual line crossing geometry and direction
detection. No YOLO/model dependency, runs fast.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from lines import Line, LineCrossingDetector, load_lines_for_camera


def test_vertical_line_splits_left_right():
    # Vertical line at x=0.5 of a 100x100 frame -> x=50.
    line = Line("boundary", [0.5, 0.0], [0.5, 1.0], "LEFT", "RIGHT")
    left_side = line.side_of(20, 50, 100, 100)
    right_side = line.side_of(80, 50, 100, 100)
    assert left_side != right_side
    assert left_side != 0 and right_side != 0


def test_no_crossing_reported_on_first_sighting():
    detector = LineCrossingDetector([Line("boundary", [0.5, 0.0], [0.5, 1.0])])
    events = detector.update("t1", (20, 50), 100, 100)
    assert events == []  # nothing to compare against yet


def test_crossing_detected_when_side_flips():
    detector = LineCrossingDetector([Line("boundary", [0.5, 0.0], [0.5, 1.0], "APPROACH", "RESTRICTED")])
    detector.update("t1", (20, 50), 100, 100)   # left side
    events = detector.update("t1", (80, 50), 100, 100)  # now right side
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "LINE_CROSSING"
    assert ev["line"] == "boundary"
    assert ev["direction"] in ("A_TO_B", "B_TO_A")
    assert {ev["from_label"], ev["to_label"]} == {"APPROACH", "RESTRICTED"}


def test_no_crossing_while_staying_on_same_side():
    detector = LineCrossingDetector([Line("boundary", [0.5, 0.0], [0.5, 1.0])])
    detector.update("t1", (20, 50), 100, 100)
    events = detector.update("t1", (25, 55), 100, 100)
    assert events == []


def test_direction_is_consistent_and_reversible():
    line = Line("boundary", [0.5, 0.0], [0.5, 1.0], "APPROACH", "RESTRICTED")
    detector = LineCrossingDetector([line])
    detector.update("t1", (20, 50), 100, 100)
    forward = detector.update("t1", (80, 50), 100, 100)[0]

    detector2 = LineCrossingDetector([line])
    detector2.update("t2", (80, 50), 100, 100)
    backward = detector2.update("t2", (20, 50), 100, 100)[0]

    assert forward["direction"] != backward["direction"]
    assert forward["from_label"] == backward["to_label"]
    assert forward["to_label"] == backward["from_label"]


def test_multiple_tracks_are_independent():
    detector = LineCrossingDetector([Line("boundary", [0.5, 0.0], [0.5, 1.0])])
    detector.update("t1", (20, 50), 100, 100)
    detector.update("t2", (80, 50), 100, 100)
    # t1 crosses right, t2 stays put
    events_t1 = detector.update("t1", (80, 50), 100, 100)
    events_t2 = detector.update("t2", (81, 50), 100, 100)
    assert len(events_t1) == 1
    assert len(events_t2) == 0


def test_cleanup_and_forget_track_drop_state():
    detector = LineCrossingDetector([Line("boundary", [0.5, 0.0], [0.5, 1.0])])
    detector.update("t1", (20, 50), 100, 100)
    detector.forget_track("t1")
    # After forgetting, the next sighting is treated as "first sighting" again.
    events = detector.update("t1", (80, 50), 100, 100)
    assert events == []

    detector.update("t2", (20, 50), 100, 100)
    detector.cleanup(active_track_ids=set())
    events = detector.update("t2", (80, 50), 100, 100)
    assert events == []


def test_load_lines_for_camera_falls_back_to_default():
    config = {
        "default": [{"name": "l1", "point_a": [0, 0], "point_b": [1, 1]}],
        "CAM-SPECIFIC": [{"name": "l2", "point_a": [0, 1], "point_b": [1, 0]}],
    }
    default_lines = load_lines_for_camera(config, "UNKNOWN-CAM")
    assert len(default_lines) == 1 and default_lines[0].name == "l1"

    specific_lines = load_lines_for_camera(config, "CAM-SPECIFIC")
    assert len(specific_lines) == 1 and specific_lines[0].name == "l2"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} line-crossing tests passed.")
