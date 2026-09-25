"""
Unit tests for zones.py — point-in-polygon geometry and zone
entry/exit/dwell bookkeeping. No YOLO/model dependency, runs fast.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from zones import Zone, ZoneManager, point_in_polygon, load_zones_for_camera


def test_point_in_polygon_basic():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon(5, 5, square) is True
    assert point_in_polygon(15, 5, square) is False
    assert point_in_polygon(-1, -1, square) is False


def test_zone_contains_scales_normalized_polygon():
    zone = Zone("test_zone", "restricted", [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]])
    # Right half of a 100x100 frame should be inside; left half outside.
    assert zone.contains(75, 50, 100, 100) is True
    assert zone.contains(25, 50, 100, 100) is False


def test_zone_manager_enter_exit_dwell():
    zone = Zone("restricted_zone", "restricted", [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]])
    manager = ZoneManager([zone])

    t0 = time.time()

    # Track starts outside the zone.
    events, current = manager.update("t1", "person", (10, 50), 100, 100, timestamp=t0)
    assert events == []
    assert current == set()

    # Track moves inside -> ZONE_ENTER.
    events, current = manager.update("t1", "person", (75, 50), 100, 100, timestamp=t0 + 1)
    assert len(events) == 1 and events[0]["type"] == "ZONE_ENTER"
    assert current == {"restricted_zone"}

    # Still inside 4s later -> ZONE_DWELL with ~4s dwell.
    events, current = manager.update("t1", "person", (76, 50), 100, 100, timestamp=t0 + 5)
    assert events[0]["type"] == "ZONE_DWELL"
    assert 3.9 <= events[0]["dwell_seconds"] <= 4.1

    # Moves back out -> ZONE_EXIT with ~5s total dwell.
    events, current = manager.update("t1", "person", (10, 50), 100, 100, timestamp=t0 + 6)
    assert events[0]["type"] == "ZONE_EXIT"
    assert current == set()


def test_zone_violation_flag_for_disallowed_class():
    zone = Zone(
        "pedestrian_lane", "monitored",
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        allowed_classes=["person"],
    )
    manager = ZoneManager([zone])
    events, _ = manager.update("v1", "car", (50, 50), 100, 100, timestamp=time.time())
    assert events[0]["violation"] is True

    events, _ = manager.update("p1", "person", (50, 50), 100, 100, timestamp=time.time())
    assert events[0]["violation"] is False


def test_load_zones_for_camera_falls_back_to_default():
    config = {
        "default": [{"name": "z1", "type": "monitored", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
        "CAM-SPECIFIC": [{"name": "z2", "type": "restricted", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
    }
    default_zones = load_zones_for_camera(config, "UNKNOWN-CAM")
    assert len(default_zones) == 1 and default_zones[0].name == "z1"

    specific_zones = load_zones_for_camera(config, "CAM-SPECIFIC")
    assert len(specific_zones) == 1 and specific_zones[0].name == "z2"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} zone tests passed.")
