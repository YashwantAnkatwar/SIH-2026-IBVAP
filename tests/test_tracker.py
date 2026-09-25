"""
Unit tests for tracker.py — track lifecycle, speed estimation, staleness.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from tracker import Tracker


def _det(track_id, x, y, class_name="person"):
    return {
        "track_id": track_id, "class_id": 0, "class_name": class_name,
        "confidence": 0.9, "box": (x - 5, y - 5, x + 5, y + 5), "center": (x, y),
    }


def test_new_track_is_flagged_is_new():
    tracker = Tracker()
    enriched = tracker.update([_det(1, 10, 10)], timestamp=0.0)
    assert enriched[0]["is_new"] is True
    assert enriched[0]["age_seconds"] == 0.0


def test_existing_track_is_not_new_and_ages():
    tracker = Tracker()
    tracker.update([_det(1, 10, 10)], timestamp=0.0)
    enriched = tracker.update([_det(1, 10, 10)], timestamp=2.0)
    assert enriched[0]["is_new"] is False
    assert enriched[0]["age_seconds"] == 2.0


def test_speed_estimation_reflects_movement():
    tracker = Tracker()
    tracker.update([_det(1, 0, 0)], timestamp=0.0)
    # Move 100px in 1 second -> should register nonzero speed.
    enriched = tracker.update([_det(1, 100, 0)], timestamp=1.0)
    assert enriched[0]["speed"] > 0


def test_stationary_track_has_zero_speed():
    tracker = Tracker()
    tracker.update([_det(1, 50, 50)], timestamp=0.0)
    enriched = tracker.update([_det(1, 50, 50)], timestamp=1.0)
    assert enriched[0]["speed"] == 0.0


def test_prune_stale_removes_old_tracks():
    tracker = Tracker(stale_after_seconds=2.0)
    tracker.update([_det(1, 10, 10)], timestamp=0.0)
    stale = tracker.prune_stale(timestamp=5.0)
    assert stale == [1]
    assert tracker.active_track_ids() == set()


def test_active_tracks_not_pruned_prematurely():
    tracker = Tracker(stale_after_seconds=5.0)
    tracker.update([_det(1, 10, 10)], timestamp=0.0)
    stale = tracker.prune_stale(timestamp=2.0)
    assert stale == []
    assert tracker.active_track_ids() == {1}


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} tracker tests passed.")
