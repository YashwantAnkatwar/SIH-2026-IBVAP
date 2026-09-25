"""
Unit tests for CameraWorker's operator controls (Phase 18): pause/resume
and live confidence adjustment. These test the worker's own state
machine directly (no video decode / YOLO needed - that end-to-end path
is covered by test_pipeline_smoke.py and the live run), so they're fast
and don't require torch/ultralytics to be installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from camera_worker import CameraWorker


def _make_worker():
    return CameraWorker(
        camera_id="TEST-01",
        source=str(Path(__file__).resolve().parent.parent / "videos" / "bop1_visdrone_aerial.mp4"),
        model_path=str(Path(__file__).resolve().parent.parent / "app" / "yolo26n.pt"),  # never loaded in these tests
        confidence=0.35,
    )


def test_worker_starts_unpaused_with_configured_confidence():
    worker = _make_worker()
    assert worker.paused is False
    assert worker.confidence == 0.35


def test_pause_sets_flag_true():
    worker = _make_worker()
    worker.pause()
    assert worker.paused is True


def test_resume_sets_flag_false():
    worker = _make_worker()
    worker.pause()
    worker.resume()
    assert worker.paused is False


def test_pause_resume_is_idempotent():
    worker = _make_worker()
    worker.pause()
    worker.pause()
    assert worker.paused is True
    worker.resume()
    worker.resume()
    assert worker.paused is False


def test_set_confidence_updates_live_value():
    worker = _make_worker()
    worker.set_confidence(0.6)
    assert worker.confidence == 0.6
    worker.set_confidence("0.2")  # dashboard passes JSON floats; be lenient
    assert worker.confidence == 0.2


def test_pause_and_confidence_are_independent():
    worker = _make_worker()
    worker.pause()
    worker.set_confidence(0.7)
    assert worker.paused is True
    assert worker.confidence == 0.7
    worker.resume()
    assert worker.confidence == 0.7  # resuming doesn't reset confidence


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} camera-worker control tests passed.")
