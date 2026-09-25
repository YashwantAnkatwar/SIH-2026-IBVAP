"""
Tests that the system fails gracefully rather than crashing, per the
"error handling" requirement: missing video files, invalid sources,
and a bad model path should all be handled without raising.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from camera import VideoSource


def test_missing_video_file_does_not_open():
    source = VideoSource("/nonexistent/path/does_not_exist.mp4")
    assert source.open() is False
    assert source.is_opened() is False
    source.release()  # should not raise even though nothing was opened


def test_read_on_unopened_source_returns_false_not_exception():
    source = VideoSource("/nonexistent/path/does_not_exist.mp4")
    source.open()
    ok, frame = source.read()
    assert ok is False
    assert frame is None


def test_detector_reports_error_status_on_bad_model_path(tmp_path=Path("/tmp")):
    from camera_worker import CameraWorker
    worker = CameraWorker(
        camera_id="TEST-BAD-MODEL",
        source=str(Path(__file__).resolve().parent.parent / "videos" / "bop1_visdrone_aerial.mp4"),
        model_path="/nonexistent/model.pt",
        confidence=0.35,
    )
    worker.start()
    worker.thread.join(timeout=10)
    assert worker.status == "ERROR"
    assert worker.error_message  # a human-readable message was captured
    worker.stop()


def test_empty_detection_list_does_not_break_zone_manager():
    from zones import Zone, ZoneManager
    zone = Zone("z", "restricted", [[0, 0], [1, 0], [1, 1], [0, 1]])
    manager = ZoneManager([zone])
    # No detections this frame at all -- nothing to update, should just
    # be a no-op for any tracker bookkeeping.
    manager.cleanup(active_track_ids=set())


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} failure-handling tests passed.")
