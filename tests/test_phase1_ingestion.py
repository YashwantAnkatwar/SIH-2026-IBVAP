"""
test_phase1_ingestion.py

Focused test suite for SIH 2026 Phase 1 requirements:
- Camera configuration metadata and per-camera feature flags
- VideoSource opening, decoding, properties, and rewind looping
- Unavailable source handling without crashes
- Clean resource release
- Bounded latest-frame buffering (zero unbounded memory queue)
- Multi-camera CameraManager initialization
- Multi-camera concurrency and clean thread shutdown
"""

import os
import sys
import threading
import time
from pathlib import Path

# Add app to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "app"))

import config
from camera import VideoSource
from camera_worker import CameraWorker
from camera_manager import CameraManager


def test_camera_configuration_metadata():
    """Requirement A: Verify the three logical CCTV channels and their metadata."""
    assert "BOP-01" in config.CAMERAS, "BOP-01 must be configured"
    assert "BOP-02" in config.CAMERAS, "BOP-02 must be configured"
    assert "BOP-03" in config.CAMERAS, "BOP-03 must be configured"

    # BOP-01: Aerial/Day surveillance
    bop1 = config.CAMERAS["BOP-01"]
    assert "Aerial" in bop1["purpose"], f"Expected Aerial in BOP-01 purpose, got: {bop1['purpose']}"
    assert "visdrone" in bop1["source"].lower(), f"Expected VisDrone footage for BOP-01, got: {bop1['source']}"
    assert bop1["anpr_enabled"] is False, "ANPR should be disabled for aerial drone"
    assert bop1["face_enabled"] is False, "Face should be disabled for aerial drone"

    # BOP-02: Night/thermal surveillance
    bop2 = config.CAMERAS["BOP-02"]
    assert "Night" in bop2["purpose"] or "thermal" in bop2["purpose"].lower(), f"Expected Night/thermal in BOP-02 purpose: {bop2['purpose']}"
    assert "kaist" in bop2["source"].lower() or "night" in bop2["source"].lower(), f"Expected KAIST footage for BOP-02: {bop2['source']}"
    assert bop2.get("night_enabled") is True, "Night enabled should be True on BOP-02"

    # BOP-03: Checkpoint surveillance with ANPR and Face enabled by configuration
    bop3 = config.CAMERAS["BOP-03"]
    assert "Checkpoint" in bop3["purpose"], f"Expected Checkpoint in BOP-03 purpose: {bop3['purpose']}"
    assert bop3["anpr_enabled"] is True, "BOP-03 must have ANPR enabled by configuration"
    assert bop3["face_enabled"] is True, "BOP-03 must have Face recognition enabled by configuration"


def test_camera_source_env_override():
    """Requirement E: Verify environment variable overrides work cleanly without hardcoded paths."""
    env_key = "IBVAP_CAMERA_BOP-TEST"
    try:
        # Override with custom path
        os.environ[env_key] = "custom_test_stream.mp4"
        resolved = config._camera_source("BOP-TEST", "default.mp4")
        assert resolved == str(config.PROJECT_ROOT / "custom_test_stream.mp4")

        # Fallback to default when unset
        del os.environ[env_key]
        fallback = config._camera_source("BOP-TEST", "default.mp4")
        assert fallback == str(config.VIDEO_DIR / "default.mp4")
    finally:
        if env_key in os.environ:
            del os.environ[env_key]


def test_video_source_open_decode_and_properties():
    """Requirement B: Verify VideoSource can open files, report properties, and decode frames."""
    sample_path = str(config.VIDEO_DIR / "bop3_indian_checkpoint_traffic.mp4")
    source = VideoSource(sample_path)

    assert source.open() is True, f"Failed to open {sample_path}"
    assert source.is_opened() is True

    w = source.width
    h = source.height
    fps = source.source_fps
    assert w > 0, f"Expected positive width, got {w}"
    assert h > 0, f"Expected positive height, got {h}"
    assert fps > 0.0, f"Expected positive FPS, got {fps}"

    ok, frame = source.read()
    assert ok is True, "Failed to read initial frame"
    assert frame is not None
    assert frame.shape[0] == h
    assert frame.shape[1] == w

    source.release()
    assert source.is_opened() is False


def test_video_source_rewind_looping():
    """Requirement B: Verify VideoSource rewind() loops back to frame 0 cleanly."""
    sample_path = str(config.VIDEO_DIR / "bop3_indian_checkpoint_traffic.mp4")
    source = VideoSource(sample_path)
    assert source.open() is True

    # Read several frames
    for _ in range(5):
        ok, frame = source.read()
        assert ok is True

    # Rewind to loop
    assert source.rewind() is True, "rewind() should succeed"
    ok, frame_rewound = source.read()
    assert ok is True, "Failed to read after rewind"
    assert frame_rewound is not None

    source.release()


def test_video_source_unavailable_handling():
    """Requirement B: Ensure VideoSource handles missing/unavailable sources without crashing."""
    bad_source = VideoSource("/nonexistent/video_file_does_not_exist.mp4")
    assert bad_source.open() is False
    assert bad_source.is_opened() is False

    ok, frame = bad_source.read()
    assert ok is False
    assert frame is None

    # Should not raise exception
    bad_source.release()


def test_bounded_frame_buffering():
    """Requirement C: Ensure bounded/latest-frame buffer avoids memory leaks or unbounded queues."""
    worker = CameraWorker(
        camera_id="TEST-BUFFER",
        source="dummy.mp4",
        model_path=str(config.MODEL_PATH),
        confidence=0.35,
    )

    # Initially empty
    assert worker.get_latest_jpeg() is None

    # Store frame 1
    frame1 = b"JPEG_FRAME_1_BYTES"
    worker._set_latest_jpeg(frame1)
    assert worker.get_latest_jpeg() == frame1

    # Overwrite with frame 2: buffer replaces, does not queue up
    frame2 = b"JPEG_FRAME_2_BYTES"
    worker._set_latest_jpeg(frame2)
    assert worker.get_latest_jpeg() == frame2


def test_multi_camera_manager_initialization():
    """Requirement A & C: CameraManager initializes all 3 channels with per-camera metadata."""
    manager = CameraManager(
        camera_config=config.CAMERAS,
        model_path=config.MODEL_PATH,
        confidence=config.CONFIDENCE_THRESHOLD,
    )

    assert len(manager.workers) == 3
    assert "BOP-01" in manager.workers
    assert "BOP-02" in manager.workers
    assert "BOP-03" in manager.workers

    w1 = manager.get_worker("BOP-01")
    w2 = manager.get_worker("BOP-02")
    w3 = manager.get_worker("BOP-03")

    assert w1.anpr_enabled is False
    assert w2.anpr_enabled is False
    assert w3.anpr_enabled is True, "BOP-03 worker must inherit anpr_enabled=True"
    assert w3.face_enabled is True, "BOP-03 worker must inherit face_enabled=True"

    # Status dictionary should include name and purpose
    full_status = manager.get_full_status()
    assert full_status["BOP-01"]["purpose"] == "Aerial/Day surveillance"
    assert full_status["BOP-02"]["purpose"] == "Night/thermal surveillance"
    assert full_status["BOP-03"]["purpose"] == "Checkpoint surveillance"


def test_multi_camera_worker_concurrency_and_clean_shutdown():
    """Requirement C & G: Test multi-camera concurrency, non-blocking execution, and clean shutdown."""
    workers = []
    for cam_id in ["BOP-01", "BOP-02", "BOP-03"]:
        cfg = config.CAMERAS[cam_id]
        worker = CameraWorker(
            camera_id=cam_id,
            source=cfg["source"],
            model_path=str(config.MODEL_PATH),
            confidence=0.35,
            anpr_enabled=cfg.get("anpr_enabled"),
            face_enabled=cfg.get("face_enabled"),
        )
        workers.append(worker)

    # Start all 3 workers concurrently
    for w in workers:
        w.start()

    # Verify they run in distinct background threads
    thread_names = set()
    for w in workers:
        assert w.running is True
        assert w.thread is not None
        assert w.thread.is_alive() is True
        thread_names.add(w.thread.name)

    assert len(thread_names) == 3, "Each worker must execute on its own distinct thread"

    # Allow workers to initialize and read at least one frame
    time.sleep(2.0)

    # Clean shutdown
    for w in workers:
        w.stop()

    # Verify clean thread termination with zero orphan threads
    for w in workers:
        w.thread.join(timeout=4.0)
        assert w.thread.is_alive() is False, f"Worker {w.camera_id} thread failed to terminate"
        assert w.status == "OFFLINE"


if __name__ == "__main__":
    test_functions = [
        test_camera_configuration_metadata,
        test_camera_source_env_override,
        test_video_source_open_decode_and_properties,
        test_video_source_rewind_looping,
        test_video_source_unavailable_handling,
        test_bounded_frame_buffering,
        test_multi_camera_manager_initialization,
        test_multi_camera_worker_concurrency_and_clean_shutdown,
    ]

    print("=" * 60)
    print("PHASE 1: MULTI-CAMERA STREAM INGESTION & PIPELINE TESTS")
    print("=" * 60)

    for fn in test_functions:
        fn()
        print(f"PASS: {fn.__name__}")

    print("=" * 60)
    print(f"ALL {len(test_functions)} PHASE 1 TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
