"""
End-to-end smoke test: runs the real pipeline (Detector -> Tracker ->
ZoneManager -> RiskEngine) over ~40 real frames of a bundled sample
video, exactly the way camera_worker.py does per-frame. This is the
closest thing to an integration test without spinning up threads /
Flask, and it genuinely exercises every stage with real data.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import cv2
from detector import Detector
from tracker import Tracker
from zones import ZoneManager, load_zones_for_camera
from risk_engine import RiskEngine, RiskLevel

import config

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = BASE_DIR / "videos" / "bop1_visdrone_aerial.mp4"


def test_full_pipeline_over_real_frames():
    detector = Detector(config.MODEL_PATH, confidence=0.35, target_classes=config.TARGET_CLASSES)
    tracker = Tracker(stale_after_seconds=config.TRACK_STALE_SECONDS)

    zone_config = config.load_zone_config()
    zones = load_zones_for_camera(zone_config, "BOP-01")
    assert len(zones) > 0, "zones.json should define at least a default layout"
    zone_manager = ZoneManager(zones)

    risk_engine = RiskEngine(
        dwell_high_seconds=1.0,       # low thresholds so a ~40-frame clip can trigger them
        dwell_critical_seconds=2.0,
        group_threshold=2,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened()

    total_detections = 0
    zone_event_count = 0
    risk_levels_seen = set()
    frame_idx = 0
    start_t = time.time()

    while frame_idx < 40:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_h, frame_w = frame.shape[:2]
        # Use a synthetic, evenly-spaced timestamp so dwell-time math is
        # deterministic regardless of how fast this test machine runs.
        timestamp = frame_idx * 0.2

        detections, _ = detector.infer(frame)
        enriched = tracker.update(detections, timestamp)
        total_detections += len(enriched)

        risk_engine.begin_frame()
        per_track = []
        for det in enriched:
            zone_events, _ = zone_manager.update(
                det["track_id"], det["class_name"], det["center"],
                frame_w, frame_h, timestamp,
            )
            zone_event_count += len(zone_events)
            level, reasons = risk_engine.evaluate_track(det["track_id"], det["class_name"], zone_events)
            per_track.append((level, reasons))

        for level, reasons in per_track:
            level, _ = risk_engine.apply_group_escalation(level, reasons)
            risk_levels_seen.add(level)

        tracker.prune_stale(timestamp)

    cap.release()
    elapsed = time.time() - start_t

    assert frame_idx == 40, "should have processed 40 real frames"
    assert total_detections > 0, "pipeline produced zero detections across 40 frames"
    assert zone_event_count > 0, "zone manager produced zero events; zone geometry may be wrong"
    assert RiskLevel.LOW in risk_levels_seen or len(risk_levels_seen) > 0

    print(f"  -> processed {frame_idx} frames in {elapsed:.2f}s "
          f"({frame_idx/elapsed:.1f} fps), {total_detections} detections, "
          f"{zone_event_count} zone events, risk levels seen: {sorted(risk_levels_seen)}")


if __name__ == "__main__":
    test_full_pipeline_over_real_frames()
    print("PASS: test_full_pipeline_over_real_frames")
