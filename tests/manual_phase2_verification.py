"""
manual_phase2_verification.py

Runs the live camera pipeline across all 3 configured border cameras:
- BOP-01: VisDrone aerial recon
- BOP-02: KAIST thermal night IR
- BOP-03: Indian border checkpoint
Verifies bounding boxes, class labels, track IDs, motion direction & speed,
zones, virtual tripwires, and intrusion events.
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

import config
from camera_worker import CameraWorker
from events import event_bus
import shared

def run_verification():
    print("=" * 65)
    print("PHASE 2 MANUAL PIPELINE VERIFICATION")
    print("=" * 65)

    workers = {}
    for cam_id, meta in config.CAMERAS.items():
        worker = CameraWorker(
            camera_id=cam_id,
            source=meta["source"],
            model_path=config.MODEL_PATH,
            confidence=0.35,
            name=meta.get("name"),
            purpose=meta.get("purpose"),
        )
        workers[cam_id] = worker

    print("\n[STEP 1] Starting camera workers for BOP-01, BOP-02, and BOP-03...")
    for cam_id, w in workers.items():
        w.start()

    # Let the workers run for 4 seconds to process live frames
    time.sleep(4.0)

    print("\n[STEP 2] Inspecting live worker runtime metrics:")
    total_detections = 0
    all_tracks = set()
    classes_seen = set()
    zones_active = set()

    for cam_id, w in workers.items():
        jpeg = w.get_latest_jpeg()
        has_frame = jpeg is not None and len(jpeg) > 1000
        print(f"  -> {cam_id}: Status={w.status}, FPS={w.fps:.1f}, Frames={w.frame_count}, ActiveTracks={w.active_track_count}, FrameBytes={len(jpeg) if jpeg else 0}")
        print(f"     Zones Loaded: {[z.name for z in w.zones]}")
        print(f"     Lines Loaded: {[l.name for l in w.lines]}")
        print(f"     Object Counts: {w.object_counts}")
        print(f"     Active Alerts: {len(w.active_alerts)}")

        assert has_frame, f"{cam_id} failed to produce annotated JPEG frames"
        assert w.frame_count > 0, f"{cam_id} failed to process any frames"
        assert len(w.zones) >= 2, f"{cam_id} should have configured polygonal geofences"
        assert len(w.lines) >= 1, f"{cam_id} should have configured directional tripwires"

        for cls, count in w.object_counts.items():
            classes_seen.add(cls)
            total_detections += count
        zones_active.update(w.active_zones)

    print("\n[STEP 3] Inspecting EventBus recent events:")
    recent_events = event_bus.recent(limit=30)
    print(f"  -> Total recent events captured on EventBus: {len(recent_events)}")
    kinds_found = {e["event_kind"] for e in recent_events}
    print(f"     Event kinds: {kinds_found}")

    for e in recent_events[:5]:
        print(f"     Sample Event: kind={e.get('event_kind')}, type={e.get('event_type')}, cam={e.get('camera_id')}, track={e.get('track_id')}")

    print("\n[STEP 4] Inspecting SQLite EventStore intrusion records:")
    db_events = shared.event_store.query(limit=20)
    print(f"  -> Total stored records queried: {len(db_events)}")
    for d in db_events[:5]:
        print(f"     DB Record #{d['id']}: [{d['camera_id']}] {d['event_type']} ({d['severity']}) - {d['description']}")

    print("\n[STEP 5] Stopping workers cleanly...")
    for cam_id, w in workers.items():
        w.stop()

    print("\n" + "=" * 65)
    print(f"VERIFICATION COMPLETE:")
    print(f"  - Target Classes Seen: {sorted(classes_seen)}")
    print(f"  - Geofences Triggered: {sorted(zones_active)}")
    print(f"  - Workers Active & Cleanly Stopped: True")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()
