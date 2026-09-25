"""
manual_e2e_check.py

Runs the REAL CameraWorker (not a mock) against the real BOP-04 (ANPR)
and BOP-05 (face) demo videos for a bounded duration, then inspects the
real, persistent event_store to report what actually got recorded.
Not part of the automated suite — a one-off honest end-to-end check.
"""
import os
os.environ["IBVAP_ANPR_ENABLED"] = "true"
os.environ["IBVAP_FACE_ENABLED"] = "true"

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config
import shared
from camera_worker import CameraWorker

RUN_SECONDS = int(sys.argv[1]) if len(sys.argv) > 1 else 45


def main():
    print(f"Gallery identities at startup: {shared.face_gallery.enrolled_identities if shared.face_gallery else None}")

    workers = []
    for cam_id in ("BOP-04", "BOP-05"):
        w = CameraWorker(cam_id, config.CAMERAS[cam_id]["source"], config.MODEL_PATH, config.CONFIDENCE_THRESHOLD)
        t = w.start()
        workers.append(w)
        print(f"Started worker for {cam_id} (source={config.CAMERAS[cam_id]['source']})")

    print(f"Running for {RUN_SECONDS}s...")
    time.sleep(RUN_SECONDS)

    for w in workers:
        w.stop()
    time.sleep(2)

    print("\n=== Real events recorded in the persistent event store ===")
    rows = shared.event_store.query(limit=500)
    anpr_rows = [r for r in rows if r["event_type"] == "ANPR_PLATE_READ"]
    face_rows = [r for r in rows if r["event_type"] == "FACE_RECOGNITION"]
    print(f"Total events in store: {len(rows)}")
    print(f"ANPR_PLATE_READ events: {len(anpr_rows)}")
    for r in anpr_rows:
        print(f"  {r}")
    print(f"FACE_RECOGNITION events: {len(face_rows)}")
    for r in face_rows[:15]:
        print(f"  {r}")


if __name__ == "__main__":
    main()
