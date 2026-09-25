"""
manual_anpr_agh_check.py

NOT part of the automated suite (tests/run_all.py) — this is a one-off,
honest, real-data validation script, not a synthetic unit test. It runs
the actual ANPR pipeline against the actual AGH parking-lot video and
prints exactly what came out: real detections, real localizations, real
OCR reads, real aggregated plates. Nothing here is asserted to "pass";
it's meant to be read by a human to judge real-world pipeline behavior
on real footage, honestly reported, misreads included.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import cv2

from detector import Detector
from anpr.pipeline import ANPRPipeline, VEHICLE_CLASSES

VIDEO_PATH = Path(__file__).resolve().parent.parent / "videos" / "anpr_agh_parking.mp4"
MODEL_PATH = Path(__file__).resolve().parent.parent / "app" / "yolo26n.pt"


def main():
    print(f"Loading YOLO model from {MODEL_PATH} ...")
    detector = Detector(model_path=MODEL_PATH, confidence=0.35)

    anpr = ANPRPipeline()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Could not open {VIDEO_PATH}")
        return

    frame_idx = 0
    stable_results = []
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        timestamp = frame_idx / 25.0  # known source fps

        detections, _result = detector.infer(frame)
        for det in detections:
            if det["class_name"] not in VEHICLE_CLASSES:
                continue
            global_track_id = f"AGH:{det['track_id']}"
            result = anpr.process("AGH", global_track_id, frame, det["box"], timestamp)
            if result is not None:
                stable_results.append(result)
                print(
                    f"[frame {frame_idx:4d} / t={timestamp:5.1f}s] "
                    f"STABLE PLATE  track={result.vehicle_track_id:>10}  "
                    f"text='{result.plate_text}'  confidence={result.confidence:.1f}"
                )

    elapsed = time.time() - t0
    cap.release()

    print("\n" + "=" * 70)
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"Frames where process() was called for a vehicle track: {anpr.frames_processed}")
    print(f"Plate localization hits (cascade or contour found a candidate): {anpr.localization_hits}")
    print(f"Frames with a usable OCR read (post length/format filtering): {anpr.ocr_reads}")
    print(f"Distinct vehicle tracks with a STABLE plate result: {len(stable_results)}")
    for r in stable_results:
        print(f"  - track {r.vehicle_track_id}: '{r.plate_text}' (confidence {r.confidence:.1f})")
    print("=" * 70)
    print(
        "NOTE: this video (AGH parking dataset) has European-style "
        "plates, not Indian plates. This run validates that the ANPR "
        "PIPELINE MECHANISM works end-to-end on real footage — it does "
        "NOT validate accuracy on Indian plates, which need Indian "
        "footage to test against."
    )


if __name__ == "__main__":
    main()
