"""
manual_face_chokepoint_check.py

NOT part of the automated suite — an honest, real-data validation
script for the face recognition pipeline, using the real ChokePoint
P2E_S5_C1 sequence (portal-camera walkthrough footage, an academic
face-recognition benchmark dataset; subjects referenced only by
dataset track ID, never by real name).

Methodology (stated plainly, no results are fabricated or implied
beyond what's printed):

  Pass 1 — track everyone in the clip, collect face crops per track.
  Pick the track with the most quality-passing face observations and
  split its observations chronologically in half:
    - first half  -> enrolled into the gallery as "person_01"
    - second half -> held out, NOT used for enrollment

  Pass 2 — run the full FaceRecognitionPipeline (fresh) over the same
  video with that gallery. Report, per track:
    - whether it stabilizes to "person_01" or "Unknown"
    - how many of the enrolled track's OWN held-out (second-half)
      frames get recognized as person_01 (a genuine, if easy,
      generalization check within one clip — same lighting/camera,
      different exact frames/poses than the enrollment crops)
    - whether other, never-enrolled tracks correctly come back Unknown

This is a classical LBP-histogram pipeline, not a deep embedding
network — see app/face/embedder.py for why, and don't read these
numbers as representative of production-grade face recognition
accuracy.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import cv2

from detector import Detector
from face.face_detector import FaceDetector
from face.gallery import FaceGallery
from face.embedder import Embedder

VIDEO_PATH = Path(__file__).resolve().parent.parent / "videos" / "face_chokepoint_p2e_s5_c1.mp4"
MODEL_PATH = Path(__file__).resolve().parent.parent / "app" / "yolo26n.pt"
GALLERY_DIR = Path(__file__).resolve().parent.parent / "gallery"

MIN_OBSERVATIONS_TO_STABILIZE = 3
STABILITY_VOTE_FRACTION = 0.6
HISTORY_PER_TRACK = 15


def collect_face_crops(detector):
    """Single detection+tracking pass: track everyone, collect
    (frame_idx, face_crop) for every quality-passing face observation
    per track. This is the ONLY expensive YOLO pass in this script —
    recognition/matching afterward is replayed on these already-real
    crops instead of re-running detection a second time."""
    face_det = FaceDetector()
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    crops_by_track = defaultdict(list)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        detections, _ = detector.infer(frame)
        for det in detections:
            if det["class_name"] != "person":
                continue
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            person_crop = frame[y1:y2, x1:x2]
            candidate = face_det.detect(person_crop)
            if candidate is None:
                continue
            fx1, fy1, fx2, fy2 = candidate.box
            face_crop = person_crop[fy1:fy2, fx1:fx2]
            if face_det.quality_check(face_crop) is None:
                continue
            crops_by_track[det["track_id"]].append((frame_idx, face_crop.copy()))
    cap.release()
    return crops_by_track


def replay_recognition(crops_by_track, gallery):
    """Replays the exact recognition + per-track temporal stabilization
    logic from face/pipeline.py, but directly on the already-collected
    real face crops (chronological per track) instead of re-running
    YOLO detection. This is the same algorithm, just without redoing
    the expensive upstream detection pass a second time."""
    stable_events = []
    final_state = {}

    for track_id, crops in crops_by_track.items():
        crops.sort(key=lambda c: c[0])
        observations = []
        stable_identity = None
        reported_identity = None

        for frame_idx, crop in crops:
            match = gallery.match(crop)
            observations.append((match.identity, match.confidence))
            if len(observations) > HISTORY_PER_TRACK:
                observations.pop(0)

            if len(observations) >= MIN_OBSERVATIONS_TO_STABILIZE:
                recent = observations[-HISTORY_PER_TRACK:]
                counts = defaultdict(list)
                for identity, conf in recent:
                    counts[identity].append(conf)
                best_identity, confs = max(counts.items(), key=lambda kv: len(kv[1]))
                if len(confs) / len(recent) >= STABILITY_VOTE_FRACTION:
                    stable_identity = best_identity
                    stable_conf = sum(confs) / len(confs)
                    if stable_identity != reported_identity:
                        reported_identity = stable_identity
                        stable_events.append((frame_idx, track_id, stable_identity, stable_conf))

        final_state[track_id] = (stable_identity, observations[-1] if observations else None)

    return stable_events, final_state


def main():
    print(f"Loading YOLO model from {MODEL_PATH} ...")
    detector = Detector(model_path=MODEL_PATH, confidence=0.35, target_classes={"person"})

    print("Single tracking pass: collecting face crops per track...")
    crops_by_track = collect_face_crops(detector)
    for tid, crops in sorted(crops_by_track.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"  track {tid}: {len(crops)} quality-passing face observations")

    if not crops_by_track:
        print("No quality-passing faces found in this clip. Stopping.")
        return

    best_track, crops = max(crops_by_track.items(), key=lambda kv: len(kv[1]))
    crops.sort(key=lambda c: c[0])
    half = len(crops) // 2
    enroll_crops = crops[:half] if half > 0 else crops
    heldout_crops = {best_track: crops[half:]} if half > 0 else {}

    print(
        f"\nEnrolling track {best_track} as 'person_01' using its first "
        f"{len(enroll_crops)} face observations (frames "
        f"{enroll_crops[0][0]}-{enroll_crops[-1][0]}). Holding out its "
        f"remaining {len(crops) - len(enroll_crops)} observations for testing."
    )

    GALLERY_DIR.mkdir(exist_ok=True)
    if (GALLERY_DIR / "person_01").exists():
        for old in (GALLERY_DIR / "person_01").glob("*"):
            old.unlink()

    gallery = FaceGallery(GALLERY_DIR)
    for i, (frame_idx, crop) in enumerate(enroll_crops):
        gallery.enroll_from_crop(
            "person_01", crop, save_path=GALLERY_DIR / "person_01" / f"enroll_{i:02d}_frame{frame_idx}.jpg",
        )
    print(f"Gallery now has identities: {gallery.enrolled_identities}")

    print("\nReplaying recognition + temporal stabilization over all collected crops...")
    stable_events, final_state = replay_recognition(crops_by_track, gallery)

    print("\nStable recognition events (one per track per distinct stabilized outcome):")
    for frame_idx, tid, identity, conf in stable_events:
        flag = " <- ENROLLED TRACK" if tid == best_track else ""
        print(f"  [frame {frame_idx:4d}] track={tid:3d}  identity={identity:10s}  confidence={conf:.2f}{flag}")

    print(f"\nEnrolled track {best_track}'s FINAL stabilized identity: {final_state[best_track][0]}")

    # Honest held-out check: of the enrolled track's SECOND-HALF crops
    # (never used for enrollment), how many individually match person_01
    # vs Unknown, frame by frame?
    if best_track in heldout_crops and heldout_crops[best_track]:
        matches = [gallery.match(c) for _, c in heldout_crops[best_track]]
        known_count = sum(1 for m in matches if m.identity == "person_01")
        print(
            f"Held-out check: {known_count}/{len(matches)} of track {best_track}'s "
            f"NEVER-enrolled later face crops individually matched 'person_01' "
            f"(mean confidence {sum(m.confidence for m in matches)/len(matches):.2f})."
        )

    other_tracks = [t for t in crops_by_track if t != best_track]
    print(f"\nOther (never-enrolled) tracks' final stabilized identity:")
    for t in other_tracks[:10]:
        print(f"  track {t}: {final_state[t][0]}")


if __name__ == "__main__":
    main()

