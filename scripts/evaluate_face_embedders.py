"""
evaluate_face_embedders.py

Reproducible, honest evaluation of the LBP embedder (the original
weak baseline) vs. the new pretrained deep FaceNet/VGGFace2 embedder,
on the SAME real ChokePoint P2E_S5_C1 footage the 12/84 (~14%) baseline
number was measured on. This script does not use synthetic images
anywhere; every embedding below comes from a real face crop extracted
from the real video by the actual production Detector + FaceDetector
code (app/detector.py, app/face/face_detector.py).

WHAT THIS SCRIPT DOES
----------------------
1. ONE detection+tracking pass over the clip (the only expensive YOLO
   pass) collecting every quality-passing face crop, per track.
2. Picks the top-N tracks by face-observation count and treats each as
   a distinct real identity (ChokePoint subjects, referenced only by
   the tracker's numeric track ID -- never by name).
3. For each identity: splits its crops chronologically in half.
   First half -> enrolled into a fresh in-memory gallery. Second half
   -> held out, NEVER used for enrollment.
4. For BOTH embedders (LBP, then FaceNet/VGGFace2 deep), on the exact
   same crops:
     a. SCENARIO A -- reproduces the original single-identity baseline
        methodology exactly (only the single largest track enrolled,
        held-out recognition rate reported) -- this is the number the
        12/84 figure came from.
     b. SCENARIO B -- a fuller multi-identity evaluation: every
        top-N track enrolled simultaneously, per-identity held-out
        recognition rate, and confusions across different identities
        (a real false-accept check that Scenario A cannot show, since
        it only ever has one identity in the gallery).
     c. Genuine/impostor pair similarity distributions (mean, stdev,
        min, max) and, at each embedder's actual configured threshold,
        false-accept / false-reject counts computed directly from
        those pair distributions -- not asserted, computed.
5. Prints a side-by-side LBP vs. deep comparison table.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
-------------------------------------------
- It does not tune the threshold to make either embedder's demo number
  look better; the thresholds evaluated are the ones actually
  configured in config.py (FACE_MATCH_THRESHOLD_LBP /
  FACE_MATCH_THRESHOLD_DEEP), i.e. the ones the running application
  actually uses.
- It does not claim general face-recognition accuracy from one clip of
  one dataset. ChokePoint P2E_S5_C1 is a portal-camera walkthrough
  sequence: fairly controlled indoor lighting, subjects walking toward
  a fixed camera. It genuinely does exercise pose/scale change (as
  subjects approach the camera and turn), which is exactly the
  weakness this change targets, but a single clip is not a substitute
  for a full face-verification benchmark. See the final report for
  this caveat stated in full.

USAGE
-----
    python3 scripts/evaluate_face_embedders.py
"""

import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import cv2
import numpy as np

import config
from detector import Detector
from face.face_detector import FaceDetector
from face.embedder import Embedder
from face.embedder_deep import DeepEmbedder
from face.gallery import FaceGallery

VIDEO_PATH = PROJECT_ROOT / "videos" / "face_chokepoint_p2e_s5_c1.mp4"
MODEL_PATH = PROJECT_ROOT / "app" / "yolo26n.pt"

TOP_N_IDENTITIES = 5  # how many distinct real tracks to use as "identities" for Scenario B


def collect_face_crops():
    print(f"Loading YOLO model from {MODEL_PATH} ...")
    detector = Detector(model_path=MODEL_PATH, confidence=0.35, target_classes={"person"})
    face_det = FaceDetector()

    print("Single tracking pass over the real ChokePoint clip, collecting face crops per track...")
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


def split_enroll_heldout(crops):
    crops = sorted(crops, key=lambda c: c[0])
    half = len(crops) // 2
    if half == 0:
        return crops, []
    return crops[:half], crops[half:]


def build_gallery(embedder_name, identities_enroll_crops):
    """identities_enroll_crops: dict[identity_name] -> list of (frame_idx, crop)"""
    embedder = Embedder() if embedder_name == "lbp" else DeepEmbedder()
    gallery = FaceGallery(PROJECT_ROOT / "data" / f"_eval_gallery_{embedder_name}_unused", embedder=embedder)
    # In-memory only: don't touch disk / the real gallery/ directory.
    gallery._identities = {}
    for identity, crops in identities_enroll_crops.items():
        for _, crop in crops:
            gallery.enroll_from_crop(identity, crop)
    return gallery


def held_out_recognition_rate(gallery, identity, heldout_crops):
    if not heldout_crops:
        return None
    matches = [gallery.match(crop) for _, crop in heldout_crops]
    correct = sum(1 for m in matches if m.identity == identity)
    confidences = [m.confidence for m in matches]
    confusions = defaultdict(int)
    for m in matches:
        if m.identity != identity:
            confusions[m.identity] += 1
    return {
        "n": len(matches), "correct": correct,
        "rate": correct / len(matches),
        "mean_confidence": mean(confidences) if confidences else 0.0,
        "confusions": dict(confusions),
    }


def pair_stats(values):
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(values), "mean": mean(values), "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values),
    }


def genuine_impostor_similarity(embedder_name, identity_crops):
    """identity_crops: dict[identity] -> list of (frame_idx, crop) (ALL crops for that
    identity, enroll+heldout together) -- pairwise similarity within an identity is
    "genuine", across identities is "impostor". Returns (genuine_sims, impostor_sims)
    as SIMILARITY values (higher = more alike) on a comparable [-1, 1]-ish scale for
    both embedders, by converting LBP's Euclidean distance into 1 - distance/2."""
    embedder = Embedder() if embedder_name == "lbp" else DeepEmbedder()
    vecs_by_identity = {}
    for identity, crops in identity_crops.items():
        vecs_by_identity[identity] = [embedder.embed(crop) for _, crop in crops]

    def sim(a, b):
        if embedder_name == "lbp":
            return 1.0 - Embedder.distance(a, b) / 2.0
        return DeepEmbedder.similarity(a, b)

    genuine, impostor = [], []
    identities = list(vecs_by_identity.keys())
    for identity in identities:
        vecs = vecs_by_identity[identity]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                genuine.append(sim(vecs[i], vecs[j]))
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            for a in vecs_by_identity[identities[i]]:
                for b in vecs_by_identity[identities[j]]:
                    impostor.append(sim(a, b))
    return genuine, impostor


def far_frr_at_threshold(genuine_sims, impostor_sims, threshold, embedder_name):
    """threshold here is the embedder's configured `.compare()` threshold
    (smaller-is-match), converted to the same similarity scale used
    above so it lines up with genuine_sims/impostor_sims."""
    sim_threshold = 1.0 - threshold if embedder_name != "lbp" else 1.0 - threshold / 2.0
    false_rejects = sum(1 for s in genuine_sims if s < sim_threshold)
    false_accepts = sum(1 for s in impostor_sims if s >= sim_threshold)
    frr = false_rejects / len(genuine_sims) if genuine_sims else float("nan")
    far = false_accepts / len(impostor_sims) if impostor_sims else float("nan")
    return sim_threshold, false_accepts, false_rejects, far, frr


def measure_latency(embedder_name, crop, n=20):
    embedder = Embedder() if embedder_name == "lbp" else DeepEmbedder()
    embedder.embed(crop)  # warm-up (model already loaded once at import time for deep)
    start = time.perf_counter()
    for _ in range(n):
        embedder.embed(crop)
    elapsed = time.perf_counter() - start
    return (elapsed / n) * 1000.0  # ms/call


def main():
    crops_by_track = collect_face_crops()
    ranked = sorted(crops_by_track.items(), key=lambda kv: -len(kv[1]))
    print("\nTrack face-observation counts (top 10):")
    for tid, crops in ranked[:10]:
        print(f"  track {tid}: {len(crops)}")
    if not ranked:
        print("No quality-passing faces found in this clip. Stopping.")
        return

    top_tracks = ranked[:TOP_N_IDENTITIES]
    names = {tid: f"person_{tid}" for tid, _ in top_tracks}

    enroll_crops, heldout_crops, all_crops = {}, {}, {}
    for tid, crops in top_tracks:
        e, h = split_enroll_heldout(crops)
        enroll_crops[names[tid]] = e
        heldout_crops[names[tid]] = h
        all_crops[names[tid]] = crops

    biggest_tid, biggest_crops = top_tracks[0]
    biggest_name = names[biggest_tid]
    e_big, h_big = split_enroll_heldout(biggest_crops)

    results = {}
    for embedder_name in ("lbp", "facenet"):
        print(f"\n{'=' * 70}\nEVALUATING: {embedder_name.upper()}\n{'=' * 70}")

        # --- Scenario A: reproduce the original single-identity baseline ---
        gallery_a = build_gallery(embedder_name, {biggest_name: e_big})
        scenario_a = held_out_recognition_rate(gallery_a, biggest_name, h_big)
        print(f"Scenario A (single-identity baseline, track {biggest_tid} as '{biggest_name}'):")
        print(f"  enrolled on {len(e_big)} crops, held out {len(h_big)} crops")
        print(f"  {scenario_a['correct']}/{scenario_a['n']} held-out crops recognized "
              f"({scenario_a['rate']*100:.1f}%), mean confidence {scenario_a['mean_confidence']:.2f}")

        # --- Scenario B: multi-identity gallery, per-identity held-out + confusions ---
        gallery_b = build_gallery(embedder_name, enroll_crops)
        print(f"\nScenario B (multi-identity gallery, {len(top_tracks)} identities):")
        scenario_b = {}
        for name in names.values():
            r = held_out_recognition_rate(gallery_b, name, heldout_crops[name])
            scenario_b[name] = r
            if r is None:
                print(f"  {name}: no held-out crops (too few observations to split)")
                continue
            conf_str = f", confused as {r['confusions']}" if r["confusions"] else ""
            print(f"  {name}: {r['correct']}/{r['n']} recognized ({r['rate']*100:.1f}%){conf_str}")

        # --- Genuine/impostor similarity distributions + FAR/FRR at the configured threshold ---
        genuine, impostor = genuine_impostor_similarity(embedder_name, all_crops)
        g_stats, i_stats = pair_stats(genuine), pair_stats(impostor)
        threshold = config.FACE_MATCH_THRESHOLD_LBP if embedder_name == "lbp" else config.FACE_MATCH_THRESHOLD_DEEP
        sim_thr, fa, fr, far, frr = far_frr_at_threshold(genuine, impostor, threshold, embedder_name)
        print(f"\nGenuine pairs (n={g_stats['n']}): mean sim={g_stats['mean']:.3f} "
              f"std={g_stats['std']:.3f} min={g_stats['min']:.3f} max={g_stats['max']:.3f}")
        print(f"Impostor pairs (n={i_stats['n']}): mean sim={i_stats['mean']:.3f} "
              f"std={i_stats['std']:.3f} min={i_stats['min']:.3f} max={i_stats['max']:.3f}")
        print(f"Configured threshold (raw .compare() value): {threshold} "
              f"(similarity-scale equivalent: {sim_thr:.3f})")
        print(f"  False accepts: {fa}/{i_stats['n']} (FAR={far*100:.1f}%)")
        print(f"  False rejects: {fr}/{g_stats['n']} (FRR={frr*100:.1f}%)")

        latency_ms = measure_latency(embedder_name, biggest_crops[0][1])
        print(f"\nMean embed() latency on this machine: {latency_ms:.1f} ms/call")

        results[embedder_name] = {
            "scenario_a_rate": scenario_a["rate"], "scenario_a_n": scenario_a["n"],
            "scenario_a_correct": scenario_a["correct"],
            "genuine": g_stats, "impostor": i_stats,
            "far": far, "frr": frr, "latency_ms": latency_ms,
        }

    print(f"\n{'=' * 70}\nSUMMARY: LBP vs. DEEP (FaceNet/VGGFace2)\n{'=' * 70}")
    lbp, deep = results["lbp"], results["facenet"]
    print(f"{'Metric':<38}{'LBP':>15}{'Deep (FaceNet)':>18}")
    print(f"{'-'*38}{'-'*15}{'-'*18}")
    print(f"{'Scenario A held-out recognition':<38}"
          f"{lbp['scenario_a_correct']}/{lbp['scenario_a_n']:<12}"
          f"{deep['scenario_a_correct']}/{deep['scenario_a_n']:>16}")
    print(f"{'Genuine similarity (mean)':<38}{lbp['genuine']['mean']:>15.3f}{deep['genuine']['mean']:>18.3f}")
    print(f"{'Impostor similarity (mean)':<38}{lbp['impostor']['mean']:>15.3f}{deep['impostor']['mean']:>18.3f}")
    print(f"{'Genuine/impostor separation':<38}"
          f"{lbp['genuine']['mean']-lbp['impostor']['mean']:>15.3f}"
          f"{deep['genuine']['mean']-deep['impostor']['mean']:>18.3f}")
    print(f"{'False Accept Rate at threshold':<38}{lbp['far']*100:>14.1f}%{deep['far']*100:>17.1f}%")
    print(f"{'False Reject Rate at threshold':<38}{lbp['frr']*100:>14.1f}%{deep['frr']*100:>17.1f}%")
    print(f"{'Mean embed() latency (ms)':<38}{lbp['latency_ms']:>15.1f}{deep['latency_ms']:>18.1f}")


if __name__ == "__main__":
    main()
