#!/usr/bin/env python3
"""
scripts/enroll_member.py

CLI tool to enroll personnel / team members into the IBVAP authorized
face gallery (data/gallery/<identity_name>/).

Usage:
  # Enroll from an existing photo
  python scripts/enroll_member.py --name "Capt_Yashwant" --image /path/to/photo.jpg

  # List currently enrolled identities
  python scripts/enroll_member.py --list

  # Test recognition against enrolled gallery
  python scripts/enroll_member.py --verify /path/to/test_face.jpg
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root and app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import cv2
import config
from face.face_detector import FaceDetector
from face.gallery import FaceGallery


def list_gallery(gallery_dir: Path):
    print(f"\n--- IBVAP Authorized Face Gallery ({gallery_dir}) ---")
    if not gallery_dir.exists():
        print("Gallery directory does not exist yet.")
        return
    identities = [d for d in sorted(gallery_dir.iterdir()) if d.is_dir()]
    if not identities:
        print("Gallery is currently empty (0 enrolled identities).")
        return
    for idx, d in enumerate(identities, 1):
        images = [f for f in d.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        print(f"  {idx}. {d.name} ({len(images)} reference image{'s' if len(images) != 1 else ''})")
    print(f"Total: {len(identities)} identity profiles.\n")


def enroll_identity(name: str, image_path: str, gallery_dir: Path):
    clean_name = name.strip().replace(" ", "_")
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"ERROR: Image file '{image_path}' not found.")
        sys.exit(1)

    frame = cv2.imread(str(img_file))
    if frame is None:
        print(f"ERROR: Failed to read image from '{image_path}'. Ensure it is a valid JPEG/PNG.")
        sys.exit(1)

    detector = FaceDetector()
    detections = detector.detect(frame)

    person_dir = gallery_dir / clean_name
    person_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{int(time.time())}_{img_file.stem}.jpg"
    out_path = person_dir / out_filename

    if detections:
        # Sort by confidence or size, take largest face
        best_det = max(detections, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))
        x1, y1, x2, y2 = best_det.box
        # Add slight margin around face
        h, w = frame.shape[:2]
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.15)
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(w, x2 + margin_x)
        crop_y2 = min(h, y2 + margin_y)
        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.imwrite(str(out_path), face_crop)
        print(f"[+] Face detected (box: {best_det.box}, conf: {best_det.confidence:.2f}).")
    else:
        # Fallback: image is already a cropped face portrait
        cv2.imwrite(str(out_path), frame)
        print("[!] No face bounding box isolated by Haar cascade; enrolled entire image as crop portrait.")

    # Re-verify by loading gallery
    gallery = FaceGallery(gallery_dir)
    enrolled = gallery.enrolled_identities
    print(f"\n[SUCCESS] Successfully enrolled '{clean_name}' into IBVAP Face Gallery!")
    print(f"  Target file : {out_path}")
    print(f"  Active identities in gallery: {enrolled}")


def verify_image(image_path: str, gallery_dir: Path):
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"ERROR: Image file '{image_path}' not found.")
        sys.exit(1)
    frame = cv2.imread(str(img_file))
    if frame is None:
        print("ERROR: Failed to decode image.")
        sys.exit(1)

    gallery = FaceGallery(gallery_dir)
    detector = FaceDetector()
    detections = detector.detect(frame)

    print(f"\n--- Verification Scan: {image_path} ---")
    if not detections:
        match = gallery.match(frame)
        print(f"Direct match result: Identity='{match.identity}', Known={match.is_known}, Conf={match.confidence:.2f}, Dist={match.distance:.3f}")
        return

    for idx, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det.box
        crop = frame[y1:y2, x1:x2]
        match = gallery.match(crop)
        print(f"Face #{idx}: Identity='{match.identity}' | Known={match.is_known} | Confidence={match.confidence:.2f} | Distance={match.distance:.3f}")


def main():
    parser = argparse.ArgumentParser(description="IBVAP Facial Gallery Personnel Enrollment CLI")
    parser.add_argument("--name", type=str, help="Name or callsign of person to enroll (e.g. 'Capt_Yashwant')")
    parser.add_argument("--image", type=str, help="Path to input photo containing face")
    parser.add_argument("--list", action="store_true", help="List all enrolled identities in the gallery")
    parser.add_argument("--verify", type=str, help="Verify a probe face image against gallery")
    parser.add_argument("--gallery-dir", type=str, default=str(config.GALLERY_DIR), help="Custom gallery directory path")
    args = parser.parse_args()

    gallery_dir = Path(args.gallery_dir)

    if args.list:
        list_gallery(gallery_dir)
    elif args.verify:
        verify_image(args.verify, gallery_dir)
    elif args.name and args.image:
        enroll_identity(args.name, args.image, gallery_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
