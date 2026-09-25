"""
face_detector.py

Finds a face inside a person crop using OpenCV's bundled Haar cascade
(fully offline — no model download). Applies basic quality filtering
so obviously unusable detections (too small, too blurry) never reach
the embedder — the brief explicitly requires "face quality filtering"
as its own pipeline stage, not just detection.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

MIN_FACE_SIZE = (24, 24)
MIN_FACE_SIDE_PX = 30          # after finding a candidate, reject tiny/far faces
BLUR_VARIANCE_THRESHOLD = 25.0  # Laplacian variance; below this = too blurry to trust


@dataclass
class FaceCandidate:
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2 relative to the person crop
    sharpness: float                 # Laplacian variance, higher = sharper


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load face cascade from {cascade_path}")

    def detect(self, person_crop) -> Optional[FaceCandidate]:
        if person_crop is None or person_crop.size == 0:
            return None

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE,
        )
        if len(faces) == 0:
            return None

        # A person crop should contain at most one face for our use case
        # (single tracked person); if the cascade fires multiple times,
        # take the largest box as the real face.
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return FaceCandidate(box=(x, y, x + w, y + h), sharpness=0.0)  # sharpness filled by quality_check

    def quality_check(self, face_crop) -> Optional[float]:
        """Returns the Laplacian-variance sharpness score if the crop
        passes minimum size + blur thresholds, else None (reject)."""
        if face_crop is None or face_crop.size == 0:
            return None
        h, w = face_crop.shape[:2]
        if h < MIN_FACE_SIDE_PX or w < MIN_FACE_SIDE_PX:
            return None
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < BLUR_VARIANCE_THRESHOLD:
            return None
        return sharpness
