"""
plate_localizer.py

Finds a candidate license-plate rectangle inside a cropped vehicle
image. Two methods are combined, both fully offline (no network / no
downloaded weights beyond what ships inside opencv-python):

1. Haar cascade (`haarcascade_russian_plate_number.xml`, bundled with
   OpenCV). Despite the name, this cascade is trained on the general
   rectangular-plate-with-dark-text-on-light-background pattern common
   to European-style plates and works reasonably as a first pass on
   footage like the AGH parking dataset. It will NOT reliably find
   Indian plates (different aspect ratio / mounting conventions) —
   see the ANPR skill notes / README for what this does and does not
   validate.
2. Classical contour-based fallback: edge detection + contour
   filtering by aspect ratio, used when the cascade finds nothing.
   This is the standard "textbook" ANPR localization approach and is
   intentionally simple/explainable rather than a trained detector.

Both methods only ever look for candidate rectangles; neither method
invents or hallucinates a plate when none is visibly present — if
nothing passes the geometric filters, `locate()` returns None.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# A real license plate (any country) is always noticeably wider than
# tall. These bounds are deliberately generous (covers EU ~4.7:1,
# Indian ~4.3:1 and US ~2:1 plates) so we don't reject a real plate
# on shape alone; OCR + the caller's confidence filtering do the rest.
MIN_ASPECT_RATIO = 1.8
MAX_ASPECT_RATIO = 6.0
MIN_PLATE_WIDTH_FRACTION = 0.10   # relative to the vehicle crop width
MAX_PLATE_HEIGHT_FRACTION = 0.5   # relative to the vehicle crop height


@dataclass
class PlateCandidate:
    box: Tuple[int, int, int, int]   # x1, y1, x2, y2, relative to the vehicle crop
    method: str                       # "cascade" | "contour"
    quality: float                    # 0..1, geometric plausibility only (not OCR confidence)


class PlateLocalizer:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load plate cascade from {cascade_path}")

    def locate(self, vehicle_crop) -> Optional[PlateCandidate]:
        """
        vehicle_crop: BGR image of just the vehicle (already cropped from
        the full frame using the detector's bounding box).

        Returns the single best PlateCandidate, or None if nothing in the
        crop plausibly looks like a plate.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None

        h, w = vehicle_crop.shape[:2]
        if h < 40 or w < 65:
            return None

        for method_fn in (self._locate_via_cascade, self._locate_via_morphology, self._locate_via_contours):
            candidate = method_fn(vehicle_crop)
            if candidate is not None:
                bx1, by1, bx2, by2 = candidate.box
                bw = bx2 - bx1
                bh = by2 - by1
                if bw >= 30 and bh >= 10:
                    aspect = bw / float(bh)
                    if MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO:
                        return candidate

        return None

    # ------------------------------------------------------------------

    def _locate_via_cascade(self, crop) -> Optional[PlateCandidate]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 12),
        )
        if len(detections) == 0:
            return None

        h, w = crop.shape[:2]
        best = None
        best_score = -1.0
        for (x, y, pw, ph) in detections:
            if ph == 0:
                continue
            aspect = pw / float(ph)
            if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
                continue
            if pw < MIN_PLATE_WIDTH_FRACTION * w:
                continue
            # Plates on a vehicle's rear/front sit in the lower portion
            # of the vehicle's bounding box far more often than the top.
            position_bonus = y / float(h)
            score = position_bonus + (pw * ph) / float(w * h)
            if score > best_score:
                best_score = score
                best = (x, y, x + pw, y + ph)

        if best is None:
            return None
        return PlateCandidate(box=best, method="cascade", quality=min(1.0, 0.6 + best_score / 2))

    def _locate_via_morphology(self, crop) -> Optional[PlateCandidate]:
        """
        Sobel-X vertical edge extraction + horizontal morphological closing.
        Specifically robust for Indian High Security Registration Plates
        (HSRP) where character strokes create dense vertical edges inside a
        horizontal plate strip.
        """
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_x = np.absolute(grad_x)
        min_v, max_v = np.min(grad_x), np.max(grad_x)
        if max_v > min_v:
            grad_x = 255 * ((grad_x - min_v) / (max_v - min_v))
        grad_x = grad_x.astype(np.uint8)

        blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_close)

        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean)
        clean = cv2.erode(clean, None, iterations=1)
        clean = cv2.dilate(clean, None, iterations=2)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = -1.0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0:
                continue
            aspect = cw / float(ch)
            if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
                continue
            if cw < MIN_PLATE_WIDTH_FRACTION * w:
                continue
            if ch > MAX_PLATE_HEIGHT_FRACTION * h:
                continue
            area_frac = (cw * ch) / float(w * h)
            if not (0.003 <= area_frac <= 0.40):
                continue

            pos_bonus = y / float(h)
            score = pos_bonus * 1.5 + (cw * ch) / float(w * h)
            if score > best_score:
                best_score = score
                pad_x = int(cw * 0.04)
                pad_y = int(ch * 0.04)
                best = (
                    max(0, x - pad_x),
                    max(0, y - pad_y),
                    min(w, x + cw + pad_x),
                    min(h, y + ch + pad_y),
                )

        if best is None:
            return None
        return PlateCandidate(box=best, method="morphology", quality=min(1.0, 0.5 + best_score / 2))

    def _locate_via_contours(self, crop) -> Optional[PlateCandidate]:
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(gray, 30, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1.0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0:
                continue
            aspect = cw / float(ch)
            if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
                continue
            if cw < MIN_PLATE_WIDTH_FRACTION * w:
                continue
            if ch > MAX_PLATE_HEIGHT_FRACTION * h:
                continue
            area_fraction = (cw * ch) / float(w * h)
            if area_fraction < 0.005 or area_fraction > 0.35:
                continue
            # Rectangularity: how much of the bounding rect the contour's
            # own area actually fills (a real plate outline fills most of
            # its bounding rect; noisy edge junk usually does not).
            contour_area = cv2.contourArea(c)
            rectangularity = contour_area / float(cw * ch)
            if rectangularity < 0.25:
                continue

            position_bonus = y / float(h)
            score = rectangularity + position_bonus
            if score > best_score:
                best_score = score
                best = (x, y, x + cw, y + ch)

        if best is None:
            return None
        return PlateCandidate(box=best, method="contour", quality=min(1.0, best_score / 2))
