"""
pipeline.py

ANPRPipeline: the single object camera_worker.py calls, once per
vehicle detection per frame.

Full chain, matching the brief's required pipeline exactly:

    vehicle detection (already done upstream by Detector/YOLO)
        -> vehicle tracking (already done upstream by ByteTrack)
        -> plate localization (plate_localizer.py)
        -> plate crop
        -> preprocessing (ocr.py)
        -> OCR (ocr.py, real Tesseract)
        -> plate text
        -> associate with vehicle track (aggregator.py)
        -> camera + timestamp + vehicle ID + plate number
        -> stable result returned to caller, which raises the
           event/alert and (optionally) saves the evidence crop

No fake plate numbers are ever produced: every character in every
result traces back to an actual Tesseract OCR call on an actual pixel
crop taken from the frame the caller passed in.
"""

from dataclasses import dataclass
from typing import Optional

from anpr.plate_localizer import PlateLocalizer
from anpr.ocr import read_plate, validate_indian_plate
from anpr.aggregator import PlateTrackAggregator

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@dataclass
class ANPRResult:
    camera_id: str
    vehicle_track_id: str  # global, e.g. "BOP-01:17"
    plate_text: str
    confidence: float
    timestamp: float
    evidence_crop: "any"  # BGR numpy array of the plate crop, for evidence saving
    plate_box: tuple = ()
    supporting_frames: int = 1
    is_indian: bool = False
    state_name: str = ""


class ANPRPipeline:
    def __init__(self):
        self._localizer = PlateLocalizer()
        self._aggregator = PlateTrackAggregator()
        # Per-track running count of frames processed without a plate
        # localization hit, purely for diagnostics/tests — not used for
        # any pass/fail decision.
        self.frames_processed = 0
        self.localization_hits = 0
        self.ocr_reads = 0

    def process(self, camera_id, global_track_id, frame, box, timestamp) -> Optional[ANPRResult]:
        """
        box: (x1, y1, x2, y2) in the full frame's pixel coordinates, from
        the detector, for a detection whose class_name is a vehicle
        class. Returns an ANPRResult only on the frame where a track's
        plate FIRST becomes stable (see aggregator.py) — i.e. at most
        once per vehicle, not once per frame. Returns None otherwise,
        including every frame where the vehicle simply doesn't yet have
        a stable read.
        """
        self.frames_processed += 1

        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return None
        vehicle_crop = frame[y1:y2, x1:x2]

        # A plate sits in the lower half of a vehicle's bounding box
        # (front/rear bumper area) in essentially every camera angle
        # this system will see; restricting the search region there
        # measurably cuts false positives from windows/mirrors/shadows
        # in the upper half without ever excluding a real plate.
        vh = vehicle_crop.shape[0]
        search_region = vehicle_crop[vh // 2:, :]
        region_y_offset = vh // 2

        candidate = self._localizer.locate(search_region)
        plate_crop = None
        plate_full_box = ()
        if candidate is not None:
            self.localization_hits += 1
            px1, py1, px2, py2 = candidate.box
            plate_crop = search_region[py1:py2, px1:px2]
            plate_full_box = (
                x1 + px1,
                y1 + region_y_offset + py1,
                x1 + px2,
                y1 + region_y_offset + py2,
            )
            ocr_result = read_plate(plate_crop)
            if ocr_result.text:
                self.ocr_reads += 1
            self._aggregator.observe(
                global_track_id,
                ocr_result.text,
                ocr_result.confidence,
                box=plate_full_box,
                timestamp=timestamp,
            )
        else:
            self._aggregator.observe(global_track_id, "", 0.0, timestamp=timestamp)

        stable = self._aggregator.get_stable_unreported(global_track_id)
        if stable is None:
            return None

        text, confidence = stable
        details = self._aggregator.get_stable_details(global_track_id)
        supporting_frames = details[2] if details else 1
        stored_box = details[3] if details and details[3] else plate_full_box

        val = validate_indian_plate(text)

        return ANPRResult(
            camera_id=camera_id,
            vehicle_track_id=global_track_id,
            plate_text=text,
            confidence=confidence,
            timestamp=timestamp,
            evidence_crop=plate_crop,
            plate_box=stored_box,
            supporting_frames=supporting_frames,
            is_indian=val.is_valid,
            state_name=val.state_name or "",
        )

    def forget_track(self, global_track_id):
        self._aggregator.forget(global_track_id)

    def current_plate_for(self, global_track_id):
        """Read-only lookup for the dashboard: current stable plate for a
        track, whether or not it has already been reported as an event."""
        return self._aggregator.get_current(global_track_id)
