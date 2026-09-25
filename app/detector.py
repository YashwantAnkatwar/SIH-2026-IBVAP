"""
detector.py

Wraps the Ultralytics YOLO model: loading it once, running detection +
ByteTrack tracking on a frame, and parsing the raw result into simple
plain-Python dictionaries the rest of the pipeline can work with
(zones.py, risk_engine.py, events.py, the dashboard) without needing to
know anything about torch tensors or the Ultralytics API.
"""

from pathlib import Path
from ultralytics import YOLO


class Detector:
    """Loads a YOLO model once and runs detection + tracking on frames."""

    def __init__(self, model_path, confidence=0.35, tracker_config="bytetrack.yaml",
                 target_classes=None):
        self.model_path = str(model_path)
        self.confidence = confidence
        self.tracker_config = tracker_config

        # None / empty means "accept every class the model knows about".
        self.target_classes = set(target_classes) if target_classes else None

        if ("/" in self.model_path or "\\" in self.model_path) and not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names

    def infer(self, frame):
        """
        Run detection + ByteTrack tracking on a single frame.

        Returns a list of detection dicts:
            {
                "track_id": int,
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "box": (x1, y1, x2, y2),
                "center": (cx, cy),
            }
        and the raw ultralytics Result object (needed for .plot()).
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]
        detections = []

        boxes = result.boxes
        has_tracks = (
            boxes is not None
            and getattr(boxes, "is_track", False)
            and boxes.id is not None
        )

        if has_tracks:
            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().numpy()

            for track_id, box, class_id, conf in zip(
                track_ids, coordinates, class_ids, confidences
            ):
                class_name = self.class_names.get(class_id, str(class_id))

                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                detections.append({
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                    "center": center,
                })

        return detections, result
