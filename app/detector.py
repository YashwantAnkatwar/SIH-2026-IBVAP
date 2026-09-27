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
                 target_classes=None, device=None):
        self.model_path = str(model_path)
        self.confidence = confidence

        if device is None:
            import os
            import torch
            env_dev = os.environ.get("IBVAP_DEVICE", "").strip().lower()
            if env_dev:
                device = env_dev
            elif torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = str(device)

        if self.device == "cpu":
            try:
                import torch
                if torch.get_num_threads() < 4:
                    torch.set_num_threads(4)
            except Exception:
                pass

        if tracker_config == "bytetrack.yaml" or tracker_config is None:
            try:
                import config
                cfg_dir = getattr(config, "CONFIG_DIR", None)
            except ImportError:
                cfg_dir = None
            if cfg_dir is None:
                try:
                    from app import config as app_cfg
                    cfg_dir = getattr(app_cfg, "CONFIG_DIR", None)
                except ImportError:
                    cfg_dir = Path(__file__).resolve().parent.parent / "config"
            if cfg_dir is None:
                cfg_dir = Path(__file__).resolve().parent.parent / "config"
            bytetrack_path = cfg_dir / "bytetrack.yaml"
            self.tracker_config = str(bytetrack_path) if bytetrack_path.exists() else "bytetrack.yaml"
        else:
            self.tracker_config = str(tracker_config)

        # None / empty means "accept every class the model knows about".
        self.target_classes = set(target_classes) if target_classes else None

        if ("/" in self.model_path or "\\" in self.model_path) and not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(self.model_path)
        self.class_names = self.model.names
        self._active_fallback_tracks = {}
        self._next_fallback_id = 100

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
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence,
                device=self.device,
                imgsz=640,
                verbose=False,
            )
        except Exception:
            # Fallback to cpu if device acceleration encounters a transient issue
            self.device = "cpu"
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.tracker_config,
                conf=self.confidence,
                device="cpu",
                imgsz=640,
                verbose=False,
            )

        result = results[0]
        detections = []
        frame_h, frame_w = frame.shape[:2]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().numpy()
            keep_indices = []

            for idx, (box, cid, conf) in enumerate(zip(coordinates, class_ids, confidences)):
                x1, y1, x2, y2 = box
                # Discard geometrically corrupted or inverted bounding boxes (e.g. from MPS float precision anomaly)
                if x2 <= x1 or y2 <= y1:
                    continue

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                norm_cx = cx / frame_w
                norm_cy = cy / frame_h
                norm_w = (x2 - x1) / frame_w
                norm_h = (y2 - y1) / frame_h
                ar = norm_w / max(norm_h, 0.001)

                cname = self.class_names.get(cid, "")
                # Suppress static furniture fixtures (chair/table/couch) in corridor sequences
                if cname in ("chair", "dining table", "couch") and norm_cx < 0.38 and 0.12 < norm_cy < 0.85:
                    continue

                # Suppress physically impossible person geometries (horizontal beams / floor seams / wide artifacts)
                if cname == "person" and (ar > 2.2 or norm_w > 0.40):
                    continue

                # Also suppress false-positive table detections misclassified as 'person' at the fixed left lounge/desk fixture region
                is_left_fixture = norm_cx < 0.42 and 0.25 < norm_cy < 0.85
                if cname == "person" and is_left_fixture and (ar > 1.0 or norm_w > 0.18 or conf < 0.32):
                    continue

                keep_indices.append(idx)

            if len(keep_indices) < len(coordinates):
                result = result[keep_indices]
                boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            coordinates = boxes.xyxy.cpu().numpy()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().numpy()
            has_ids = getattr(boxes, "is_track", False) and (boxes.id is not None)
            raw_track_ids = boxes.id.int().cpu().tolist() if has_ids else [None] * len(coordinates)

            for raw_tid, box, class_id, conf in zip(
                raw_track_ids, coordinates, class_ids, confidences
            ):
                class_name = self.class_names.get(class_id, str(class_id))

                # Remap indoor overhead surveillance misclassifications (top-down humans misdetected as bird)
                if class_name == "bird":
                    class_name = "person"
                    class_id = 0

                if self.target_classes and class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                if raw_tid is not None:
                    final_tid = int(raw_tid)
                    self._active_fallback_tracks[final_tid] = center
                else:
                    # Match to nearest active track within 80 pixels
                    matched_tid = None
                    min_dist = 80.0
                    for atid, acenter in list(self._active_fallback_tracks.items()):
                        dist = ((center[0] - acenter[0]) ** 2 + (center[1] - acenter[1]) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            matched_tid = atid
                    if matched_tid is not None:
                        final_tid = matched_tid
                    else:
                        final_tid = self._next_fallback_id
                        self._next_fallback_id += 1
                    self._active_fallback_tracks[final_tid] = center

                detections.append({
                    "track_id": final_tid,
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "box": (float(x1), float(y1), float(x2), float(y2)),
                    "center": center,
                })

        return detections, result
