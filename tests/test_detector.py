"""
Integration test for detector.py using the real bundled YOLO model and a
real sample video frame. This is intentionally NOT mocked: it verifies
the model actually loads and produces genuine detections, per the "do
not fake results" requirement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import cv2
from detector import Detector

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "app" / "yolo26n.pt"
VIDEO_PATH = BASE_DIR / "videos" / "bop1_visdrone_aerial.mp4"


def test_model_loads():
    detector = Detector(MODEL_PATH, confidence=0.35, target_classes={"person"})
    assert detector.model is not None
    assert "person" in detector.class_names.values()


def test_detects_real_objects_in_sample_video():
    detector = Detector(MODEL_PATH, confidence=0.35, target_classes={"person", "car", "truck", "bus", "motorcycle"})
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    assert cap.isOpened(), "sample video failed to open"

    ok, frame = cap.read()
    assert ok, "failed to read first frame"

    detections, result = detector.infer(frame)
    cap.release()

    assert isinstance(detections, list)
    assert len(detections) > 0, "expected at least one real detection in video frame 1"
    for det in detections:
        assert det["class_name"] in {"person", "car", "truck", "bus", "motorcycle"}
        assert 0.0 <= det["confidence"] <= 1.0
        assert isinstance(det["track_id"], int)


def test_target_class_filtering_excludes_other_classes():
    # Restricting to a class that shouldn't appear in a road/checkpoint
    # scene should yield zero detections rather than silently returning
    # unfiltered results.
    detector = Detector(MODEL_PATH, confidence=0.35, target_classes={"giraffe"})
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    ok, frame = cap.read()
    cap.release()
    assert ok
    detections, _ = detector.infer(frame)
    assert detections == []


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} detector tests passed.")
