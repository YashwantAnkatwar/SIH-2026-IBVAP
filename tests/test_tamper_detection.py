"""
test_tamper_detection.py

Unit tests for app/tamper_detector.py:
- Normal natural frame is not tampered
- Uniform black frame triggers OCCLUSION
- Uniform white frame triggers BLINDING
- Extreme blur frame triggers DEFOCUS
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import numpy as np
from tamper_detector import TamperDetector


def test_normal_frame_is_not_tampered():
    detector = TamperDetector(history_len=5)
    # Natural frame with texture
    rng = np.random.default_rng(42)
    for _ in range(5):
        frame = rng.integers(40, 200, size=(100, 100, 3), dtype=np.uint8)
        res = detector.update(frame)
    assert not res["is_tampered"]
    assert res["tamper_type"] == "NONE"


def test_black_frame_triggers_occlusion():
    detector = TamperDetector(history_len=5)
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(5):
        res = detector.update(black_frame)
    assert res["is_tampered"]
    assert res["tamper_type"] == "OCCLUSION"
    assert "obstructed" in res["reason"].lower()


def test_white_frame_triggers_blinding():
    detector = TamperDetector(history_len=5)
    white_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    for _ in range(5):
        res = detector.update(white_frame)
    assert res["is_tampered"]
    assert res["tamper_type"] == "BLINDING"
    assert "blinded" in res["reason"].lower()


def test_solid_flat_color_triggers_defocus():
    detector = TamperDetector(history_len=5)
    # Solid flat gray frame (no texture/edges)
    flat_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    for _ in range(5):
        res = detector.update(flat_frame)
    assert res["is_tampered"]
    assert res["tamper_type"] == "DEFOCUS"


if __name__ == "__main__":
    test_normal_frame_is_not_tampered()
    print("PASS: test_normal_frame_is_not_tampered")
    test_black_frame_triggers_occlusion()
    print("PASS: test_black_frame_triggers_occlusion")
    test_white_frame_triggers_blinding()
    print("PASS: test_white_frame_triggers_blinding")
    test_solid_flat_color_triggers_defocus()
    print("PASS: test_solid_flat_color_triggers_defocus")
    print("\n4 tamper detection tests passed.")
