"""
Unit tests for the ANPR pipeline (app/anpr/) — plate localization
geometry, OCR mechanism, and temporal aggregation. Uses small synthetic
images (rendered rectangles/text) so these run fast and deterministically
without needing real vehicle footage; real-footage validation lives
separately in tests/manual_anpr_agh_check.py (not part of this
automated suite, since it needs real video + takes minutes to run).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import cv2
import numpy as np

from anpr.plate_localizer import PlateLocalizer
from anpr.ocr import read_plate
from anpr.aggregator import PlateTrackAggregator, MIN_AGREEING_READS, MIN_CONFIDENCE


def _synthetic_vehicle_with_plate(text="MH12AB1234"):
    """A plain gray rectangle standing in for a vehicle crop, with a
    light plate-shaped rectangle + dark text in its lower half — enough
    for the contour-based localizer fallback to find geometrically,
    without needing a real photo."""
    img = np.full((300, 500, 3), 90, dtype=np.uint8)  # vehicle body
    # Plate: light rectangle in the lower-middle area.
    cv2.rectangle(img, (150, 220), (350, 270), (230, 230, 230), -1)
    cv2.rectangle(img, (150, 220), (350, 270), (0, 0, 0), 2)  # plate border helps contour detection
    cv2.putText(img, text, (160, 258), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def test_localizer_finds_plate_shaped_region_via_contour_fallback():
    localizer = PlateLocalizer()
    img = _synthetic_vehicle_with_plate()
    candidate = localizer.locate(img)
    assert candidate is not None
    x1, y1, x2, y2 = candidate.box
    # The found box should meaningfully overlap the real plate region
    # (150-350, 220-270) — not exact pixel match, just "found roughly there".
    assert x1 < 350 and x2 > 150
    assert y1 < 270 and y2 > 220


def test_localizer_returns_none_on_blank_image():
    localizer = PlateLocalizer()
    blank = np.full((200, 300, 3), 128, dtype=np.uint8)
    assert localizer.locate(blank) is None


def test_localizer_returns_none_on_empty_crop():
    localizer = PlateLocalizer()
    assert localizer.locate(np.zeros((0, 0, 3), dtype=np.uint8)) is None
    assert localizer.locate(None) is None


def test_ocr_reads_real_rendered_text_mechanism():
    """This is a real Tesseract OCR call on a real rendered image — it
    is not guaranteed to be pixel-perfect (that's the honest nature of
    OCR), but it should recognize SOME plausible alphanumeric text with
    non-zero confidence, proving the OCR mechanism actually runs rather
    than being stubbed."""
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "MH12AB1234", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
    result = read_plate(img)
    assert result.text != ""
    assert result.confidence > 0
    assert result.text.isalnum()
    # Should get most characters right even if not 100%.
    assert result.text[:2] == "MH"


def test_ocr_returns_empty_on_blank_image():
    blank = np.full((100, 300, 3), 255, dtype=np.uint8)
    result = read_plate(blank)
    assert result.text == ""
    assert result.confidence == 0.0


def test_ocr_returns_empty_on_none_or_tiny_crop():
    assert read_plate(None).text == ""
    assert read_plate(np.zeros((0, 0, 3), dtype=np.uint8)).text == ""


def test_aggregator_does_not_stabilize_on_single_read():
    agg = PlateTrackAggregator()
    agg.observe("track1", "ABC1234", 80.0)
    assert agg.get_stable_unreported("track1") is None


def test_aggregator_stabilizes_after_enough_agreeing_reads():
    agg = PlateTrackAggregator()
    for _ in range(MIN_AGREEING_READS):
        agg.observe("track1", "ABC1234", MIN_CONFIDENCE + 10)
    result = agg.get_stable_unreported("track1")
    assert result is not None
    text, conf = result
    assert text == "ABC1234"
    assert conf >= MIN_CONFIDENCE


def test_aggregator_does_not_stabilize_below_confidence_threshold():
    agg = PlateTrackAggregator()
    for _ in range(MIN_AGREEING_READS + 2):
        agg.observe("track1", "ABC1234", MIN_CONFIDENCE - 20)
    assert agg.get_stable_unreported("track1") is None


def test_aggregator_reports_stable_result_only_once():
    agg = PlateTrackAggregator()
    for _ in range(MIN_AGREEING_READS):
        agg.observe("track1", "ABC1234", MIN_CONFIDENCE + 10)
    first = agg.get_stable_unreported("track1")
    assert first is not None
    # No new observation, same stable text -> already reported -> None.
    second = agg.get_stable_unreported("track1")
    assert second is None


def test_aggregator_re_reports_when_stable_text_changes():
    agg = PlateTrackAggregator()
    for _ in range(MIN_AGREEING_READS):
        agg.observe("track1", "ABC1234", MIN_CONFIDENCE + 10)
    assert agg.get_stable_unreported("track1") is not None
    # A new, different plate majority for the same track (e.g. tracker
    # ID reused for a different vehicle) should produce a new stable event.
    for _ in range(MIN_AGREEING_READS + 1):
        agg.observe("track1", "XYZ9999", MIN_CONFIDENCE + 10)
    result = agg.get_stable_unreported("track1")
    assert result is not None
    assert result[0] == "XYZ9999"


def test_aggregator_tracks_are_independent():
    agg = PlateTrackAggregator()
    for _ in range(MIN_AGREEING_READS):
        agg.observe("track1", "ABC1234", MIN_CONFIDENCE + 10)
    assert agg.get_stable_unreported("track2") is None  # never observed


def test_aggregator_forget_clears_state():
    agg = PlateTrackAggregator()
    agg.observe("track1", "ABC1234", 80.0)
    agg.forget("track1")
    assert "track1" not in agg.active_track_keys()


def test_indian_hsrp_plate_formats():
    from anpr.ocr import is_indian_hsrp
    # Valid Indian formats
    assert is_indian_hsrp("DL01AB1234")
    assert is_indian_hsrp("MH12DE1432")
    assert is_indian_hsrp("PB08C9999")
    assert is_indian_hsrp("JK02BB0001")
    assert is_indian_hsrp("22BH1234AA")
    # Invalid formats
    assert not is_indian_hsrp("INVALID123")
    assert not is_indian_hsrp("ZZ99AA1234")  # ZZ is not an Indian state
    assert not is_indian_hsrp("12345")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} ANPR pipeline tests passed.")
