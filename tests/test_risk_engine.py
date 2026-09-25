"""
Unit tests for risk_engine.py — explainable rule-based risk scoring.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from risk_engine import RiskEngine, RiskLevel


def _zone_event(kind, zone="restricted_zone", zone_type="restricted", dwell=0.0, violation=False):
    return {
        "type": kind,
        "zone": zone,
        "zone_type": zone_type,
        "dwell_seconds": dwell,
        "class_name": "person",
        "violation": violation,
    }


def test_low_risk_when_no_zone_events():
    engine = RiskEngine()
    engine.begin_frame()
    level, reasons = engine.evaluate_track("t1", "person", [])
    assert level == RiskLevel.LOW
    assert reasons == []


def test_medium_on_restricted_zone_entry():
    engine = RiskEngine()
    engine.begin_frame()
    level, reasons = engine.evaluate_track("t1", "person", [_zone_event("ZONE_ENTER", dwell=0.0)])
    assert level == RiskLevel.MEDIUM
    assert reasons


def test_escalates_to_high_then_critical_with_dwell_time():
    engine = RiskEngine(dwell_high_seconds=5.0, dwell_critical_seconds=15.0)
    engine.begin_frame()
    level, _ = engine.evaluate_track("t1", "person", [_zone_event("ZONE_DWELL", dwell=6.0)])
    assert level == RiskLevel.HIGH

    engine.begin_frame()
    level, _ = engine.evaluate_track("t1", "person", [_zone_event("ZONE_DWELL", dwell=20.0)])
    assert level == RiskLevel.CRITICAL


def test_monitored_zone_alone_is_not_risky():
    engine = RiskEngine()
    engine.begin_frame()
    level, reasons = engine.evaluate_track(
        "t1", "person", [_zone_event("ZONE_ENTER", zone="pedestrian_lane", zone_type="monitored")]
    )
    assert level == RiskLevel.LOW
    assert reasons == []


def test_violation_in_monitored_zone_is_high():
    engine = RiskEngine()
    engine.begin_frame()
    level, reasons = engine.evaluate_track(
        "t1", "car",
        [_zone_event("ZONE_ENTER", zone="pedestrian_lane", zone_type="monitored", violation=True)],
    )
    assert level == RiskLevel.HIGH
    assert reasons


def test_violation_in_restricted_zone_is_critical():
    engine = RiskEngine()
    engine.begin_frame()
    level, reasons = engine.evaluate_track(
        "t1", "car",
        [_zone_event("ZONE_ENTER", zone="restricted_zone", zone_type="restricted", violation=True)],
    )
    assert level == RiskLevel.CRITICAL


def test_group_incursion_escalates_by_one_level():
    engine = RiskEngine(group_threshold=3)
    engine.begin_frame()

    results = []
    for tid in ("a", "b", "c"):
        level, reasons = engine.evaluate_track(tid, "person", [_zone_event("ZONE_ENTER", dwell=0.0)])
        results.append((tid, level, reasons))

    escalated = []
    for tid, level, reasons in results:
        level, reasons = engine.apply_group_escalation(level, reasons)
        escalated.append(level)

    # Each was MEDIUM alone; with 3 simultaneous occupants (>= threshold),
    # every one of them should escalate by exactly one level, to HIGH.
    assert all(level == RiskLevel.HIGH for level in escalated)


def test_group_incursion_does_not_fire_below_threshold():
    engine = RiskEngine(group_threshold=3)
    engine.begin_frame()

    level, reasons = engine.evaluate_track("a", "person", [_zone_event("ZONE_ENTER", dwell=0.0)])
    level, reasons = engine.apply_group_escalation(level, reasons)
    assert level == RiskLevel.MEDIUM  # only 1 occupant, below threshold of 3


def test_risklevel_max_ordering():
    assert RiskLevel.max(RiskLevel.LOW, RiskLevel.HIGH) == RiskLevel.HIGH
    assert RiskLevel.max(RiskLevel.CRITICAL, RiskLevel.MEDIUM) == RiskLevel.CRITICAL
    assert RiskLevel.max(RiskLevel.LOW, RiskLevel.LOW) == RiskLevel.LOW


def test_score_for_low_is_in_low_band():
    score, band = RiskEngine.score_for(RiskLevel.LOW, [])
    assert band == RiskLevel.LOW
    assert 0 <= score <= 24


def test_score_for_critical_is_in_critical_band():
    score, band = RiskEngine.score_for(RiskLevel.CRITICAL, ["reason one"])
    assert band == RiskLevel.CRITICAL
    assert 75 <= score <= 100


def test_score_increases_with_more_reasons_but_stays_in_band():
    score_one, _ = RiskEngine.score_for(RiskLevel.HIGH, ["r1"])
    score_many, _ = RiskEngine.score_for(RiskLevel.HIGH, ["r1", "r2", "r3", "r4"])
    assert score_many > score_one
    assert 50 <= score_one <= 74
    assert 50 <= score_many <= 74


def test_score_never_crosses_into_a_different_band():
    # Even with many reasons, a HIGH-level score must never reach 75+
    # (which would visually look like CRITICAL).
    score, _ = RiskEngine.score_for(RiskLevel.HIGH, ["r"] * 20)
    assert score < 75


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} risk engine tests passed.")
