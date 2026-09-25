"""
tests/test_phase4_behavioral_risk.py

Comprehensive Phase 4 Verification Suite:
- Dwell time / loitering warning & critical thresholding with enriched payloads
- Pedestrian kinematics: sudden acceleration / running vs steady walking
- Group incursion via pairwise Euclidean spatial clustering
- Camera tamper telemetry: occlusion, blinding, defocus
- Explainable composite 0-100 risk scoring & exact boundary tests (24, 25, 49, 50, 74, 75, 100)
- Tactical Event Correlation and Incident Relationship Model
"""

import sys
import time
from pathlib import Path

# Add project app directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import numpy as np
import cv2

from zones import Zone, ZoneManager
from tracker import Tracker
from risk_engine import RiskEngine, RiskLevel
from tamper_detector import TamperDetector
from correlator import EventCorrelator, Incident
from events import (
    event_bus,
    CameraTamperEvent,
    DwellEvent,
    SuddenMovementEvent,
    GroupIncursionEvent,
    IncidentEvent,
)


# =====================================================================
# 1. DWELL TIME / LOITERING TESTS
# =====================================================================

def test_dwell_time_warning_threshold_crossing():
    """
    Persons inside a restricted zone crossing dwell_warning_seconds (5.0s)
    must trigger an EXTENDED_DWELL event with WARNING threshold crossed
    and accurate entry_time / dwell_seconds.
    """
    restricted_zone = Zone(
        name="perimeter_restricted",
        zone_type="restricted",
        points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
        dwell_warning_seconds=5.0,
        dwell_critical_seconds=15.0,
    )
    zm = ZoneManager(zones=[restricted_zone])
    zm.begin_frame()

    t0 = 1000.0
    # Entry at t0
    events, current = zm.update("track_1", "person", (500, 500), 1000, 1000, timestamp=t0)
    assert "perimeter_restricted" in current
    enter_ev = [e for e in events if e["type"] == "ZONE_ENTER"]
    assert len(enter_ev) == 1

    # At t0 + 4.0s: below warning threshold (no EXTENDED_DWELL)
    zm.begin_frame()
    events, current = zm.update("track_1", "person", (500, 500), 1000, 1000, timestamp=t0 + 4.0)
    warn_ev = [e for e in events if e["type"] == "EXTENDED_DWELL"]
    assert len(warn_ev) == 0

    # At t0 + 5.5s: crosses warning threshold (> 5.0s)
    zm.begin_frame()
    events, current = zm.update("track_1", "person", (500, 500), 1000, 1000, timestamp=t0 + 5.5)
    warn_ev = [e for e in events if e["type"] == "EXTENDED_DWELL"]
    assert len(warn_ev) == 1
    w = warn_ev[0]
    assert w["threshold_crossed"] == "WARNING"
    assert w["dwell_seconds"] >= 5.5
    assert w["entry_time"] == t0
    assert w["threshold_seconds"] == 5.0

    # Next frame at t0 + 6.0s: debouncing ensures EXTENDED_DWELL does not fire again
    zm.begin_frame()
    events, current = zm.update("track_1", "person", (500, 500), 1000, 1000, timestamp=t0 + 6.0)
    warn_ev2 = [e for e in events if e["type"] == "EXTENDED_DWELL"]
    assert len(warn_ev2) == 0


def test_dwell_time_critical_threshold_crossing():
    """
    Crossing dwell_critical_seconds (15.0s) must trigger a LOITERING event
    with CRITICAL threshold crossed.
    """
    restricted_zone = Zone(
        name="perimeter_restricted",
        zone_type="restricted",
        points=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
        dwell_warning_seconds=5.0,
        dwell_critical_seconds=15.0,
    )
    zm = ZoneManager(zones=[restricted_zone])
    t0 = 2000.0

    # Enter
    zm.begin_frame()
    zm.update("track_2", "person", (500, 500), 1000, 1000, timestamp=t0)

    # Jump to 16.0 seconds
    zm.begin_frame()
    events, _ = zm.update("track_2", "person", (500, 500), 1000, 1000, timestamp=t0 + 16.0)
    crit_ev = [e for e in events if e["type"] == "LOITERING"]
    assert len(crit_ev) == 1
    c = crit_ev[0]
    assert c["threshold_crossed"] == "CRITICAL"
    assert c["dwell_seconds"] >= 16.0
    assert c["entry_time"] == t0
    assert c["threshold_seconds"] == 15.0


# =====================================================================
# 2. SUDDEN SPEED / RUNNING / KINEMATIC TESTS
# =====================================================================

def test_sudden_acceleration_pedestrian_kinematics():
    """
    Tracker records velocity and acceleration. When a pedestrian accelerates
    rapidly (>120 px/s^2) or jumps from walking to running, acceleration is captured.
    """
    tracker = Tracker(stale_after_seconds=5.0)

    t0 = 100.0
    # Step 1: initial detection walking slowly (dx = 5px in 0.1s -> 50 px/s)
    tracker.update([{"track_id": 10, "center": (120, 140), "box": (100, 100, 140, 180), "class_name": "person", "confidence": 0.9}], timestamp=t0)
    t1 = t0 + 0.1
    tracker.update([{"track_id": 10, "center": (125, 140), "box": (105, 100, 145, 180), "class_name": "person", "confidence": 0.9}], timestamp=t1)

    t2 = t1 + 0.1
    tracker.update([{"track_id": 10, "center": (130, 140), "box": (110, 100, 150, 180), "class_name": "person", "confidence": 0.9}], timestamp=t2)
    track = tracker.tracks[10]
    initial_speed = track.speed
    assert 30.0 <= initial_speed <= 50.0

    # Step 2: sudden sprint (dx = 40px in 0.1s -> 400 px/s)
    t3 = t2 + 0.1
    enriched = tracker.update([{"track_id": 10, "center": (170, 140), "box": (150, 100, 190, 180), "class_name": "person", "confidence": 0.9}], timestamp=t3)
    sprint_track = tracker.tracks[10]

    assert sprint_track.speed >= 120.0
    assert sprint_track.prev_speed <= 50.0
    assert sprint_track.acceleration >= 120.0

    # Verify enriched dict contains kinematic fields
    det = [d for d in enriched if d["track_id"] == 10][0]
    assert det["speed"] >= 120.0
    assert det["prev_velocity"] <= 50.0
    assert det["acceleration"] >= 120.0


def test_benign_walking_speed_not_flagged():
    """
    Steady pedestrian walking (speed ~30 px/s, near-zero acceleration)
    must NOT trigger sudden acceleration alerts.
    """
    tracker = Tracker(stale_after_seconds=5.0)
    t = 200.0
    enriched = []
    for i in range(5):
        t += 0.1
        enriched = tracker.update([{"track_id": 11, "center": (100 + i * 3, 140), "box": (100 + i * 3, 100, 140 + i * 3, 180), "class_name": "person", "confidence": 0.9}], timestamp=t)

    det = enriched[0]
    assert det["speed"] < 40.0
    assert abs(det["acceleration"]) < 50.0


# =====================================================================
# 3. GROUP INCURSION / SPATIAL CLUSTERING TESTS
# =====================================================================

def test_spatial_clustering_group_incursion():
    """
    find_spatial_clusters groups persons within cluster_radius (250px).
    3 persons within 250px form a cluster >= 3, triggering group incursion.
    """
    # 3 persons close together: (100, 100), (150, 120), (200, 100) -> distances <= 100px
    occupants = [
        (1, (100.0, 100.0)),
        (2, (150.0, 120.0)),
        (3, (200.0, 100.0)),
    ]
    clusters = RiskEngine.find_spatial_clusters(occupants, cluster_radius=250.0)
    assert len(clusters) == 1
    assert set(clusters[0]) == {1, 2, 3}


def test_dispersed_persons_not_clustered_as_group():
    """
    3 persons far apart in a large restricted zone (> 250px pairwise)
    must NOT be clustered into a group >= 3.
    """
    # Person 1 at (50, 50), Person 2 at (400, 400), Person 3 at (800, 800)
    # Distances between each pair exceed 400px > 250px
    occupants = [
        (1, (50.0, 50.0)),
        (2, (400.0, 400.0)),
        (3, (800.0, 800.0)),
    ]
    clusters = RiskEngine.find_spatial_clusters(occupants, cluster_radius=250.0)
    # Each person forms their own single-element cluster
    assert len(clusters) == 3
    for c in clusters:
        assert len(c) == 1


# =====================================================================
# 4. CAMERA TAMPER DETECTION TESTS
# =====================================================================

def test_camera_tamper_telemetry_occlusion_and_blinding():
    """
    TamperDetector detects occlusion, optical blinding, and defocus with explainable metrics.
    """
    detector = TamperDetector(history_len=5)

    # 1. Normal frame (mean ~128, high texture)
    normal_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    normal_frame[::2, ::2] = 200
    normal_frame[1::2, 1::2] = 50
    state = {}
    for _ in range(5):
        state = detector.update(normal_frame)
    assert not state["is_tampered"]
    assert state["tamper_type"] == "NONE"

    # 2. Total dark occlusion (mean = 0)
    detector.reset()
    dark_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(4):
        state = detector.update(dark_frame)
    assert state["is_tampered"]
    assert state["tamper_type"] == "OCCLUSION"
    assert state["metrics"]["mean_luminance"] < 15.0
    assert "covered" in state["reason"] or "obstructed" in state["reason"]

    # 3. Optical blinding (mean = 255)
    detector.reset()
    bright_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    for _ in range(4):
        state = detector.update(bright_frame)
    assert state["is_tampered"]
    assert state["tamper_type"] == "BLINDING"
    assert state["metrics"]["mean_luminance"] > 245.0

    # 4. Defocus / blur (flat grey, no texture, variance = 0)
    detector.reset()
    flat_frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    for _ in range(4):
        state = detector.update(flat_frame)
    assert state["is_tampered"]
    assert state["tamper_type"] == "DEFOCUS"
    assert state["metrics"]["sharpness"] < 5.0


# =====================================================================
# 5. COMPOSITE 0-100 RISK SCORING & STRICT BOUNDARY TESTS
# =====================================================================

def test_composite_risk_score_exact_boundaries():
    """
    Verify exact category boundaries:
    - LOW: 0 - 24 (tested at 0 and 24)
    - MEDIUM: 25 - 49 (tested at 25 and 49)
    - HIGH: 50 - 74 (tested at 50 and 74)
    - CRITICAL: 75 - 100 (tested at 75 and 100)
    """
    assert RiskEngine.severity_for_score(0) == RiskLevel.LOW
    assert RiskEngine.severity_for_score(24) == RiskLevel.LOW

    assert RiskEngine.severity_for_score(25) == RiskLevel.MEDIUM
    assert RiskEngine.severity_for_score(49) == RiskLevel.MEDIUM

    assert RiskEngine.severity_for_score(50) == RiskLevel.HIGH
    assert RiskEngine.severity_for_score(74) == RiskLevel.HIGH

    assert RiskEngine.severity_for_score(75) == RiskLevel.CRITICAL
    assert RiskEngine.severity_for_score(100) == RiskLevel.CRITICAL


def test_composite_risk_multi_factor_calculation():
    """
    calculate_composite_risk combines rule-based behavioral factors with transparent breakdown.
    """
    # Factor 1: Restricted zone entry alone (+30) -> Score 30 (MEDIUM)
    res_entry = RiskEngine.calculate_composite_risk([{"factor": "ZONE_ENTRY_RESTRICTED"}])
    assert res_entry["total_score"] == 30
    assert res_entry["severity"] == RiskLevel.MEDIUM
    assert len(res_entry["factor_breakdown"]) == 1

    # Factor 2: Restricted entry (+30) + Loitering Warning (+20) -> Score 50 (HIGH)
    res_warn = RiskEngine.calculate_composite_risk([
        {"factor": "ZONE_ENTRY_RESTRICTED"},
        {"factor": "LOITERING_WARNING"},
    ])
    assert res_warn["total_score"] == 50
    assert res_warn["severity"] == RiskLevel.HIGH

    # Factor 3: Restricted (+30) + Loitering Critical (+35) + Group Incursion (+25) -> Score 90 (CRITICAL)
    res_crit = RiskEngine.calculate_composite_risk([
        {"factor": "ZONE_ENTRY_RESTRICTED"},
        {"factor": "LOITERING_CRITICAL"},
        {"factor": "GROUP_INCURSION"},
    ])
    assert res_crit["total_score"] == 90
    assert res_crit["severity"] == RiskLevel.CRITICAL

    # Clamping test: extreme factors cannot exceed 100
    res_clamp = RiskEngine.calculate_composite_risk([
        {"factor": "ZONE_ENTRY_RESTRICTED"},
        {"factor": "LOITERING_CRITICAL"},
        {"factor": "GROUP_INCURSION"},
        {"factor": "SUDDEN_ACCELERATION"},
        {"factor": "CAMERA_TAMPER"},
    ])
    assert res_clamp["total_score"] == 100
    assert res_clamp["severity"] == RiskLevel.CRITICAL


# =====================================================================
# 6. TACTICAL EVENT CORRELATION & INCIDENT RELATIONSHIP TESTS
# =====================================================================

def test_event_correlator_multi_event_same_track_merges():
    """
    Multiple events occurring for the same track within 30s window
    must correlate into a single Incident, escalating severity and merging metadata.
    """
    correlator = EventCorrelator(correlation_window_seconds=30.0)

    t0 = 1000.0
    # Event 1: Zone Entry (LOW severity, score 20)
    inc1, is_new1 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:42",
        object_type="person",
        event_type="ZONE_ENTER",
        severity="LOW",
        description="Person entered monitored zone",
        zone="monitored_gate",
        risk_score=20,
        timestamp=t0,
    )
    assert is_new1 is True
    assert inc1["severity"] == "LOW"
    assert inc1["risk_score"] == 20
    assert inc1["status"] == "ACTIVE"
    inc_id = inc1["incident_id"]

    # Event 2: Line Crossing (MEDIUM severity, score 45) at t0 + 5.0s
    inc2, is_new2 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:42",
        object_type="person",
        event_type="LINE_CROSSING",
        severity="MEDIUM",
        description="Person crossed perimeter fence",
        zone="fence_line",
        risk_score=45,
        timestamp=t0 + 5.0,
    )
    assert is_new2 is False
    assert inc2["incident_id"] == inc_id
    assert inc2["severity"] == "MEDIUM"
    assert inc2["risk_score"] == 45
    assert len(inc2["events"]) == 2
    assert "monitored_gate" in inc2["zones"]
    assert "fence_line" in inc2["zones"]

    # Event 3: Facial recognition attaches identity
    inc3, is_new3 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:42",
        object_type="person",
        event_type="FACE_RECOGNITION",
        severity="HIGH",
        description="Face matched watchlist: Agent Smith",
        identity="Agent Smith",
        risk_score=70,
        timestamp=t0 + 10.0,
    )
    assert is_new3 is False
    assert inc3["incident_id"] == inc_id
    assert inc3["identity"] == "Agent Smith"
    assert inc3["severity"] == "HIGH"
    assert inc3["risk_score"] == 70
    assert inc3["duration_seconds"] == 10.0


def test_event_correlator_window_expiry_creates_new_incident():
    """
    Events separated by more than correlation_window_seconds (30s)
    must spawn a separate Incident.
    """
    correlator = EventCorrelator(correlation_window_seconds=30.0)

    t0 = 500.0
    inc1, is_new1 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:99",
        object_type="vehicle",
        event_type="ZONE_ENTER",
        severity="LOW",
        description="Vehicle entered checkpoint",
        timestamp=t0,
    )
    assert is_new1 is True

    # Next event 35 seconds later (> 30s)
    inc2, is_new2 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:99",
        object_type="vehicle",
        event_type="ZONE_ENTER",
        severity="LOW",
        description="Vehicle returned to checkpoint",
        timestamp=t0 + 35.0,
    )
    assert is_new2 is True
    assert inc2["incident_id"] != inc1["incident_id"]


def test_event_correlator_cross_camera_isolation():
    """
    Events with the same numeric track id on different cameras (cam_01 vs cam_02)
    must NOT merge into the same Incident.
    """
    correlator = EventCorrelator(correlation_window_seconds=30.0)
    t = 100.0

    inc_cam1, _ = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:5",
        object_type="person",
        event_type="ZONE_ENTER",
        severity="LOW",
        description="Person at cam 1",
        timestamp=t,
    )

    inc_cam2, _ = correlator.correlate(
        camera_id="cam_02",
        track_id="cam_02:5",
        object_type="person",
        event_type="ZONE_ENTER",
        severity="LOW",
        description="Person at cam 2",
        timestamp=t + 1.0,
    )

    assert inc_cam1["incident_id"] != inc_cam2["incident_id"]
    assert inc_cam1["camera_id"] == "cam_01"
    assert inc_cam2["camera_id"] == "cam_02"


def test_incident_lifecycle_acknowledge_and_resolve():
    """
    Correlator allows operators to acknowledge and resolve incidents.
    """
    correlator = EventCorrelator()
    inc, _ = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:7",
        object_type="car",
        event_type="ANPR_PLATE_READ",
        severity="MEDIUM",
        description="ANPR read DL01AB1234",
        plate_text="DL01AB1234",
    )
    inc_id = inc["incident_id"]
    assert inc["status"] == "ACTIVE"

    # Acknowledge
    ack = correlator.acknowledge(inc_id)
    assert ack["status"] == "ACKNOWLEDGED"
    assert ack["acknowledged_at"] is not None

    # Resolve
    res = correlator.resolve(inc_id)
    assert res["status"] == "RESOLVED"
    assert res["resolved_at"] is not None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed} Phase 4 Behavioral Risk & Suspicious Activity tests passed.")
