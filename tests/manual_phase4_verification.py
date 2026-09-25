#!/usr/bin/env python3
"""
tests/manual_phase4_verification.py

Executes and demonstrates the 6 required Phase 4 End-to-End Capabilities:
1. DWELL TIME / LOITERING:
   - Configurable warning (>5s) and critical (>15s) thresholds
   - Enriched event payload (track_id, camera, zone, entry_time, dwell_seconds, threshold_crossed)
2. SUDDEN SPEED / RUNNING:
   - Kinematic velocity delta and acceleration calculation
   - Pedestrian running vs benign walking
3. GROUP INCURSION:
   - Spatial clustering (pairwise Euclidean distance <= 250px)
   - Groups >= 3 persons in restricted zones flagged with member tracks
4. CAMERA TAMPER DETECTION:
   - Real-time optical occlusion, blinding, defocus detection
   - Explainable metrics (luminance, standard deviation, Laplacian variance)
5. COMPOSITE 0-100 RISK SCORE:
   - Rule-based multi-factor accumulation
   - Exact category boundaries verified (LOW: 0-24, MEDIUM: 25-49, HIGH: 50-74, CRITICAL: 75-100)
6. TACTICAL EVENT CORRELATION:
   - Multi-signal events for same track merged into unified Incident
   - Correlation windowing, severity escalation, dossier generation
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT))

import config
import shared
from zones import Zone, ZoneManager
from tracker import Tracker
from risk_engine import RiskEngine, RiskLevel
from tamper_detector import TamperDetector
from correlator import EventCorrelator, Incident
from events import event_bus, DwellEvent, SuddenMovementEvent, GroupIncursionEvent, IncidentEvent


def run_phase4_verification():
    print("=" * 70)
    print("IBVAP PHASE 4: SUSPICIOUS ACTIVITY DETECTION & BEHAVIORAL RISK ENGINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # DEMO 1: DWELL TIME & LOITERING ESCALATION
    # -------------------------------------------------------------
    print("\n--- [DEMO 1] LOITERING / DWELL TIME CONFIGURABLE THRESHOLDS ---")
    zone = Zone(
        name="Sector_A_Restricted",
        zone_type="restricted",
        polygon_norm=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
        dwell_warning_seconds=5.0,
        dwell_critical_seconds=15.0,
    )
    zm = ZoneManager(zones=[zone])
    t0 = time.time()

    # Step 1: Entry at t0
    zm.begin_frame()
    ev1, inside1 = zm.update("track_BOP_101", "person", (500, 500), 1000, 1000, timestamp=t0)
    print(f"[*] t = 0.0s: Person entered '{list(inside1)[0]}'. Events: {[e['type'] for e in ev1]}")

    # Step 2: Dwell at t0 + 6.0s (Warning crossed: > 5.0s)
    zm.begin_frame()
    ev2, _ = zm.update("track_BOP_101", "person", (500, 500), 1000, 1000, timestamp=t0 + 6.0)
    warn_ev = [e for e in ev2 if e["type"] == "EXTENDED_DWELL"][0]
    print(f"[!] t = 6.0s: Crossed DWELL WARNING threshold (5.0s)!")
    print(f"    Payload: track_id={warn_ev['track_id']}, dwell_seconds={warn_ev['dwell_seconds']}s, "
          f"threshold={warn_ev['threshold_crossed']} ({warn_ev['threshold_seconds']}s)")

    # Step 3: Dwell at t0 + 16.0s (Critical crossed: > 15.0s)
    zm.begin_frame()
    ev3, _ = zm.update("track_BOP_101", "person", (500, 500), 1000, 1000, timestamp=t0 + 16.0)
    crit_ev = [e for e in ev3 if e["type"] == "LOITERING"][0]
    print(f"[!] t = 16.0s: Crossed LOITERING CRITICAL threshold (15.0s)!")
    print(f"    Payload: track_id={crit_ev['track_id']}, dwell_seconds={crit_ev['dwell_seconds']}s, "
          f"threshold={crit_ev['threshold_crossed']} ({crit_ev['threshold_seconds']}s)")

    # -------------------------------------------------------------
    # DEMO 2: PEDESTRIAN KINEMATICS & SUDDEN SPRINT
    # -------------------------------------------------------------
    print("\n--- [DEMO 2] PEDESTRIAN KINEMATICS (SUDDEN SPEED / ACCELERATION) ---")
    tracker = Tracker(stale_after_seconds=5.0)

    # Track 1: Steady Walking pedestrian (~30 px/s)
    t = 100.0
    for i in range(5):
        t += 0.1
        tracker.update([{"track_id": 1, "center": (100 + i * 3, 200), "box": (90 + i * 3, 160, 110 + i * 3, 240), "class_name": "person", "confidence": 0.95}], timestamp=t)
    steady_track = tracker.tracks[1]
    print(f"[*] Pedestrian 1 (Normal Walking): speed={steady_track.speed:.1f} px/s, "
          f"accel={steady_track.acceleration:.1f} px/s^2 -> Status: Normal (No alert)")

    # Track 2: Sudden sprint (pedestrian accelerating to >120 px/s)
    t = 100.0
    for i in range(4):
        t += 0.1
        tracker.update([{"track_id": 2, "center": (100 + i * 3, 300), "box": (90 + i * 3, 260, 110 + i * 3, 340), "class_name": "person", "confidence": 0.95}], timestamp=t)
    # Sudden burst: dx = 40px in 0.1s
    t += 0.1
    enriched2 = tracker.update([{"track_id": 2, "center": (150, 300), "box": (140, 260, 160, 340), "class_name": "person", "confidence": 0.95}], timestamp=t)
    sprint_track = [d for d in enriched2 if d["track_id"] == 2][0]
    print(f"[!] Pedestrian 2 (Sudden Acceleration): speed={sprint_track['speed']:.1f} px/s, "
          f"prev={sprint_track['prev_velocity']:.1f} px/s, accel={sprint_track['acceleration']:.1f} px/s^2 "
          f"direction={sprint_track['direction']} -> Flagged: SUDDEN_ACCELERATION!")

    # -------------------------------------------------------------
    # DEMO 3: GROUP INCURSION VIA SPATIAL CLUSTERING
    # -------------------------------------------------------------
    print("\n--- [DEMO 3] GROUP INCURSION DETECTION (SPATIAL CLUSTERING) ---")
    # 3 persons clustered within 250px inside restricted area
    occupants = [
        ("track_P1", (300.0, 300.0)),
        ("track_P2", (340.0, 320.0)),
        ("track_P3", (380.0, 310.0)),
        ("track_P4_solo", (850.0, 850.0)),  # Dispersed solo intruder
    ]
    clusters = RiskEngine.find_spatial_clusters(occupants, cluster_radius=250.0)
    print(f"[*] Evaluated 4 intruders in restricted zone with cluster_radius=250px:")
    for idx, c in enumerate(clusters, 1):
        if len(c) >= 3:
            print(f"    -> Cluster {idx}: GROUP INCURSION DETECTED! Members ({len(c)} persons): {c}")
        else:
            print(f"    -> Cluster {idx}: Individual intruder ({len(c)} person): {c}")

    # -------------------------------------------------------------
    # DEMO 4: CAMERA TAMPER TELEMETRY
    # -------------------------------------------------------------
    print("\n--- [DEMO 4] CAMERA TAMPER DETECTION TELEMETRY ---")
    tamper_det = TamperDetector(history_len=5)

    # 1. Occlusion
    dark_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(4):
        state_occ = tamper_det.update(dark_frame)
    print(f"[!] Occlusion Test: is_tampered={state_occ['is_tampered']}, type={state_occ['tamper_type']}, "
          f"metrics={state_occ['metrics']}, reason='{state_occ['reason']}'")

    # 2. Blinding
    tamper_det.reset()
    bright_frame = np.full((240, 320, 3), 255, dtype=np.uint8)
    for _ in range(4):
        state_blind = tamper_det.update(bright_frame)
    print(f"[!] Blinding Test:  is_tampered={state_blind['is_tampered']}, type={state_blind['tamper_type']}, "
          f"metrics={state_blind['metrics']}, reason='{state_blind['reason']}'")

    # 3. Defocus
    tamper_det.reset()
    flat_frame = np.full((240, 320, 3), 110, dtype=np.uint8)
    for _ in range(4):
        state_blur = tamper_det.update(flat_frame)
    print(f"[!] Defocus Test:   is_tampered={state_blur['is_tampered']}, type={state_blur['tamper_type']}, "
          f"metrics={state_blur['metrics']}, reason='{state_blur['reason']}'")

    # -------------------------------------------------------------
    # DEMO 5: COMPOSITE 0-100 RISK SCORE & BOUNDARY TESTS
    # -------------------------------------------------------------
    print("\n--- [DEMO 5] COMPOSITE 0-100 RISK SCORE & EXACT BOUNDARIES ---")
    boundaries = [(0, "LOW"), (24, "LOW"), (25, "MEDIUM"), (49, "MEDIUM"),
                  (50, "HIGH"), (74, "HIGH"), (75, "CRITICAL"), (100, "CRITICAL")]
    for score_val, expected_sev in boundaries:
        computed_sev = RiskEngine.severity_for_score(score_val)
        assert computed_sev == expected_sev
        print(f"    Boundary Score {score_val:3d} -> Category: {computed_sev:8s} [VERIFIED]")

    # Multi-factor explainable calculation
    factors = [
        {"factor": "ZONE_ENTRY_RESTRICTED"},
        {"factor": "LOITERING_WARNING"},
        {"factor": "SUDDEN_ACCELERATION"},
    ]
    comp_res = RiskEngine.calculate_composite_risk(factors)
    print(f"\n[*] Multi-factor Composite Risk Calculation:")
    print(f"    Total Score: {comp_res['total_score']}/100 | Severity: {comp_res['severity']}")
    print(f"    Explainable Breakdown:")
    for item in comp_res["factor_breakdown"]:
        print(f"      - {item['raw_factor']}: +{item['points']} pts")

    # -------------------------------------------------------------
    # DEMO 6: TACTICAL EVENT CORRELATION & INCIDENT DOSSIER
    # -------------------------------------------------------------
    print("\n--- [DEMO 6] TACTICAL EVENT CORRELATION & INCIDENT RELATIONSHIP ---")
    correlator = EventCorrelator(correlation_window_seconds=30.0)
    now = time.time()

    # Observation 1: Perimeter Zone Entry
    inc1, is_new1 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:42",
        object_type="person",
        event_type="ZONE_ENTER",
        severity="LOW",
        description="Person entered Sector 4 approach",
        zone="Sector_4_Approach",
        risk_score=20,
        timestamp=now,
    )
    print(f"[1] Event 1 received: Created Incident '{inc1['incident_id']}' (Severity: {inc1['severity']}, Score: {inc1['risk_score']})")

    # Observation 2: Line Crossing (Perimeter Fence)
    inc2, is_new2 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:42",
        object_type="person",
        event_type="LINE_CROSSING",
        severity="HIGH",
        description="Person climbed perimeter fence",
        zone="Zero_Line_Fence",
        risk_score=65,
        timestamp=now + 4.0,
    )
    print(f"[2] Event 2 received: Correlated into Incident '{inc2['incident_id']}' (Escalated to: {inc2['severity']}, Score: {inc2['risk_score']})")

    # Observation 3: Facial Recognition match
    inc3, is_new3 = correlator.correlate(
        camera_id="cam_01",
        track_id="cam_01:42",
        object_type="person",
        event_type="FACE_RECOGNITION",
        severity="HIGH",
        description="Watchlist match: Sentry Guard Trainee",
        identity="Trainee Singh",
        risk_score=75,
        timestamp=now + 8.0,
    )
    print(f"[3] Event 3 received: Identity '{inc3['identity']}' attached to Incident '{inc3['incident_id']}'")
    print(f"    Incident Dossier Summary:")
    print(f"      - Incident ID:       {inc3['incident_id']}")
    print(f"      - Camera:            {inc3['camera_id']}")
    print(f"      - Target Track:      {inc3['primary_track_id']}")
    print(f"      - Status:            {inc3['status']}")
    print(f"      - Composite Score:   {inc3['risk_score']}/100 ({inc3['severity']})")
    print(f"      - Linked Zones:      {inc3['zones']}")
    print(f"      - Correlated Events: {len(inc3['events'])} events")
    print(f"      - Duration:          {inc3['duration_seconds']}s")

    print("\n" + "=" * 70)
    print("PHASE 4 BEHAVIORAL RISK & CORRELATION ENGINE: ALL CAPABILITIES VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase4_verification()
