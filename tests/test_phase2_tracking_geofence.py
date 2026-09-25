"""
test_phase2_tracking_geofence.py

Comprehensive Phase 2 automated test suite covering:
1. Detection output structure & classes (person, car, truck, bus, motorcycle)
2. Track ID persistence across consecutive frames
3. Velocity and cardinal direction calculation (including stationary state)
4. Polygonal geofence inside/outside ray-casting logic
5. Virtual tripwire crossing detection
6. Directional crossing: OUTSIDE -> INSIDE
7. Directional crossing: INSIDE -> OUTSIDE
8. Duplicate crossing suppression & boundary debouncing
9. Multi-camera zone & tripwire isolation (BOP-01, BOP-02, BOP-03)
10. Class-in-zone rule enforcement & intrusion event schema
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

import cv2
import config
from detector import Detector
from tracker import Tracker, TrackInfo
from zones import Zone, ZoneManager, point_in_polygon, load_zones_for_camera
from lines import Line, LineCrossingDetector, load_lines_for_camera
from risk_engine import RiskEngine, RiskLevel
from events import (
    create_intrusion_event, IntrusionEvent, ZoneEvent, LineCrossingEvent
)


# ----------------------------------------------------------------------
# 1. Detection Output Structure & Target Classes
# ----------------------------------------------------------------------

def test_detection_output_structure():
    """Verify detector handles person, car, truck, bus, motorcycle and returns valid schema."""
    target_classes = {"person", "car", "truck", "bus", "motorcycle"}
    detector = Detector(
        config.MODEL_PATH,
        confidence=0.35,
        target_classes=target_classes,
    )
    assert detector.model is not None
    # Verify all 5 classes are recognized by the model's vocabulary
    model_classes = set(detector.class_names.values())
    for cls_name in target_classes:
        assert cls_name in model_classes, f"Class {cls_name} not in model vocab"

    # Test on a real video frame from BOP-01
    video_path = BASE_DIR / "videos" / "bop1_visdrone_aerial.mp4"
    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), "Could not open BOP-01 video"
    ok, frame = cap.read()
    cap.release()
    assert ok, "Could not read frame 1 from BOP-01"

    detections, result = detector.infer(frame)
    assert isinstance(detections, list)
    assert len(detections) > 0, "Expected detections in BOP-01 frame"

    for det in detections:
        assert "track_id" in det and isinstance(det["track_id"], int)
        assert "class_id" in det and isinstance(det["class_id"], int)
        assert "class_name" in det and det["class_name"] in target_classes
        assert "confidence" in det and 0.0 <= det["confidence"] <= 1.0
        assert "box" in det and len(det["box"]) == 4
        assert "center" in det and len(det["center"]) == 2
        x1, y1, x2, y2 = det["box"]
        cx, cy = det["center"]
        assert x1 <= x2 and y1 <= y2
        assert abs(cx - int((x1 + x2) / 2)) <= 1
        assert abs(cy - int((y1 + y2) / 2)) <= 1


# ----------------------------------------------------------------------
# 2. Track ID Persistence Across Frames
# ----------------------------------------------------------------------

def test_track_id_persistence():
    """Verify ByteTrack / Tracker maintains track ID identity across consecutive frames."""
    tracker = Tracker()

    def make_det(tid, x, y):
        return {
            "track_id": tid, "class_id": 0, "class_name": "person",
            "confidence": 0.92, "box": (x - 10, y - 20, x + 10, y + 20),
            "center": (x, y),
        }

    # Frame 1: New sighting
    t0 = 1000.0
    f1 = tracker.update([make_det(42, 100, 100)], timestamp=t0)
    assert f1[0]["track_id"] == 42
    assert f1[0]["is_new"] is True
    assert f1[0]["frames_seen"] == 1
    assert f1[0]["age_seconds"] == 0.0

    # Frames 2, 3, 4: Consecutive updates for track 42
    f2 = tracker.update([make_det(42, 105, 100)], timestamp=t0 + 0.1)
    assert f2[0]["track_id"] == 42
    assert f2[0]["is_new"] is False
    assert f2[0]["frames_seen"] == 2

    f3 = tracker.update([make_det(42, 110, 100)], timestamp=t0 + 0.2)
    assert f3[0]["track_id"] == 42
    assert f3[0]["frames_seen"] == 3

    f4 = tracker.update([make_det(42, 115, 100)], timestamp=t0 + 0.3)
    assert f4[0]["track_id"] == 42
    assert f4[0]["frames_seen"] == 4
    assert round(f4[0]["age_seconds"], 2) == 0.3
    assert len(f4[0]["history"]) == 4


# ----------------------------------------------------------------------
# 3. Direction & Pixel Velocity Calculation
# ----------------------------------------------------------------------

def test_direction_and_velocity_calculation():
    """Verify cardinal direction computation, pixel velocity smoothing, and stationary threshold."""
    def det(tid, x, y):
        return {
            "track_id": tid, "class_id": 0, "class_name": "person",
            "confidence": 0.90, "box": (x - 5, y - 5, x + 5, y + 5),
            "center": (x, y),
        }

    # Case A: Moving East (+x)
    tr_east = Tracker()
    tr_east.update([det(1, 100, 200)], timestamp=0.0)
    res_east = tr_east.update([det(1, 150, 200)], timestamp=1.0)
    assert res_east[0]["movement_state"] == "MOVING"
    assert res_east[0]["direction"] == "EAST"
    assert res_east[0]["speed"] > 0
    assert res_east[0]["velocity_vector"][0] > 0

    # Case B: Moving West (-x)
    tr_west = Tracker()
    tr_west.update([det(2, 200, 200)], timestamp=0.0)
    res_west = tr_west.update([det(2, 150, 200)], timestamp=1.0)
    assert res_west[0]["movement_state"] == "MOVING"
    assert res_west[0]["direction"] == "WEST"

    # Case C: Moving South (+y, downward in frame coordinates)
    tr_south = Tracker()
    tr_south.update([det(3, 200, 100)], timestamp=0.0)
    res_south = tr_south.update([det(3, 200, 160)], timestamp=1.0)
    assert res_south[0]["movement_state"] == "MOVING"
    assert res_south[0]["direction"] == "SOUTH"

    # Case D: Moving North (-y, upward in frame coordinates)
    tr_north = Tracker()
    tr_north.update([det(4, 200, 200)], timestamp=0.0)
    res_north = tr_north.update([det(4, 200, 140)], timestamp=1.0)
    assert res_north[0]["movement_state"] == "MOVING"
    assert res_north[0]["direction"] == "NORTH"

    # Case E: Stationary object
    tr_stat = Tracker()
    tr_stat.update([det(5, 300, 300)], timestamp=0.0)
    res_stat = tr_stat.update([det(5, 300, 300)], timestamp=1.0)
    assert res_stat[0]["movement_state"] == "STATIONARY"
    assert res_stat[0]["direction"] == "STATIONARY"
    assert res_stat[0]["speed"] == 0.0

    # Case F: Sub-threshold bounding box jitter (< 8 px/s)
    tr_jitter = Tracker()
    tr_jitter.update([det(6, 400, 400)], timestamp=0.0)
    # Move 3 px in 1 second = 3 px/s < 8 px/s threshold
    res_jitter = tr_jitter.update([det(6, 402, 401)], timestamp=1.0)
    assert res_jitter[0]["movement_state"] == "STATIONARY"
    assert res_jitter[0]["direction"] == "STATIONARY"


# ----------------------------------------------------------------------
# 4. Polygonal Geofences: Inside/Outside Logic
# ----------------------------------------------------------------------

def test_polygon_inside_outside_logic():
    """Verify point-in-polygon ray-casting and normalized coordinate scaling."""
    # L-shaped polygon:
    # (0,0) ---- (10,0)
    #   |          |
    #   |     (5,5)-(10,5)
    #   |       |
    # (0,10)--(5,10)
    l_shape = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    assert point_in_polygon(2, 2, l_shape) is True
    assert point_in_polygon(2, 8, l_shape) is True
    # The cutout region (8, 8) must be outside
    assert point_in_polygon(8, 8, l_shape) is False
    assert point_in_polygon(12, 5, l_shape) is False
    assert point_in_polygon(-1, -1, l_shape) is False

    # Scaled normalized zone
    zone = Zone("bop1_zero_line", "restricted", [[0.60, 0.0], [1.0, 0.0], [1.0, 1.0], [0.60, 1.0]])
    frame_w, frame_h = 1920, 1080
    # Inside point at x=1500 (1500/1920 = 0.78 > 0.60)
    assert zone.contains(1500, 540, frame_w, frame_h) is True
    # Outside point at x=800 (800/1920 = 0.416 < 0.60)
    assert zone.contains(800, 540, frame_w, frame_h) is False


# ----------------------------------------------------------------------
# 5. Virtual Tripwire Crossing Detection
# ----------------------------------------------------------------------

def test_tripwire_crossing_detection():
    """Verify tripwire crossing triggers when track side flips."""
    # Vertical tripwire at x=0.60
    line = Line("test_tripwire", [0.60, 0.0], [0.60, 1.0], "OUTSIDE", "INSIDE")
    detector = LineCrossingDetector([line], debounce_seconds=2.0)

    # Frame 1: Position on left (OUTSIDE, side +1)
    evs1 = detector.update("t1", (500, 500), 1000, 1000, timestamp=1.0)
    assert evs1 == []

    # Frame 2: Same side (still OUTSIDE)
    evs2 = detector.update("t1", (550, 500), 1000, 1000, timestamp=1.1)
    assert evs2 == []

    # Frame 3: Crosses to right (INSIDE, side -1)
    evs3 = detector.update("t1", (650, 500), 1000, 1000, timestamp=1.2)
    assert len(evs3) == 1
    assert evs3[0]["type"] == "LINE_CROSSING"
    assert evs3[0]["line"] == "test_tripwire"


# ----------------------------------------------------------------------
# 6. Directional Crossing: OUTSIDE -> INSIDE
# ----------------------------------------------------------------------

def test_directional_crossing_outside_to_inside():
    """Verify OUTSIDE -> INSIDE crossing vector is explicitly distinguished."""
    line = Line("perimeter", [0.60, 0.0], [0.60, 1.0], "OUTSIDE", "INSIDE")
    detector = LineCrossingDetector([line], debounce_seconds=2.0)

    # First sighting at x=500 (OUTSIDE)
    detector.update("t1", (500, 500), 1000, 1000, timestamp=10.0)

    # Crossing into x=700 (INSIDE)
    evs = detector.update("t1", (700, 500), 1000, 1000, timestamp=10.5)
    assert len(evs) == 1
    ev = evs[0]
    assert ev["from_label"] == "OUTSIDE"
    assert ev["to_label"] == "INSIDE"
    assert ev["crossing_direction"] == "OUTSIDE -> INSIDE"
    assert ev["direction"] == "A_TO_B"


# ----------------------------------------------------------------------
# 7. Directional Crossing: INSIDE -> OUTSIDE
# ----------------------------------------------------------------------

def test_directional_crossing_inside_to_outside():
    """Verify INSIDE -> OUTSIDE exit crossing vector is explicitly distinguished."""
    line = Line("perimeter", [0.60, 0.0], [0.60, 1.0], "OUTSIDE", "INSIDE")
    detector = LineCrossingDetector([line], debounce_seconds=2.0)

    # First sighting at x=750 (INSIDE)
    detector.update("t2", (750, 500), 1000, 1000, timestamp=20.0)

    # Crossing back out to x=450 (OUTSIDE)
    evs = detector.update("t2", (450, 500), 1000, 1000, timestamp=20.5)
    assert len(evs) == 1
    ev = evs[0]
    assert ev["from_label"] == "INSIDE"
    assert ev["to_label"] == "OUTSIDE"
    assert ev["crossing_direction"] == "INSIDE -> OUTSIDE"
    assert ev["direction"] == "B_TO_A"


# ----------------------------------------------------------------------
# 8. Duplicate Event Suppression & Debouncing
# ----------------------------------------------------------------------

def test_duplicate_event_suppression_and_debouncing():
    """Verify duplicate crossing events are suppressed within debounce cooldown period."""
    line = Line("fence", [0.50, 0.0], [0.50, 1.0], "OUTSIDE", "INSIDE")
    detector = LineCrossingDetector([line], debounce_seconds=2.0)

    # Start at x=400 (OUTSIDE) at t=100.0
    detector.update("t3", (400, 500), 1000, 1000, timestamp=100.0)

    # Cross to x=600 (INSIDE) at t=100.2 -> First crossing accepted
    evs1 = detector.update("t3", (600, 500), 1000, 1000, timestamp=100.2)
    assert len(evs1) == 1

    # Rapid oscillation: flips back to x=400 at t=100.5 (within 2.0s cooldown) -> suppressed
    evs_flap1 = detector.update("t3", (400, 500), 1000, 1000, timestamp=100.5)
    assert len(evs_flap1) == 0, "Flapping back within debounce window must be suppressed"

    # Flips again to x=600 at t=101.0 (still within 2.0s cooldown) -> suppressed
    evs_flap2 = detector.update("t3", (600, 500), 1000, 1000, timestamp=101.0)
    assert len(evs_flap2) == 0, "Flapping forward within debounce window must be suppressed"

    # After cooldown expires (t=102.5 > 100.2 + 2.0s) -> crossing permitted
    evs_cooldown = detector.update("t3", (400, 500), 1000, 1000, timestamp=102.5)
    assert len(evs_cooldown) == 1, "Crossing after debounce cooldown must generate event"
    assert evs_cooldown[0]["crossing_direction"] == "INSIDE -> OUTSIDE"


# ----------------------------------------------------------------------
# 9. Multi-Camera Zone & Line Isolation
# ----------------------------------------------------------------------

def test_multi_camera_zone_and_line_isolation():
    """Verify BOP-01, BOP-02, and BOP-03 load separate camera configurations and do not crosstalk."""
    zone_cfg = config.load_zone_config()
    line_cfg = config.load_line_config()

    # BOP-01: Zero-line restricted + Buffer patrol corridor
    z_bop1 = load_zones_for_camera(zone_cfg, "BOP-01")
    z1_names = {z.name for z in z_bop1}
    assert "zero_line_restricted" in z1_names
    assert "buffer_patrol_corridor" in z1_names

    l_bop1 = load_lines_for_camera(line_cfg, "BOP-01")
    l1_names = {l.name for l in l_bop1}
    assert "zero_line_tripwire" in l1_names

    # BOP-02: Dark-sector boundary + Thermal patrol buffer
    z_bop2 = load_zones_for_camera(zone_cfg, "BOP-02")
    z2_names = {z.name for z in z_bop2}
    assert "dark_sector_boundary" in z2_names
    assert "thermal_patrol_buffer" in z2_names

    l_bop2 = load_lines_for_camera(line_cfg, "BOP-02")
    l2_names = {l.name for l in l_bop2}
    assert "dark_sector_fence_tripwire" in l2_names

    # BOP-03: Restricted checkpoint/barrier zone + Vehicle queue lane
    z_bop3 = load_zones_for_camera(zone_cfg, "BOP-03")
    z3_names = {z.name for z in z_bop3}
    assert "restricted_barrier_zone" in z3_names
    assert "vehicle_queue_lane" in z3_names

    l_bop3 = load_lines_for_camera(line_cfg, "BOP-03")
    l3_names = {l.name for l in l_bop3}
    assert "checkpoint_barrier_tripwire" in l3_names

    # Verify no state leakage between separate ZoneManagers
    zm1 = ZoneManager(z_bop1)
    zm2 = ZoneManager(z_bop2)

    # Track 1 inside BOP-01 zero-line (x=0.80, y=0.50)
    ev1, c1 = zm1.update(1, "person", (800, 500), 1000, 1000)
    assert "zero_line_restricted" in c1

    # Same track ID 1 on BOP-02 does not have any residual state from BOP-01
    assert 1 not in zm2._track_zone_entry
    ev2, c2 = zm2.update(1, "person", (200, 500), 1000, 1000)
    assert "dark_sector_boundary" not in c2
    assert "thermal_patrol_buffer" in c2


# ----------------------------------------------------------------------
# 10. Class-In-Zone Rules & Intrusion Event Schema
# ----------------------------------------------------------------------

def test_class_in_zone_rules_and_event_schema():
    """Verify class-in-zone rule violations (pedestrian in vehicle zone) and complete event schema."""
    # BOP-03 restricted_barrier_zone allows only vehicles: ["car", "truck", "bus", "motorcycle"]
    barrier_zone = Zone(
        name="restricted_barrier_zone",
        zone_type="restricted",
        polygon_norm=[[0.0, 0.0], [1.0, 0.0], [1.0, 0.50], [0.0, 0.50]],
        allowed_classes=["car", "truck", "bus", "motorcycle"],
    )
    zm = ZoneManager([barrier_zone])
    re = RiskEngine()

    # Case A: Authorized vehicle enters barrier zone -> Allowed (not a class violation)
    car_events, _ = zm.update(10, "car", (500, 250), 1000, 1000)
    assert len(car_events) == 1
    assert car_events[0]["violation"] is False

    # Case B: Unauthorized pedestrian enters vehicle barrier zone -> VIOLATION
    person_events, _ = zm.update(11, "person", (500, 250), 1000, 1000)
    assert len(person_events) == 1
    assert person_events[0]["violation"] is True

    # RiskEngine evaluation: Violation in restricted zone escalates to CRITICAL
    level, reasons = re.evaluate_track(11, "person", person_events)
    assert level == RiskLevel.CRITICAL
    assert any("not permitted" in r for r in reasons)

    # Case C: IntrusionEvent schema completeness
    evt = create_intrusion_event(
        camera_id="BOP-03",
        track_id="BOP-03:11",
        object_class="person",
        zone_or_line="restricted_barrier_zone",
        direction="OUTSIDE -> INSIDE",
        bounding_box=(480, 200, 520, 300),
        event_type="UNAUTHORIZED_ZONE_ENTRY",
        severity="CRITICAL",
        reasons=reasons,
    )
    assert isinstance(evt, IntrusionEvent)
    assert evt.event_id.startswith("EVT-BOP-03-")
    assert evt.camera_id == "BOP-03"
    assert evt.track_id == "BOP-03:11"
    assert evt.object_class == "person"
    assert evt.zone_or_line == "restricted_barrier_zone"
    assert evt.direction == "OUTSIDE -> INSIDE"
    assert evt.bounding_box == (480, 200, 520, 300)
    assert evt.event_type == "UNAUTHORIZED_ZONE_ENTRY"
    assert evt.severity == "CRITICAL"
    assert "reasons" in evt.metadata


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n============================================================")
    print(f"ALL {len(tests)} PHASE 2 TESTS PASSED SUCCESSFULLY!")
    print(f"============================================================")
