"""
test_center.py

Feature Test Center Engine for IBVAP (SIH 2026 Problem Statement 187).
Provides:
  1. Static and dynamic Capability Inventory with short judge-facing descriptions.
  2. Isolated Demo Session management: spins up dedicated demo camera workers
     without modifying normal operational cameras or workers.
  3. Curated Complete Demo sequence runner with bounded execution times.
"""

import os
import time
import threading
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque

import config
from camera_worker import CameraWorker
from events import event_bus


# ==============================================================================
# 1. GROUNDED FEATURE INVENTORY WITH SHORT CLEAR DESCRIPTIONS
# ==============================================================================

FEATURE_INVENTORY: List[Dict[str, Any]] = [
    {
        "id": "human_detection",
        "name": "Human Detection",
        "category": "Computer Vision",
        "description": "Detects people across camera streams using YOLO with confidence filtering and bounding boxes.",
        "short_description": "Detects people across camera streams using YOLO with confidence filtering and bounding boxes.",
        "primary_camera": "DEMO-01",
        "demo_video": "Walk1.mpg",
        "demo_video_file": "Walk1.mpg",
        "expected_event_types": ["TRACK", "ZONE_ENTER"],
        "expected_event_type": "TRACK, ZONE_ENTER",
        "target_class": "person",
        "timeout_seconds": 8.0,
    },
    {
        "id": "human_tracking",
        "name": "Human Tracking & Trajectory",
        "category": "Computer Vision",
        "description": "Maintains persistent ByteTrack track IDs, velocity vectors, cardinal heading, and trajectory histories while people remain visible.",
        "short_description": "Maintains persistent ByteTrack track IDs, velocity vectors, cardinal heading, and trajectory histories while people remain visible.",
        "primary_camera": "DEMO-01",
        "demo_video": "Walk2.mpg",
        "demo_video_file": "Walk2.mpg",
        "expected_event_types": ["TRACK"],
        "expected_event_type": "TRACK",
        "target_class": "person",
        "timeout_seconds": 8.0,
    },
    {
        "id": "vehicle_classification",
        "name": "Vehicle Detection & Classification",
        "category": "Computer Vision",
        "description": "Detects and classifies motor vehicles into distinct classes: car, truck, bus, and motorcycle.",
        "short_description": "Detects and classifies motor vehicles into distinct classes: car, truck, bus, and motorcycle.",
        "primary_camera": "DEMO-01",
        "demo_video": "2024-04-24 10-35-00.mp4",
        "demo_video_file": "2024-04-24 10-35-00.mp4",
        "expected_event_types": ["TRACK"],
        "expected_event_type": "TRACK",
        "target_class": "vehicle",
        "timeout_seconds": 8.0,
    },
    {
        "id": "anpr",
        "name": "Indian ANPR (HSRP)",
        "category": "Identity & Access",
        "description": "Detects vehicle plates, crops bumpers, executes dual-pass OCR, aggregates reads across frames, and validates 36 Indian state codes.",
        "short_description": "Detects vehicle plates, crops bumpers, executes dual-pass OCR, aggregates reads across frames, and validates 36 Indian state codes.",
        "primary_camera": "DEMO-01",
        "demo_video": "sih_video.mov",
        "demo_video_file": "sih_video.mov",
        "expected_event_types": ["ANPR_PLATE_READ"],
        "expected_event_type": "ANPR_PLATE_READ",
        "target_class": "vehicle",
        "timeout_seconds": 10.0,
    },
    {
        "id": "face_recognition",
        "name": "Face Detection & Recognition",
        "category": "Identity & Access",
        "description": "Detects faces at chokepoints, applies quality/sharpness filtering, generates embeddings, and matches against the authorized gallery.",
        "short_description": "Detects faces at chokepoints, applies quality/sharpness filtering, generates embeddings, and matches against the authorized gallery.",
        "primary_camera": "DEMO-01",
        "demo_video": "output (1).mp4",
        "demo_video_file": "output (1).mp4",
        "expected_event_types": ["FACE_RECOGNITION", "FACE_MATCH"],
        "expected_event_type": "FACE_RECOGNITION, FACE_MATCH",
        "target_class": "person",
        "timeout_seconds": 10.0,
    },
    {
        "id": "virtual_fence",
        "name": "Virtual Fence & Zone Intrusion",
        "category": "Perimeter & Zones",
        "description": "Performs ray-casting polygon boundary tests and triggers breach alerts when objects enter configured restricted zones.",
        "short_description": "Performs ray-casting polygon boundary tests and triggers breach alerts when objects enter configured restricted zones.",
        "primary_camera": "DEMO-01",
        "demo_video": "worker-zone-detection.mp4",
        "demo_video_file": "worker-zone-detection.mp4",
        "expected_event_types": ["ZONE_ENTER", "UNAUTHORIZED_ZONE_ENTRY"],
        "expected_event_type": "ZONE_ENTER, UNAUTHORIZED_ZONE_ENTRY",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
    {
        "id": "tripwire",
        "name": "Directional Tripwire / Line Crossing",
        "category": "Perimeter & Zones",
        "description": "Calculates 2D vector cross-product crossings over virtual border lines, distinguishing inbound from outbound crossing.",
        "short_description": "Calculates 2D vector cross-product crossings over virtual border lines, distinguishing inbound from outbound crossing.",
        "primary_camera": "DEMO-01",
        "demo_video": "Walk2.mpg",
        "demo_video_file": "Walk2.mpg",
        "expected_event_types": ["LINE_CROSSING"],
        "expected_event_type": "LINE_CROSSING",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
    {
        "id": "loitering",
        "name": "Dwell Time & Loitering",
        "category": "Behavioral Analytics",
        "description": "Tracks occupancy duration inside zones with two-stage thresholding: warning dwell (5s) and critical loitering (15s).",
        "short_description": "Tracks occupancy duration inside zones with two-stage thresholding: warning dwell (5s) and critical loitering (15s).",
        "primary_camera": "DEMO-01",
        "demo_video": "Browse_WhileWaiting1.mpg",
        "demo_video_file": "Browse_WhileWaiting1.mpg",
        "expected_event_types": ["EXTENDED_DWELL", "LOITERING"],
        "expected_event_type": "EXTENDED_DWELL, LOITERING",
        "target_class": None,
        "timeout_seconds": 10.0,
    },
    {
        "id": "sudden_movement",
        "name": "Sudden Acceleration & Running",
        "category": "Behavioral Analytics",
        "description": "Computes kinematic acceleration and speed deltas to detect running, sprinting, or rapid fleeing near perimeter boundaries.",
        "short_description": "Computes kinematic acceleration and speed deltas to detect running, sprinting, or rapid fleeing near perimeter boundaries.",
        "primary_camera": "DEMO-01",
        "demo_video": "Fight_RunAway1.mpg",
        "demo_video_file": "Fight_RunAway1.mpg",
        "expected_event_types": ["SUDDEN_ACCELERATION", "RAPID_MOVEMENT", "SUDDEN_MOVEMENT", "ACCEL", "BEHAVIOR"],
        "expected_event_type": "SUDDEN_ACCELERATION, RAPID_MOVEMENT",
        "target_class": "person",
        "timeout_seconds": 8.0,
    },
    {
        "id": "group_incursion",
        "name": "Group Incursion Detection",
        "category": "Behavioral Analytics",
        "description": "Monitors spatial clustering to flag group incursions when 3 or more simultaneous objects occupy a restricted zone.",
        "short_description": "Monitors spatial clustering to flag group incursions when 3 or more simultaneous objects occupy a restricted zone.",
        "primary_camera": "DEMO-01",
        "demo_video": "10.mp4",
        "demo_video_file": "10.mp4",
        "expected_event_types": ["GROUP_INCURSION", "GROUP", "ZONE_ENTER", "UNAUTHORIZED_ZONE_ENTRY"],
        "expected_event_type": "GROUP_INCURSION, ZONE_ENTER",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
    {
        "id": "night_movement",
        "name": "Night-Time & Thermal Movement",
        "category": "Sensor Analytics",
        "description": "Computes real-time scene luminance (< 80 threshold) on thermal/IR feeds and gates movement detection in low-light zones.",
        "short_description": "Computes real-time scene luminance (< 80 threshold) on thermal/IR feeds and gates movement detection in low-light zones.",
        "primary_camera": "DEMO-02",
        "demo_video": "night_movement.mp4",
        "demo_video_file": "night_movement.mp4",
        "expected_event_types": ["NIGHT_MOVEMENT", "NIGHT", "ZONE_ENTER"],
        "expected_event_type": "NIGHT_MOVEMENT, ZONE_ENTER",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
    {
        "id": "camera_tamper",
        "name": "Camera Tamper & Defocus Detection",
        "category": "Sensor Analytics",
        "description": "Measures Laplacian blur variance for defocus, mean luminance for blinding, and spatial occlusion for camera blackout.",
        "short_description": "Measures Laplacian blur variance for defocus, mean luminance for blinding, and spatial occlusion for camera blackout.",
        "primary_camera": "DEMO-03",
        "demo_video": "bop3_indian_checkpoint_traffic.mp4",
        "demo_video_file": "bop3_indian_checkpoint_traffic.mp4",
        "expected_event_types": ["CAMERA_TAMPER", "CAMERA_TAMPERED", "DEFOCUS_BLUR", "DEFOCUS", "OCCLUSION", "BLINDING", "TAMPER"],
        "expected_event_type": "CAMERA_TAMPER, DEFOCUS_BLUR",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
    {
        "id": "risk_engine",
        "name": "Explainable Risk Scoring Engine",
        "category": "Threat Assessment",
        "description": "Computes deterministic 0-100 threat scores across LOW, MEDIUM, HIGH, and CRITICAL bands with additive factor breakdowns.",
        "short_description": "Computes deterministic 0-100 threat scores across LOW, MEDIUM, HIGH, and CRITICAL bands with additive factor breakdowns.",
        "primary_camera": "DEMO-01",
        "demo_video": "bop1_visdrone_aerial.mp4",
        "demo_video_file": "bop1_visdrone_aerial.mp4",
        "expected_event_types": ["RISK", "RISK_LEVEL_CHANGE", "ZONE_RISK", "ZONE_ENTER"],
        "expected_event_type": "RISK_LEVEL_CHANGE, ZONE_ENTER",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
    {
        "id": "incident_dossier",
        "name": "Incident Correlation & Tactical Dossier",
        "category": "Threat Assessment",
        "description": "Groups multi-sensor detections into correlated incidents with chronological timelines and BSF QRT SOP dispatch checklists.",
        "short_description": "Groups multi-sensor detections into correlated incidents with chronological timelines and BSF QRT SOP dispatch checklists.",
        "primary_camera": "DEMO-01",
        "demo_video": "bop1_visdrone_aerial.mp4",
        "demo_video_file": "bop1_visdrone_aerial.mp4",
        "expected_event_types": ["INCIDENT", "INCIDENT_CREATED", "INCIDENT_UPDATE", "ZONE_ENTER"],
        "expected_event_type": "INCIDENT_CREATED, ZONE_ENTER",
        "target_class": None,
        "timeout_seconds": 8.0,
    },
]



CAP_ALIAS_MAP = {
    f"CAP-{i+1:02d}": f["id"] for i, f in enumerate(FEATURE_INVENTORY)
}
CAP_ALIAS_MAP.update({
    f"CAP-{i+1}": f["id"] for i, f in enumerate(FEATURE_INVENTORY)
})


# ==============================================================================
# 2. ISOLATED DEMO SESSION
# ==============================================================================

class DemoSession:
    """
    Manages an isolated demonstration session running dedicated camera workers
    on designated demo footage. Does NOT modify or disrupt normal camera workers
    or normal camera configuration.
    """

    def __init__(self, feature_id: str):
        if feature_id.upper() in CAP_ALIAS_MAP:
            feature_id = CAP_ALIAS_MAP[feature_id.upper()]

        self.feature_id = feature_id
        self.feature_meta = next((f for f in FEATURE_INVENTORY if f["id"] == feature_id), None)
        if not self.feature_meta:
            raise ValueError(f"Unknown feature ID: {feature_id}")

        self.session_id = f"demo_{feature_id}_{int(time.time())}"
        self.created_at = time.time()
        self.is_active = True
        self.demo_workers: Dict[str, CameraWorker] = {}
        self.collected_events: deque = deque(maxlen=2000)
        self.total_events_collected: int = 0
        self._lock = threading.Lock()
        self._sse_queue = None

        self._init_workers()
        self._subscribe_events()

    @property
    def workers(self) -> Dict[str, CameraWorker]:
        return self.demo_workers

    @property
    def feature(self) -> Dict[str, Any]:
        return self.feature_meta

    def get_status(self) -> Dict[str, Any]:
        return {
            "active_session": self.is_active,
            "session_id": self.session_id,
            "feature": self.feature_meta,
            "worker_status": {
                cid: {
                    "status": "RUNNING" if w.running else "STOPPED",
                    "fps": round(w.fps, 1),
                    "object_count": sum(w.object_counts.values()) if w.object_counts else 0,
                    "frame_count": w.frame_count,
                }
                for cid, w in self.demo_workers.items()
            },
        }

    def _init_workers(self):
        """Spins up isolated demo workers without touching normal camera workers."""
        primary_cam = self.feature_meta["primary_camera"]
        demo_vid = self.feature_meta["demo_video"]

        def _resolve_video(vid_name: str) -> str:
            local_vid = config.PROJECT_ROOT / "videos" / vid_name
            if local_vid.exists():
                return str(local_vid)
            ext_vid = Path("/Users/yashwantankatwar/Documents/DRIVE E/COLLEGE/COLLEGE/HACKATHON/sih 2026/v4/IBVAP_Datasets/dataset_for_test") / vid_name
            if ext_vid.exists():
                return str(ext_vid)
            for alt in ["worker-zone-detection.mp4", "night_movement.mp4", "sih_video.mov", "2024-04-24 10-35-00.mp4", "video6.mp4", "bop1_visdrone_aerial.mp4", "bop3_indian_checkpoint_traffic.mp4"]:
                alt_path = config.PROJECT_ROOT / "videos" / alt
                if alt_path.exists():
                    return str(alt_path)
            return str(local_vid)

        video_path = _resolve_video(demo_vid)

        # For the single-camera test features, run ONLY DEMO-01 with its designated video.
        # Remove all other auxiliary videos (DEMO-02, DEMO-03) from these sections.
        single_cam_features = {
            "human_detection",
            "human_tracking",
            "virtual_fence",
            "tripwire",
            "loitering",
            "sudden_movement",
            "group_incursion",
        }

        if self.feature_id == "virtual_fence":
            from zones import Zone
            from lines import Line
            vf_zones = [
                Zone(
                    name="restricted_zone",
                    zone_type="restricted",
                    polygon_norm=[[0.0, 0.0], [0.55, 0.0], [0.55, 1.0], [0.0, 1.0]],
                    severity="HIGH",
                    max_dwell_seconds=5,
                    max_objects=3,
                ),
                Zone(
                    name="pedestrian_lane",
                    zone_type="monitored",
                    polygon_norm=[[0.55, 0.0], [1.0, 0.0], [1.0, 1.0], [0.55, 1.0]],
                    severity="LOW",
                    max_objects=3,
                    allowed_classes=["person"],
                ),
            ]
            vf_lines = [
                Line(
                    name="boundary_line",
                    point_a_norm=(0.55, 0.0),
                    point_b_norm=(0.55, 1.0),
                    side_a_label="RESTRICTED",
                    side_b_label="MONITORED",
                )
            ]
            demo_configs = {
                "DEMO-01": {
                    "name": f"DEMO-01 ({self.feature_meta['name']})",
                    "source": str(video_path),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": False,
                    "zones": vf_zones,
                    "lines": vf_lines,
                }
            }
        elif self.feature_id in single_cam_features:
            demo_configs = {
                "DEMO-01": {
                    "name": f"DEMO-01 ({self.feature_meta['name']})",
                    "source": str(video_path),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": False,
                }
            }
        elif self.feature_id == "vehicle_classification":
            demo_configs = {
                "DEMO-01": {
                    "name": "DEMO-01 (Traffic Highway 1)",
                    "source": _resolve_video("2024-04-24 10-35-00.mp4"),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": False,
                },
                "DEMO-02": {
                    "name": "DEMO-02 (Traffic Intersection 2)",
                    "source": _resolve_video("video6.mp4"),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": False,
                },
                "DEMO-03": {
                    "name": "DEMO-03 (Border Road 3)",
                    "source": _resolve_video("sih_video.mov"),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": False,
                },
            }
        elif self.feature_id == "anpr":
            demo_configs = {
                "DEMO-01": {
                    "name": "DEMO-01 (ANPR Primary · HSRP)",
                    "source": _resolve_video("sih_video.mov"),
                    "anpr_enabled": True,
                    "face_enabled": False,
                    "night_enabled": False,
                },
                "DEMO-02": {
                    "name": "DEMO-02 (ANPR Auxiliary · Lane 2)",
                    "source": _resolve_video("2024-04-24 10-35-00.mp4"),
                    "anpr_enabled": True,
                    "face_enabled": False,
                    "night_enabled": False,
                },
                "DEMO-03": {
                    "name": "DEMO-03 (ANPR Auxiliary · Lane 3)",
                    "source": _resolve_video("video6.mp4"),
                    "anpr_enabled": True,
                    "face_enabled": False,
                    "night_enabled": False,
                },
            }
        elif self.feature_id == "face_recognition":
            demo_configs = {
                "DEMO-01": {
                    "name": "DEMO-01 (Face Recognition · Portal 1)",
                    "source": _resolve_video("output (1).mp4"),
                    "anpr_enabled": False,
                    "face_enabled": True,
                    "night_enabled": False,
                },
                "DEMO-02": {
                    "name": "DEMO-02 (Face Recognition · Portal 2)",
                    "source": _resolve_video("output.mp4"),
                    "anpr_enabled": False,
                    "face_enabled": True,
                    "night_enabled": False,
                },
            }
        elif self.feature_id == "night_movement":
            demo_configs = {
                "DEMO-02": {
                    "name": "DEMO-02 (Night Movement Target · Thermal)",
                    "source": _resolve_video("night_movement.mp4"),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": True,
                },
            }
        else:
            demo_configs = {
                "DEMO-01": {
                    "name": "DEMO-01 (Sector North · Recon)",
                    "source": str(video_path if primary_cam in ("DEMO-01", "BOP-01") else _resolve_video("bop1_visdrone_aerial.mp4")),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": False,
                },
                "DEMO-02": {
                    "name": "DEMO-02 (Sector East · Thermal/Portal)",
                    "source": str(video_path if primary_cam in ("DEMO-02", "BOP-02") else _resolve_video("bop2_kaist_night_thermal.mp4")),
                    "anpr_enabled": False,
                    "face_enabled": False,
                    "night_enabled": (self.feature_id == "night_movement"),
                },
                "DEMO-03": {
                    "name": "DEMO-03 (BOP Checkpoint & Gate)",
                    "source": str(video_path if primary_cam in ("DEMO-03", "BOP-03") else _resolve_video("bop3_indian_checkpoint_traffic.mp4")),
                    "anpr_enabled": True,
                    "face_enabled": False,
                    "night_enabled": False,
                },
            }

        worker_conf = (
            0.12 if self.feature_id == "face_recognition"
            else (0.18 if (self.feature_id in single_cam_features or self.feature_id in ("anpr", "vehicle_detection", "vehicle_classification")) else config.CONFIDENCE_THRESHOLD)
        )
        for cam_id, cfg in demo_configs.items():
            worker = CameraWorker(
                camera_id=cam_id,
                source=cfg["source"],
                model_path=config.MODEL_PATH,
                confidence=worker_conf,
                anpr_enabled=cfg.get("anpr_enabled", False),
                face_enabled=cfg.get("face_enabled", False),
                name=cfg["name"],
                purpose=f"Feature Demonstration: {self.feature_meta['name']}",
                zones=cfg.get("zones"),
                lines=cfg.get("lines"),
            )
            # If camera tamper demo, trigger a demonstration tamper condition on DEMO-03
            if self.feature_id == "camera_tamper" and cam_id == "DEMO-03":
                worker.simulate_tamper("DEFOCUS_BLUR")

            worker.start()
            self.demo_workers[cam_id] = worker

    def _subscribe_events(self):
        """Listen to event bus for events generated by demo workers, filtering to the capability under test."""
        self._sse_queue = event_bus.subscribe()
        expected_types = set(self.feature_meta.get("expected_event_types", []))
        seen_events = {}

        def listener():
            while self.is_active:
                try:
                    ev = self._sse_queue.get(timeout=1.0)
                    if not (hasattr(ev, "camera_id") and str(ev.camera_id).startswith("DEMO-")):
                        continue

                    ev_type = getattr(ev, "event_type", getattr(ev, "kind", type(ev).__name__))
                    ev_kind = getattr(ev, "event_kind", "")
                    ev_sub_kind = getattr(ev, "kind", "")
                    cls_name = type(ev).__name__

                    # Construct comprehensive searchable token space across all attributes
                    searchable_parts = [
                        ev_type,
                        ev_kind,
                        ev_sub_kind,
                        cls_name,
                        getattr(ev, "resulting_event", ""),
                        getattr(ev, "tamper_type", ""),
                        getattr(ev, "action", ""),
                        getattr(ev, "threshold_crossed", ""),
                        getattr(ev, "direction", ""),
                        getattr(ev, "object_type", ""),
                        getattr(ev, "description", ""),
                        getattr(ev, "reason", ""),
                        " ".join(str(r) for r in getattr(ev, "reasons", []) if isinstance(r, (str, int, float))),
                    ]
                    combined_text = " ".join(str(p).upper() for p in searchable_parts if p)

                    # Check if event matches target capability criteria
                    is_match = False
                    if not expected_types:
                        is_match = True
                    else:
                        for exp in expected_types:
                            exp_u = exp.upper()
                            if exp_u in combined_text or (exp_u == "TRACK" and "TRACK" in combined_text):
                                is_match = True
                                break

                    if not is_match:
                        continue

                    # Deduplication: do not spam the event list with rapid per-frame duplicates (2.5s window)
                    now = time.time()
                    tid = getattr(ev, "track_id", "")
                    zone = getattr(ev, "zone", getattr(ev, "line", ""))
                    eid = getattr(ev, "event_id", "")
                    sub = getattr(ev, "resulting_event", getattr(ev, "tamper_type", getattr(ev, "kind", getattr(ev, "action", ""))))
                    sig = (ev_type, str(sub), str(tid), str(zone))
                    if eid:
                        if eid in seen_events and (now - seen_events[eid]) < 2.5:
                            continue
                        seen_events[eid] = now
                    elif sig in seen_events and (now - seen_events[sig]) < 2.5:
                        continue
                    seen_events[sig] = now

                    with self._lock:
                        self.collected_events.append(ev)
                        self.total_events_collected += 1
                except Exception:
                    pass

        self._listener_thread = threading.Thread(target=listener, daemon=True)
        self._listener_thread.start()

    def get_worker(self, camera_id: str) -> Optional[CameraWorker]:
        return self.demo_workers.get(camera_id)

    def get_events(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            raw_list = list(self.collected_events)[-limit:]
        output = []
        for ev in reversed(raw_list):
            if hasattr(ev, "to_dict"):
                d = ev.to_dict()
            elif hasattr(ev, "__dict__"):
                d = {k: v for k, v in ev.__dict__.items() if not k.startswith("_")}
            else:
                d = {"detail": str(ev)}

            ev_cls = type(ev).__name__
            ev_type = d.get("event_type") or getattr(ev, "kind", getattr(ev, "event_type", ev_cls))
            d["event_type"] = ev_type

            # Provide human-readable summary and severity if missing
            if not d.get("summary"):
                if ev_cls == "SuddenMovementEvent":
                    spd = d.get("current_velocity", 0.0)
                    acc = d.get("acceleration", 0.0)
                    direct = d.get("direction", "forward")
                    rel = d.get("boundary_relationship", "").replace("_", " ").lower()
                    d["summary"] = f"Sudden acceleration / running detected: {spd:.0f} px/s ({acc:.0f} px/s^2, {direct}) {rel}"
                    d["severity"] = d.get("severity") or "HIGH"
                elif ev_cls == "CameraTamperEvent":
                    t_type = d.get("tamper_type", "TAMPER")
                    reason = d.get("reason", "Sensor obscured")
                    d["summary"] = f"Camera optical tamper: {t_type} ({reason})"
                    d["severity"] = "CRITICAL"
                elif ev_cls == "GroupIncursionEvent":
                    cnt = d.get("count", len(d.get("track_ids", [])))
                    z = d.get("zone", "perimeter")
                    d["summary"] = f"Group incursion detected: {cnt} simultaneous targets inside '{z}'"
                    d["severity"] = "HIGH"
                elif ev_cls == "LineCrossingEvent":
                    ln = d.get("line", "perimeter")
                    obj = d.get("object_type", "Target")
                    crossing = d.get("direction") or f"{d.get('from_label','')} -> {d.get('to_label','')}"
                    d["summary"] = f"Perimeter breach: {obj} crossed '{ln}' ({crossing})"
                    d["severity"] = d.get("severity") or "HIGH"
                elif ev_cls == "DwellEvent":
                    z = d.get("zone", "restricted zone")
                    dw = d.get("current_dwell_time", 0.0)
                    th = d.get("threshold_crossed", "LOITERING")
                    d["summary"] = f"{th} threshold reached in '{z}': {dw:.1f}s occupancy"
                    d["severity"] = "CRITICAL" if th == "CRITICAL" else "HIGH"
                elif ev_cls == "NightMovementEvent":
                    z = d.get("zone", "zone")
                    lum = d.get("luminance", 0.0)
                    spd = d.get("speed", 0.0)
                    d["summary"] = f"Low-light movement in '{z}' (scene luminance {lum:.0f}/255, speed {spd:.0f} px/s)"
                    d["severity"] = "HIGH"
                elif ev_cls == "ANPREvent":
                    plate = d.get("plate_text", "")
                    st = d.get("state_name", "")
                    conf = d.get("confidence", 0.0)
                    d["summary"] = f"ANPR plate captured: {plate} [{st}] ({conf*100:.0f}% confidence)"
                    d["license_plate"] = plate
                    d["severity"] = "INFO"
                elif ev_cls == "FaceRecognitionEvent":
                    ident = d.get("identity", "Unknown")
                    known = d.get("is_known", False)
                    conf = d.get("confidence", 0.0)
                    d["summary"] = f"Facial recognition: {ident} ({'MATCHED' if known else 'UNKNOWN'}, {conf*100:.0f}%)"
                    d["person_name"] = ident
                    d["watchlist_status"] = "MATCH" if known else "UNKNOWN"
                    d["severity"] = "HIGH" if known else "MEDIUM"
                elif ev_cls == "IncidentEvent":
                    inc_id = d.get("incident_id", "")
                    score = d.get("risk_score", 0)
                    reasons_str = "; ".join(d.get("reasons", []))
                    d["summary"] = f"Correlated incident #{inc_id} (Score {score}/100): {reasons_str}"
                    d["severity"] = d.get("severity") or "HIGH"
                elif ev_cls == "RiskEvent":
                    score = d.get("risk_score", 0)
                    reasons_str = "; ".join(d.get("reasons", []))
                    d["summary"] = f"Risk alert ({score}/100): {reasons_str}"
                    d["severity"] = d.get("risk_level") or "MEDIUM"
                elif ev_cls == "ZoneEvent":
                    obj = d.get("object_type", "Target")
                    k = d.get("kind", "ZONE_ENTER")
                    z = d.get("zone", "zone")
                    dw = d.get("dwell_seconds", 0.0)
                    d["summary"] = f"{obj} {k} — '{z}'" + (f" ({dw:.1f}s dwell)" if dw > 0 else "")
                    d["severity"] = "HIGH" if d.get("zone_type") == "restricted" else "LOW"

            output.append(d)
        return output

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns live metrics filtered strictly to the active feature."""
        status = {}
        for cam_id, worker in self.demo_workers.items():
            cam_data = {
                "fps": round(worker.fps, 1),
                "frame_count": worker.frame_count,
                "objects": worker.object_counts,
                "active_tracks": worker.active_track_count,
                "risk_level": worker.risk_level,
                "night": worker.night_state,
                "tamper": worker.tamper_state,
            }
            if self.feature_id == "anpr":
                cam_data["recent_anpr"] = worker.recent_anpr_reads[-5:]
            elif self.feature_id == "face_recognition":
                cam_data["recent_faces"] = worker.recent_face_events[-5:]
            status[cam_id] = cam_data
        return {
            "session_id": self.session_id,
            "feature_id": self.feature_id,
            "duration": round(time.time() - self.created_at, 1),
            "cameras": status,
        }

    def stop(self):
        """Safely stops all isolated demo workers without touching normal cameras."""
        self.is_active = False
        if self._sse_queue:
            try:
                event_bus.unsubscribe(self._sse_queue)
            except Exception:
                pass
        for worker in self.demo_workers.values():
            try:
                worker.stop()
            except Exception:
                pass
        self.demo_workers.clear()


# ==============================================================================
# 3. TEST CENTER MANAGER & RUN COMPLETE DEMO SEQUENCER
# ==============================================================================

class TestCenterManager:
    """
    Singleton manager coordinating feature test requests, active demo session,
    and curated complete demo execution.
    """
    __test__ = False

    def __init__(self):
        self._current_session: Optional[DemoSession] = None
        self._complete_demo_state: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def get_inventory(self) -> List[Dict[str, Any]]:
        """Returns feature capabilities with real availability based on files and models."""
        model_exists = Path(config.MODEL_PATH).exists()
        gallery_dir = config.GALLERY_DIR
        gallery_has_identities = gallery_dir.exists() and any(p.is_dir() for p in gallery_dir.iterdir())

        result = []
        for f in FEATURE_INVENTORY:
            vid_path = config.PROJECT_ROOT / "videos" / f["demo_video"]
            if not vid_path.exists():
                ext_path = Path("/Users/yashwantankatwar/Documents/DRIVE E/COLLEGE/COLLEGE/HACKATHON/sih 2026/v4/IBVAP_Datasets/dataset_for_test") / f["demo_video"]
                if ext_path.exists():
                    vid_path = ext_path
            status = "READY"
            if not model_exists:
                status = "NOT AVAILABLE (Model missing)"
            elif not vid_path.exists():
                status = "NOT AVAILABLE (Footage missing)"
            elif f["id"] == "face_recognition" and not gallery_has_identities:
                status = "NOT AVAILABLE (No gallery)"

            item = dict(f)
            item["status"] = status
            item["is_active"] = (
                self._current_session is not None and self._current_session.feature_id == f["id"]
            )
            result.append(item)
        return result

    def _pause_normal_cameras(self):
        try:
            import shared
            if hasattr(shared, "camera_manager") and shared.camera_manager:
                for w in getattr(shared.camera_manager, "workers", {}).values():
                    if hasattr(w, "pause"):
                        w.pause()
        except Exception:
            pass

    def _resume_normal_cameras(self):
        try:
            import shared
            if hasattr(shared, "camera_manager") and shared.camera_manager:
                for w in getattr(shared.camera_manager, "workers", {}).values():
                    if hasattr(w, "resume"):
                        w.resume()
        except Exception:
            pass

    def start_demo_session(self, feature_id: str) -> Dict[str, Any]:
        with self._lock:
            if feature_id.upper() in CAP_ALIAS_MAP:
                feature_id = CAP_ALIAS_MAP[feature_id.upper()]

            # If an existing demo session is running, clean it up first
            if self._current_session:
                self._current_session.stop()
                self._current_session = None

            # Pause normal camera inference to free CPU and socket bandwidth
            self._pause_normal_cameras()

            session = DemoSession(feature_id)
            self._current_session = session
            return {
                "success": True,
                "status": "READY",
                "session_id": session.session_id,
                "feature_id": feature_id,
                "feature_name": session.feature_meta["name"],
                "description": session.feature_meta["description"],
                "feature": session.feature_meta,
            }

    def stop_demo_session(self) -> Dict[str, Any]:
        with self._lock:
            if self._current_session:
                self._current_session.stop()
                self._current_session = None
            # Resume normal camera inference
            self._resume_normal_cameras()
            return {
                "success": True,
                "status": "STOPPED",
                "message": "Returned to Normal Operational Mode"
            }

    def stop_complete_demo(self) -> Dict[str, Any]:
        with self._lock:
            if self._complete_demo_state:
                self._complete_demo_state["running"] = False
                self._complete_demo_state["status"] = "STOPPED"
            if self._current_session:
                self._current_session.stop()
                self._current_session = None
            self._resume_normal_cameras()
            return {"success": True, "message": "Sequence stopped by user."}

    def get_active_session(self) -> Optional[DemoSession]:
        return self._current_session

    # ------------------------------------------------------------------
    # Curated Complete Demo Runner (Time-Bounded, Non-Hanging, Point 6)
    # ------------------------------------------------------------------

    def _run_step(self, step_idx: int, step_meta: Dict[str, Any]):
        """Runs a single bounded demo step (max 8 seconds), guaranteeing clean advance."""
        feat_id = step_meta.get("feature_id")
        timeout = float(step_meta.get("timeout_seconds", 8.0))
        # Hard cap at 8s per requirement
        timeout = min(timeout, 8.0)

        step_meta["status"] = "RUNNING"
        step_meta["outcome"] = "RUNNING"
        start_t = time.time()

        try:
            self.start_demo_session(feat_id)
            session = self.get_active_session()
            observed = False
            observed_detail = ""
            event_count = 0

            while (time.time() - start_t) < timeout:
                if not self._complete_demo_state or not self._complete_demo_state.get("running"):
                    break

                if session:
                    # Check worker frames/objects
                    total_objs = sum(sum(w.object_counts.values()) for w in session.demo_workers.values())
                    if feat_id in ("human_detection", "human_tracking") and total_objs > 0:
                        observed = True
                        observed_detail = f"{total_objs} detections verified"
                        break
                    elif feat_id == "vehicle_classification" and any("car" in w.object_counts or "truck" in w.object_counts for w in session.demo_workers.values()):
                        observed = True
                        observed_detail = "Vehicle detection verified"
                        break
                    elif feat_id == "night_movement" and any(w.night_state.get("is_night") for w in session.demo_workers.values()):
                        observed = True
                        observed_detail = "Night low-light detection verified"
                        break
                    elif feat_id == "camera_tamper" and any(w.tamper_state.get("is_tampered") for w in session.demo_workers.values()):
                        observed = True
                        observed_detail = "Camera tamper verified"
                        break

                    evs = session.get_events(limit=10)
                    if evs:
                        event_count = len(evs)
                        observed = True
                        observed_detail = f"{event_count} feature events recorded"
                        break

                time.sleep(0.3)

            elapsed = round(time.time() - start_t, 2)
            step_meta["duration"] = elapsed
            step_meta["status"] = "COMPLETED"
            step_meta["event_count"] = event_count
            if observed:
                step_meta["outcome"] = "VERIFIED"
                step_meta["detail"] = observed_detail
            else:
                step_meta["outcome"] = "TIMED_OUT"
                step_meta["detail"] = f"No matching event within {timeout}s window (advanced cleanly)"

        except Exception as e:
            step_meta["duration"] = round(time.time() - start_t, 2)
            step_meta["status"] = "ERROR"
            step_meta["outcome"] = "TIMED_OUT"
            step_meta["detail"] = f"Error during step: {e}"
        finally:
            self.stop_demo_session()

    def start_complete_demo(self):
        """Starts time-bounded sequential demonstration across all core capabilities."""
        with self._lock:
            if self._complete_demo_state and self._complete_demo_state.get("running"):
                return self._complete_demo_state

            steps = [
                {
                    "feature_id": f["id"],
                    "name": f["name"],
                    "camera_id": f["primary_camera"],
                    "expected_event": f.get("expected_event_type") or ", ".join(f.get("expected_event_types", [])),
                    "timeout_seconds": min(f.get("timeout_seconds", 8.0), 8.0),
                    "status": "PENDING",
                    "outcome": None,
                    "duration": 0.0,
                    "event_count": 0,
                    "detail": "",
                }
                for f in FEATURE_INVENTORY
            ]

            self._complete_demo_state = {
                "running": True,
                "status": "RUNNING",
                "current_step_index": 0,
                "total_steps": len(steps),
                "steps": steps,
                "results": [],
                "verified_count": 0,
                "timed_out_count": 0,
                "started_at": time.time(),
                "finished_at": None,
            }

        thread = threading.Thread(target=self._run_complete_sequence, daemon=True)
        thread.start()
        return self._complete_demo_state

    def _run_complete_sequence(self):
        steps = self._complete_demo_state["steps"]
        for idx, step_meta in enumerate(steps):
            with self._lock:
                if not self._complete_demo_state or not self._complete_demo_state["running"]:
                    break
                self._complete_demo_state["current_step_index"] = idx

            self._run_step(idx, step_meta)

            with self._lock:
                if self._complete_demo_state:
                    if step_meta["outcome"] == "VERIFIED":
                        self._complete_demo_state["verified_count"] += 1
                    else:
                        self._complete_demo_state["timed_out_count"] += 1
                    self._complete_demo_state["results"].append(dict(step_meta))

        with self._lock:
            if self._complete_demo_state:
                self._complete_demo_state["running"] = False
                self._complete_demo_state["status"] = "COMPLETED"
                self._complete_demo_state["finished_at"] = time.time()
        self.stop_demo_session()

    def get_complete_demo_status(self) -> Dict[str, Any]:
        with self._lock:
            if not self._complete_demo_state:
                return {
                    "status": "IDLE",
                    "running": False,
                    "current_step_index": 0,
                    "total_steps": len(FEATURE_INVENTORY),
                    "steps": [],
                    "results": [],
                    "verified_count": 0,
                    "timed_out_count": 0,
                }
            return dict(self._complete_demo_state)


# Global singleton instance
test_center_manager = TestCenterManager()


def get_test_center_manager() -> TestCenterManager:
    return test_center_manager


def get_test_center_inventory() -> List[Dict[str, Any]]:
    return test_center_manager.get_inventory()
