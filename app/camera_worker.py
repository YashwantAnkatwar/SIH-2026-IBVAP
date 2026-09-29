"""
camera_worker.py

Runs the full per-camera pipeline in its own thread:

    VideoSource -> Detector (YOLO + ByteTrack) -> Tracker (history/speed)
        -> ZoneManager (entry/exit/dwell/loitering/group-incursion)
        -> LineCrossingDetector (virtual boundary lines)
        -> RiskEngine (LOW..CRITICAL, + explainable 0-100 score)
        -> AlertManager (dedup, severity, ack/resolve)
        -> EventStore (persistent history for Analytics/Event History)
        -> annotated frame (for the web dashboard) + events published to
           the shared EventBus for the live feed

Design notes / cross-platform fixes vs. the original prototype:
  - No cv2.imshow(): on macOS, OpenCV's GUI functions must run on the
    main thread, so calling imshow() from worker threads (as multi-camera
    operation requires) is unreliable/unsafe. Frames are instead JPEG-
    encoded and served by the Flask dashboard (see dashboard.py), which
    also makes the prototype demoable headlessly (e.g. over SSH) and
    consistently across OSes.
  - Video files loop by default (LOOP_VIDEO_FILES) so a demo can run
    continuously instead of a camera going permanently "offline" once a
    10-second sample clip ends.
"""

import threading
import time

import cv2
import numpy as np

from camera import VideoSource
from detector import Detector
from tracker import Tracker
from zones import ZoneManager, load_zones_for_camera
from lines import LineCrossingDetector, load_lines_for_camera
from risk_engine import RiskEngine, RiskLevel
from night_detection import NightDetector
from tamper_detector import TamperDetector
from events import (
    event_bus, TrackEvent, ZoneEvent, RiskEvent, LineCrossingEvent, StatusEvent,
    ANPREvent, FaceRecognitionEvent, NightMovementEvent,
    CameraTamperEvent, DwellEvent, SuddenMovementEvent, GroupIncursionEvent, IncidentEvent,
)
import shared

import config


ZONE_COLORS = {
    "restricted": (0, 0, 255),   # red (BGR)
    "monitored": (0, 200, 255),  # amber
}

# Maps a distinctive substring of a risk reason to a specific, more
# meaningful event_type than the generic "ZONE_RISK" (Phase 3/4). This is
# just classification of reasons RiskEngine already produced - it does
# not invent any new detection logic.
_REASON_EVENT_TYPE_HINTS = (
    ("Sudden acceleration", "SUDDEN_ACCELERATION"),
    ("running", "SUDDEN_ACCELERATION"),
    ("low-light", "NIGHT_MOVEMENT"),
    ("Loitering", "LOITERING"),
    ("Extended dwell", "EXTENDED_DWELL"),
    ("Group incursion", "GROUP_INCURSION"),
    ("not permitted", "UNAUTHORIZED_ZONE_ENTRY"),
    ("Rapid movement", "UNUSUAL_MOVEMENT"),
)


def _classify_event_type(reasons):
    for reason in reasons:
        for needle, event_type in _REASON_EVENT_TYPE_HINTS:
            if needle.lower() in reason.lower():
                return event_type
    return "ZONE_RISK"


class CameraWorker:

    def __init__(
        self,
        camera_id,
        source,
        model_path,
        confidence=0.35,
        anpr_enabled=None,
        face_enabled=None,
        name=None,
        purpose=None,
        zones=None,
        lines=None,
    ):
        self.camera_id = camera_id
        self.source = str(source)
        self.model_path = str(model_path)
        self.confidence = confidence
        self.name = name or camera_id
        self.purpose = purpose or "General Border Surveillance"

        self.running = False
        self.thread = None

        # Operator controls (Phase 18): pausing keeps the camera thread
        # alive and still reading frames (so it doesn't fall behind /
        # go "stale"), but skips detection, tracking, zones, risk and
        # event generation while paused - a real, load-shedding pause,
        # not a fake UI toggle. Confidence is re-read from this attribute
        # by the detector on every frame, so an operator can raise/lower
        # it live without restarting the camera.
        self.paused = False
        self.confidence = confidence

        self.status = "OFFLINE"
        self.fps = 0.0
        self.frame_count = 0
        self.error_message = ""

        # Latest annotated frame, JPEG-encoded, guarded by a lock since the
        # dashboard thread reads it concurrently with the worker writing it.
        self._frame_lock = threading.Lock()
        self._latest_jpeg = None

        # Snapshot state for the dashboard's status/API endpoints.
        self.object_counts = {}
        self.active_zones = set()
        self.active_track_count = 0
        self.risk_level = RiskLevel.LOW
        self.active_alerts = []
        self._last_evidence_saved_at = 0.0
        self.last_evidence_path = None

        # Camera health (Phase 15).
        self.frame_failures = 0
        self.reconnect_attempts = 0
        self.last_frame_time = 0.0
        self.processing_latency_ms = 0.0

        # Movement heatmap (Phase 13): coarse grid of real observed
        # positions, accumulated over the run. Not fabricated - every
        # increment corresponds to one real tracked detection center.
        self.heatmap_cols = config.HEATMAP_GRID_COLS
        self.heatmap_rows = config.HEATMAP_GRID_ROWS
        self._heatmap_lock = threading.Lock()
        self.heatmap = [[0] * self.heatmap_cols for _ in range(self.heatmap_rows)]

        # ANPR (Priority 1) / Face recognition (Priority 2) pipelines.
        # Created lazily in run() (they load their own offline
        # cascades/etc.) only if enabled, so a baseline run pays zero
        # extra cost. Snapshots below are read by the dashboard.
        self.anpr_pipeline = None
        self.face_pipeline = None
        self.recent_anpr_reads = []   # bounded list of dicts, newest last
        self.recent_face_events = []  # bounded list of dicts, newest last

        # Night/low-light detection (Priority 4) — always on, cheap
        # (one mean() per frame), no separate enable flag needed.
        self.night_state = {"is_night": False, "raw_luminance": None, "smoothed_luminance": None, "threshold": None}

        # Tamper detection (SIH requirement)
        self.tamper_state = {"is_tampered": False, "tamper_type": "NONE", "reason": ""}
        self.simulated_tamper = None

        # Runtime feature flags (per-camera configuration or global default)
        if anpr_enabled is not None:
            self.anpr_enabled = bool(anpr_enabled)
        else:
            self.anpr_enabled = config.ANPR_ENABLED

        if face_enabled is not None:
            self.face_enabled = bool(face_enabled)
        else:
            self.face_enabled = config.FACE_RECOGNITION_ENABLED

        # Geofencing and Lines (reloadable live)
        self.zones = list(zones) if zones is not None else []
        self.lines = list(lines) if lines is not None else []
        self.zone_manager = ZoneManager(self.zones) if self.zones else None
        self.line_detector = LineCrossingDetector(self.lines) if self.lines else None

    # =========================================================
    # START / STOP
    # =========================================================

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3)
        self.status = "OFFLINE"

    def pause(self):
        """Stop running detection/tracking/risk on new frames, but keep
        the camera thread alive (Phase 18). Distinct from stop(): a
        paused camera stays ONLINE and keeps decoding frames so it can
        resume instantly, it just stops analyzing them."""
        self.paused = True

    def resume(self):
        self.paused = False

    def set_confidence(self, value):
        """Update the detection confidence threshold live (Phase 18).
        Takes effect on the next frame - no restart needed."""
        self.confidence = float(value)

    def set_feature(self, feature_name: str, enabled: bool):
        """Toggle ANPR or Face recognition live without restart."""
        key = feature_name.lower().strip()
        if key == "anpr":
            self.anpr_enabled = bool(enabled)
            if not self.anpr_enabled:
                self.anpr_pipeline = None
        elif key in ("face", "face_recognition"):
            self.face_enabled = bool(enabled)
            if not self.face_enabled:
                self.face_pipeline = None

    def reload_zones(self, zones=None):
        if zones is None:
            zone_config = config.load_zone_config()
            zones = load_zones_for_camera(zone_config, self.camera_id)
        self.zones = zones
        self.zone_manager = ZoneManager(zones)

    def reload_lines(self, lines=None):
        if lines is None:
            line_config = config.load_line_config()
            lines = load_lines_for_camera(line_config, self.camera_id)
        self.lines = lines
        self.line_detector = LineCrossingDetector(lines)

    def simulate_tamper(self, tamper_type=None):
        """Set simulated tamper state ('OCCLUSION', 'BLINDING', 'DEFOCUS', or None)."""
        if tamper_type:
            self.simulated_tamper = str(tamper_type).strip().upper()
        else:
            self.simulated_tamper = None

    def get_latest_jpeg(self):
        with self._frame_lock:
            return self._latest_jpeg

    def _set_latest_jpeg(self, jpeg_bytes):
        with self._frame_lock:
            self._latest_jpeg = jpeg_bytes

    def get_heatmap_snapshot(self):
        with self._heatmap_lock:
            return [row[:] for row in self.heatmap]

    def _publish_status(self, status, detail=""):
        self.status = status
        event_bus.publish(StatusEvent(camera_id=self.camera_id, status=status, detail=detail))
        shared.event_store.record(
            camera_id=self.camera_id, event_type="CAMERA_" + status,
            severity="LOW" if status == "ONLINE" else "MEDIUM",
            description=f"{self.camera_id} is now {status}" + (f" — {detail}" if detail else ""),
        )
        if hasattr(shared, "event_store") and hasattr(shared.event_store, "record_audit"):
            try:
                shared.event_store.record_audit(
                    user="SYSTEM", role="AI_ENGINE", action="CAMERA_" + status,
                    target=self.camera_id,
                    details=f"Camera state changed to {status}" + (f" ({detail})" if detail else ""),
                )
            except Exception:
                pass

    # =========================================================
    # MAIN LOOP
    # =========================================================

    def run(self):
        print(f"[{self.camera_id}] Starting worker...")

        try:
            detector = Detector(
                model_path=self.model_path,
                confidence=self.confidence,
                tracker_config=config.TRACKER,
                target_classes=config.TARGET_CLASSES,
            )
            print(f"[{self.camera_id}] YOLO model loaded.")
        except Exception as e:
            print(f"[{self.camera_id}] ERROR loading YOLO model: {e}")
            self._publish_status("ERROR", f"Model load failed: {e}")
            self.error_message = str(e)
            return

        tracker = Tracker(stale_after_seconds=config.TRACK_STALE_SECONDS)
        night_detector = NightDetector(threshold=config.NIGHT_LUMINANCE_THRESHOLD)
        tamper_detector = TamperDetector()

        if self.anpr_enabled:
            from anpr.pipeline import ANPRPipeline
            self.anpr_pipeline = ANPRPipeline()
            print(f"[{self.camera_id}] ANPR pipeline enabled.")

        if self.face_enabled:
            from face.pipeline import FaceRecognitionPipeline
            gallery = shared.get_face_gallery()
            self.face_pipeline = FaceRecognitionPipeline(gallery)
            print(
                f"[{self.camera_id}] Face recognition pipeline enabled "
                f"({len(gallery.enrolled_identities)} identities in gallery)."
            )

        if not self.zones:
            zone_config = config.load_zone_config()
            self.zones = load_zones_for_camera(zone_config, self.camera_id)
            self.zone_manager = ZoneManager(self.zones)
        elif self.zone_manager is None:
            self.zone_manager = ZoneManager(self.zones)

        if not self.lines:
            line_config = config.load_line_config()
            self.lines = load_lines_for_camera(line_config, self.camera_id)
            self.line_detector = LineCrossingDetector(self.lines)
        elif self.line_detector is None:
            self.line_detector = LineCrossingDetector(self.lines)

        risk_engine = RiskEngine(
            dwell_high_seconds=config.RISK_DWELL_HIGH_SECONDS,
            dwell_critical_seconds=config.RISK_DWELL_CRITICAL_SECONDS,
            group_threshold=config.RISK_GROUP_THRESHOLD,
        )

        is_file_source = not self.source.isdigit() and "://" not in self.source

        while self.running:
            print(f"[{self.camera_id}] Opening source: {self.source}")
            video = VideoSource(self.source)

            if not video.open():
                print(f"[{self.camera_id}] ERROR: Could not open video source.")
                self.reconnect_attempts += 1
                self._publish_status("OFFLINE", "Could not open source")
                video.release()
                time.sleep(config.RECONNECT_DELAY)
                continue

            self._publish_status("ONLINE")
            print(f"[{self.camera_id}] Camera ONLINE.")

            frame_counter = 0
            fps_start_time = time.time()
            anpr_last_proc_frame = {}
            face_last_proc_frame = {}

            while self.running:
                anpr_calls_this_frame = 0
                face_calls_this_frame = 0
                success, frame = video.read()

                if not success:
                    self.frame_failures += 1
                    if is_file_source and config.LOOP_VIDEO_FILES:
                        # End of a sample video clip: loop it so the demo
                        # keeps running instead of going "offline".
                        if video.rewind():
                            success, frame = video.read()
                    if not success:
                        print(f"[{self.camera_id}] Video ended or frame read failed.")
                        self._publish_status("RECONNECTING")
                        break

                frame_start = time.time()
                timestamp = frame_start
                self.last_frame_time = timestamp
                frame_h, frame_w = frame.shape[:2]

                if self.simulated_tamper in ("OCCLUSION", "LENS_OCCLUSION"):
                    frame = np.zeros_like(frame)
                elif self.simulated_tamper in ("BLINDING", "LASER_BLINDING"):
                    frame = np.full_like(frame, 255)
                elif self.simulated_tamper in ("DEFOCUS", "DEFOCUS_BLUR", "BLUR"):
                    frame = cv2.GaussianBlur(frame, (51, 51), 0)

                if self.paused:
                    # Real pause: skip inference/tracking/zones/risk
                    # entirely (no fake "frozen" frame pretending to
                    # still analyze). Still serve a live frame so the
                    # operator can see the feed is alive, just labeled.
                    paused_frame = frame.copy()
                    cv2.putText(
                        paused_frame, f"CAMERA: {self.camera_id}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        paused_frame, "PAUSED - analysis suspended", (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
                    )
                    ok, buf = cv2.imencode(
                        ".jpg", paused_frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
                    )
                    if ok:
                        self._set_latest_jpeg(buf.tobytes())
                    time.sleep(1.0 / config.STREAM_TARGET_FPS)
                    continue

                # Pick up any live confidence change (Phase 18) before
                # running inference on this frame.
                detector.confidence = self.confidence

                # Pick up dynamic ANPR/Face runtime toggle
                if self.anpr_enabled and self.anpr_pipeline is None:
                    from anpr.pipeline import ANPRPipeline
                    self.anpr_pipeline = ANPRPipeline()
                if self.face_enabled and self.face_pipeline is None:
                    from face.pipeline import FaceRecognitionPipeline
                    self.face_pipeline = FaceRecognitionPipeline(shared.get_face_gallery())

                try:
                    detections, result = detector.infer(frame)
                except Exception as e:
                    print(f"[{self.camera_id}] Detection/tracking error: {e}")
                    self._publish_status("ERROR", str(e))
                    break

                enriched = tracker.update(detections, timestamp)

                # labels=False: we draw our own single combined label per
                # box (class + track id + confidence) instead of letting
                # ultralytics draw its own on top, which produced cluttered
                # overlapping text with two labels per object.
                annotated_frame = result.plot(labels=False)
                self._draw_zones(annotated_frame, self.zones, frame_w, frame_h)
                self._draw_lines(annotated_frame, self.lines, frame_w, frame_h)

                object_counts = {}
                frame_risk = RiskLevel.LOW
                frame_alerts = []
                current_zone_names = set()

                night_state = night_detector.update(frame)
                self.night_state = night_state

                # Real-time Camera Tamper Detection (Lens occlusion / blinding / defocus)
                tamper_state = tamper_detector.update(frame)
                self.tamper_state = tamper_state
                current_tamper_type = tamper_state.get("tamper_type", "NONE")
                if tamper_state.get("is_tampered"):
                    frame_risk = RiskLevel.max(frame_risk, RiskLevel.HIGH)
                    tamper_desc = f"TAMPER ALERT: {tamper_state.get('reason')}"
                    frame_alerts.append(tamper_desc)
                    is_type_change = (current_tamper_type != getattr(self, "_last_reported_tamper_type", "NONE"))

                    # Save evidence frame for the tamper event
                    tamper_evidence_file = None
                    if config.SAVE_EVIDENCE_ON_CRITICAL and (is_type_change or not hasattr(self, "_last_tamper_evidence_time") or (timestamp - getattr(self, "_last_tamper_evidence_time", 0)) > 15.0):
                        self._last_tamper_evidence_time = timestamp
                        try:
                            tamper_dir = config.EVIDENCE_DIR / "tamper"
                            tamper_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"{self.camera_id}_{int(timestamp)}_{current_tamper_type}.jpg"
                            cv2.imwrite(str(tamper_dir / fname), frame)
                            tamper_evidence_file = f"tamper/{fname}"
                            self.last_evidence_path = tamper_evidence_file
                        except Exception as ee:
                            print(f"[{self.camera_id}] Failed to save tamper evidence: {ee}")

                    _alert_dict, is_new_tamper = self._raise_alert(
                        camera_id=self.camera_id,
                        track_id=f"{self.camera_id}:camera",
                        object_type="camera",
                        event_type=f"CAMERA_TAMPER_{current_tamper_type}",
                        severity="CRITICAL",
                        description=tamper_desc,
                        risk_score=90,
                        reasons=[tamper_state.get("reason", "Camera tampered")],
                        evidence_path=tamper_evidence_file,
                        timestamp=timestamp,
                    )
                    if is_new_tamper or is_type_change:
                        self._last_reported_tamper_type = current_tamper_type
                        tamper_eid = f"TAMPER-{self.camera_id}-{int(timestamp)}"
                        event_bus.publish(CameraTamperEvent(
                            camera_id=self.camera_id,
                            tamper_type=current_tamper_type,
                            reason=tamper_state.get("reason", "Camera tampered"),
                            metrics=tamper_state.get("metrics", {}),
                            event_id=tamper_eid,
                        ))
                        shared.event_store.record(
                            camera_id=self.camera_id,
                            track_id=f"{self.camera_id}:camera",
                            object_type="camera",
                            event_type="CAMERA_TAMPERED",
                            severity="CRITICAL",
                            description=tamper_desc,
                            timestamp=timestamp,
                            event_id=tamper_eid,
                            evidence_path=tamper_evidence_file,
                            tamper_type=current_tamper_type,
                            tamper_metrics=tamper_state.get("metrics"),
                        )
                        if hasattr(shared, "event_store") and hasattr(shared.event_store, "record_audit"):
                            try:
                                shared.event_store.record_audit(
                                    user="SYSTEM", role="AI_ENGINE", action="TAMPER_DETECTED",
                                    target=self.camera_id, details=tamper_desc
                                )
                            except Exception:
                                pass
                else:
                    if getattr(self, "_last_reported_tamper_type", "NONE") != "NONE":
                        old_type = self._last_reported_tamper_type
                        self._last_reported_tamper_type = "NONE"
                        if hasattr(shared, "event_store") and hasattr(shared.event_store, "record_audit"):
                            try:
                                shared.event_store.record_audit(
                                    user="SYSTEM", role="AI_ENGINE", action="TAMPER_RESOLVED",
                                    target=self.camera_id, details=f"Camera optical sensor restored to normal (cleared from {old_type})"
                                )
                            except Exception:
                                pass

                risk_engine.begin_frame()
                self.zone_manager.begin_frame()
                per_track_results = []

                # First pass: per-track zone membership + base risk, line
                # crossings, and populate zone/risk occupancy so group
                # incursions can be detected in the second pass.
                for det in enriched:
                    object_counts[det["class_name"]] = object_counts.get(det["class_name"], 0) + 1
                    self._accumulate_heatmap(det["center"], frame_w, frame_h)

                    global_track_id = f"{self.camera_id}:{det['track_id']}"

                    zone_events, zones_now = self.zone_manager.update(
                        det["track_id"], det["class_name"], det["center"],
                        frame_w, frame_h, timestamp,
                    )
                    current_zone_names |= zones_now

                    zone_for_alert = next(iter(zones_now), None)

                    # ANPR (Priority 1): only ever called for vehicle-class
                    # detections. Returns a result only on the single frame
                    # a track's plate first becomes temporally stable.
                    if self.anpr_pipeline is not None and det["class_name"] in config.ANPR_VEHICLE_CLASSES:
                        agg_state = getattr(self.anpr_pipeline, "_aggregator", None)
                        track_plate_state = agg_state._tracks.get(global_track_id) if agg_state else None
                        already_reported = track_plate_state.reported if track_plate_state else False
                        if not already_reported:
                            last_f = anpr_last_proc_frame.get(global_track_id, -99)
                            if (self.frame_count - last_f) >= 3 and anpr_calls_this_frame < 2:
                                anpr_last_proc_frame[global_track_id] = self.frame_count
                                anpr_calls_this_frame += 1
                                self._process_anpr(det, frame, global_track_id, timestamp)

                    # Face recognition (Priority 2): only ever called for
                    # person-class detections. Returns a result only on the
                    # frame a track's identity first becomes stable (or
                    if self.face_pipeline is not None and det["class_name"] == "person":
                        face_tracks = getattr(self.face_pipeline, "_tracks", {})
                        person_face_state = face_tracks.get(global_track_id)
                        already_face_reported = bool(getattr(person_face_state, "reported_identity", None) or getattr(person_face_state, "reported", False))
                        if not already_face_reported:
                            last_ff = face_last_proc_frame.get(global_track_id, -99)
                            if (self.frame_count - last_ff) >= 3 and face_calls_this_frame < 2:
                                face_last_proc_frame[global_track_id] = self.frame_count
                                face_calls_this_frame += 1
                                self._process_face(det, frame, global_track_id, timestamp)

                    for ev in zone_events:
                        if ev["type"] == "ZONE_DWELL":
                            continue  # too frequent to log; ZONE_ENTER/EXIT/behavioral events are enough
                        zone_sev = ev.get("zone_severity") or ("HIGH" if ev.get("zone_type") == "restricted" else "LOW")
                        if ev.get("violation"):
                            zone_sev = "CRITICAL" if ev.get("zone_type") == "restricted" else "HIGH"
                        event_bus.publish(ZoneEvent(
                            camera_id=self.camera_id,
                            track_id=global_track_id,
                            object_type=det["class_name"],
                            zone=ev["zone"],
                            zone_type=ev["zone_type"],
                            dwell_seconds=ev["dwell_seconds"],
                            kind=ev["type"],
                        ))
                        if ev["type"] in ("EXTENDED_DWELL", "LOITERING"):
                            event_bus.publish(DwellEvent(
                                camera_id=self.camera_id,
                                track_id=global_track_id,
                                zone=ev["zone"],
                                entry_time=float(ev.get("entry_time", timestamp - ev["dwell_seconds"])),
                                current_dwell_time=float(ev["dwell_seconds"]),
                                threshold_crossed=str(ev.get("threshold_crossed", "WARNING" if ev["type"] == "EXTENDED_DWELL" else "CRITICAL")),
                                threshold_seconds=float(ev.get("threshold_seconds", 5.0 if ev["type"] == "EXTENDED_DWELL" else 15.0)),
                                object_type=det["class_name"],
                                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)),
                                event_id=f"DWELL-{self.camera_id}-{int(timestamp)}-{det['track_id']}",
                            ))
                        shared.event_store.record(
                            camera_id=self.camera_id, track_id=global_track_id,
                            object_type=det["class_name"], zone=ev["zone"],
                            event_type=ev["type"], severity=zone_sev,
                            description=f"{det['class_name']} {ev['type']} — {ev['zone']}",
                            dwell_seconds=ev["dwell_seconds"], timestamp=timestamp,
                            confidence=det.get("confidence"),
                            direction=det.get("direction"),
                            bounding_box=det.get("box"),
                            threshold_crossed=ev.get("threshold_crossed"),
                            threshold_seconds=ev.get("threshold_seconds"),
                        )

                    # Virtual line crossings (Phase 4).
                    for ev in self.line_detector.update(
                        det["track_id"], det["center"], frame_w, frame_h, timestamp=timestamp
                    ):
                        crossing_dir = ev.get("crossing_direction", f"{ev['from_label']} -> {ev['to_label']}")
                        is_inbound = "inside" in ev["to_label"].lower() or "restricted" in ev["to_label"].lower()
                        crossing_sev = "HIGH" if (is_inbound and ("zero" in ev["line"].lower() or "restricted" in ev["line"].lower() or "fence" in ev["line"].lower())) else ("MEDIUM" if is_inbound else "LOW")

                        event_bus.publish(LineCrossingEvent(
                            camera_id=self.camera_id, track_id=global_track_id,
                            object_type=det["class_name"], line=ev["line"],
                            direction=crossing_dir, from_label=ev["from_label"],
                            to_label=ev["to_label"],
                        ))
                        description = (
                            f"{det['class_name']} #{det['track_id']} crossed '{ev['line']}': "
                            f"{crossing_dir}"
                        )
                        shared.event_store.record(
                            camera_id=self.camera_id, track_id=global_track_id,
                            object_type=det["class_name"],
                            zone=ev["line"],
                            event_type="LINE_CROSSING",
                            severity=crossing_sev,
                            confidence=det.get("confidence"),
                            description=description, timestamp=timestamp,
                            direction=crossing_dir,
                            bounding_box=det.get("box"),
                        )
                        if is_inbound:
                            self._raise_alert(
                                camera_id=self.camera_id, track_id=global_track_id,
                                object_type=det["class_name"], event_type="LINE_CROSSING",
                                severity=crossing_sev, description=description,
                                zone=ev["line"], risk_score=50 if crossing_sev == "HIGH" else 40,
                                reasons=[description],
                                timestamp=timestamp,
                            )

                    level, reasons = risk_engine.evaluate_track(
                        det["track_id"], det["class_name"], zone_events
                    )

                    # Night movement (Priority 4): a real, per-frame
                    # luminance measurement (see night_detection.py), not
                    # a per-video label. Only escalates when there's BOTH
                    # low light AND actual movement in a zone — matching
                    # the brief's own example exactly (movement + zone +
                    # low-light = one combined reason, not three separate
                    # generic flags).
                    if night_state["is_night"] and zones_now and det["speed"] > 5:
                        zone_label = next(iter(zones_now))
                        reasons.append(
                            f"Movement detected in '{zone_label}' under low-light conditions "
                            f"(scene luminance {night_state['smoothed_luminance']:.0f}/255, "
                            f"threshold {night_state['threshold']:.0f})"
                        )
                        level = RiskLevel.max(level, RiskLevel.HIGH)

                    # Sudden speed / running / acceleration (Phase 4)
                    speed_val = det.get("speed", 0.0)
                    accel_val = det.get("acceleration", 0.0)
                    prev_v = det.get("prev_velocity", 0.0)
                    if det["class_name"] == "person" and det.get("age_seconds", 0.0) > 0.15:
                        if speed_val >= 25.0 and (abs(accel_val) >= 28.0 or (speed_val >= 32.0 and prev_v < 24.0)):
                            boundary_rel = "INSIDE_RESTRICTED_ZONE" if zones_now else "TOWARD_RESTRICTED_PERIMETER"
                            accel_reason = (
                                f"Sudden acceleration/running detected ({speed_val:.0f} px/s, "
                                f"prev {prev_v:.0f} px/s, accel {accel_val:.0f} px/s^2, {det.get('direction', 'UNKNOWN')}) {boundary_rel}"
                            )
                            reasons.append(accel_reason)
                            level = RiskLevel.max(level, RiskLevel.HIGH if zones_now else RiskLevel.MEDIUM)
                            last_accel_pub = getattr(self, "_last_accel_pub", {})
                            if (timestamp - last_accel_pub.get(global_track_id, 0)) >= 2.0:
                                last_accel_pub[global_track_id] = timestamp
                                self._last_accel_pub = last_accel_pub
                                accel_eid = f"ACCEL-{self.camera_id}-{int(timestamp)}-{det['track_id']}"
                                event_bus.publish(SuddenMovementEvent(
                                    camera_id=self.camera_id,
                                    track_id=global_track_id,
                                    previous_velocity=float(prev_v),
                                    current_velocity=float(speed_val),
                                    direction=str(det.get("direction", "UNKNOWN")),
                                    acceleration=float(accel_val),
                                    boundary_relationship=boundary_rel,
                                    resulting_event="SUDDEN_ACCELERATION",
                                    object_type="person",
                                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(timestamp)),
                                    event_id=accel_eid,
                                ))
                                shared.event_store.record(
                                    camera_id=self.camera_id,
                                    track_id=global_track_id,
                                    object_type="person",
                                    zone=zone_for_alert or "",
                                    event_type="SUDDEN_ACCELERATION",
                                    severity="HIGH" if zones_now else "MEDIUM",
                                    description=accel_reason,
                                    timestamp=timestamp,
                                    event_id=accel_eid,
                                    speed=float(speed_val),
                                    acceleration=float(accel_val),
                                    direction=str(det.get("direction", "UNKNOWN")),
                                )
                    elif det["speed"] > 400 and det["age_seconds"] > 0.3:
                        reasons.append(
                            f"Rapid movement detected ({det['speed']:.0f} px/s)"
                        )
                        level = RiskLevel.max(level, RiskLevel.MEDIUM)

                    per_track_results.append((det, level, reasons, global_track_id, zone_for_alert))

                # Second pass: apply group-incursion escalation now that
                # zone occupancy for this frame is fully known.
                for det, level, reasons, global_track_id, zone_for_alert in per_track_results:
                    level, reasons = risk_engine.apply_group_escalation(level, reasons)

                    if level != RiskLevel.LOW:
                        frame_risk = RiskLevel.max(frame_risk, level)
                        alert_text = f"{det['class_name']} #{det['track_id']}: {'; '.join(reasons)}"
                        frame_alerts.append(alert_text)
                        score, _band = RiskEngine.score_for(level, reasons)
                        event_type = _classify_event_type(reasons)

                        # The alert manager is the source of truth for
                        # dedup: a track sitting in a restricted zone keeps
                        # re-evaluating as risky every single frame, but we
                        # only want ONE alert/history/live-feed entry per
                        # distinct occurrence (Phase 7), not one per frame.
                        # `is_new` also covers severity re-escalation of an
                        # existing alert being worth logging again.
                        _alert_dict, is_new_occurrence = self._raise_alert(
                            camera_id=self.camera_id, track_id=global_track_id,
                            object_type=det["class_name"], event_type=event_type,
                            severity=level, description=alert_text, zone=zone_for_alert,
                            risk_score=score, reasons=reasons, timestamp=timestamp,
                        )

                        if is_new_occurrence:
                            event_bus.publish(RiskEvent(
                                camera_id=self.camera_id,
                                track_id=global_track_id,
                                object_type=det["class_name"],
                                risk_level=level,
                                reasons=reasons,
                                zone=zone_for_alert or "",
                                risk_score=score,
                                event_type=event_type,
                            ))
                            night_event_id = None
                            if event_type == "NIGHT_MOVEMENT":
                                night_event_id = f"NIGHT-{self.camera_id}-{int(timestamp)}-{det['track_id']}"
                                event_bus.publish(NightMovementEvent(
                                    camera_id=self.camera_id,
                                    track_id=global_track_id,
                                    zone=zone_for_alert or "",
                                    speed=float(det["speed"]),
                                    luminance=float(night_state.get("smoothed_luminance", 0.0)),
                                    threshold=float(night_state.get("threshold", 70.0)),
                                    is_night=bool(night_state.get("is_night", False)),
                                    event_id=night_event_id,
                                ))
                            shared.event_store.record(
                                camera_id=self.camera_id, track_id=global_track_id,
                                object_type=det["class_name"], zone=zone_for_alert,
                                event_type=event_type, severity=level,
                                confidence=det["confidence"], description=alert_text,
                                timestamp=timestamp, reasons=reasons, risk_score=score,
                                event_id=night_event_id,
                                scene_luminance=night_state.get("smoothed_luminance"),
                                is_night=night_state.get("is_night"),
                            )

                    if det["is_new"]:
                        event_bus.publish(TrackEvent(
                            camera_id=self.camera_id,
                            track_id=global_track_id,
                            object_type=det["class_name"],
                            confidence=det["confidence"],
                            center_x=det["center"][0],
                            center_y=det["center"][1],
                        ))
                        shared.event_store.record(
                            camera_id=self.camera_id, track_id=global_track_id,
                            object_type=det["class_name"], event_type="OBJECT_DETECTED",
                            severity="LOW", confidence=det["confidence"],
                            description=f"New {det['class_name']} detected (#{det['track_id']})",
                            timestamp=timestamp,
                        )

                    self._draw_track_label(annotated_frame, det)

                # Zone-level (not per-track) violations, e.g. too many
                # simultaneous objects in a max_objects-limited zone.
                for zv in self.zone_manager.zone_violations():
                    frame_risk = RiskLevel.max(frame_risk, "HIGH")
                    description = (
                        f"{zv['count']} objects simultaneously in zone '{zv['zone']}' "
                        f"(limit {zv['max_objects']})"
                    )
                    frame_alerts.append(description)
                    _alert_dict, is_new_occurrence = self._raise_alert(
                        camera_id=self.camera_id, track_id=f"{self.camera_id}:{zv['zone']}",
                        object_type="group", event_type=zv["type"], severity="HIGH",
                        description=description, zone=zv["zone"], risk_score=60,
                        reasons=[description], timestamp=timestamp,
                    )
                    if is_new_occurrence:
                        zv_eid = f"GROUP-{self.camera_id}-{int(timestamp)}-{zv['zone']}"
                        event_bus.publish(GroupIncursionEvent(
                            camera_id=self.camera_id,
                            zone=zv["zone"],
                            track_ids=list(self.zone_manager._frame_occupancy.get(zv["zone"], set())),
                            count=zv["count"],
                            event_id=zv_eid,
                        ))
                        shared.event_store.record(
                            camera_id=self.camera_id, zone=zv["zone"],
                            track_id=f"{self.camera_id}:{zv['zone']}",
                            object_type="group", event_type=zv["type"],
                            severity="HIGH", description=description,
                            timestamp=timestamp,
                            event_id=zv_eid,
                            member_count=zv["count"],
                        )

                # Spatial group incursion detection (>=3 targets clustered together in restricted/monitored zone)
                for zone in self.zones:
                    if zone.zone_type in ("restricted", "monitored") or zone.max_objects:
                        occupants = [
                            (d["track_id"], d["center"]) for d, _, _, _, _ in per_track_results
                            if zone.contains(d["center"][0], d["center"][1], frame_w, frame_h)
                        ]
                        if len(occupants) >= 3:
                            clusters = RiskEngine.find_spatial_clusters(occupants, cluster_radius=280.0)
                            for cluster_tids in clusters:
                                if len(cluster_tids) >= 3:
                                    group_desc = (
                                        f"Group incursion: {len(cluster_tids)} targets clustered together "
                                        f"in zone '{zone.name}' (tracks: {', '.join(str(t) for t in cluster_tids)})"
                                    )
                                    frame_alerts.append(group_desc)
                                    frame_risk = RiskLevel.max(frame_risk, RiskLevel.HIGH)
                                    _alert_dict, is_new_group = self._raise_alert(
                                        camera_id=self.camera_id,
                                        track_id=f"{self.camera_id}:group:{zone.name}",
                                        object_type="group",
                                        event_type="GROUP_INCURSION",
                                        severity="HIGH",
                                        description=group_desc,
                                        zone=zone.name,
                                        risk_score=70,
                                        reasons=[group_desc],
                                        timestamp=timestamp,
                                    )
                                    if is_new_group:
                                        group_eid = f"GROUP-{self.camera_id}-{int(timestamp)}-{zone.name}"
                                        event_bus.publish(GroupIncursionEvent(
                                            camera_id=self.camera_id,
                                            zone=zone.name,
                                            track_ids=[f"{self.camera_id}:{t}" for t in cluster_tids],
                                            count=len(cluster_tids),
                                            event_id=group_eid,
                                        ))
                                        shared.event_store.record(
                                            camera_id=self.camera_id,
                                            zone=zone.name,
                                            track_id=f"{self.camera_id}:group:{zone.name}",
                                            object_type="group",
                                            event_type="GROUP_INCURSION",
                                            severity="HIGH",
                                            description=group_desc,
                                            timestamp=timestamp,
                                            event_id=group_eid,
                                            member_tracks=[f"{self.camera_id}:{t}" for t in cluster_tids],
                                            member_count=len(cluster_tids),
                                        )

                stale_ids = tracker.prune_stale(timestamp)
                for tid in stale_ids:
                    self.zone_manager.forget_track(tid)
                    self.line_detector.forget_track(tid)
                    global_tid = f"{self.camera_id}:{tid}"
                    if self.anpr_pipeline is not None:
                        self.anpr_pipeline.forget_track(global_tid)
                    if self.face_pipeline is not None:
                        self.face_pipeline.forget_track(global_tid)
                    anpr_last_proc_frame.pop(global_tid, None)
                    face_last_proc_frame.pop(global_tid, None)
                self.zone_manager.cleanup(tracker.active_track_ids())
                self.line_detector.cleanup(tracker.active_track_ids())

                self.object_counts = object_counts
                self.active_zones = current_zone_names
                self.active_track_count = len(enriched)
                self.risk_level = frame_risk
                self.active_alerts = frame_alerts[-5:]
                self.processing_latency_ms = (time.time() - frame_start) * 1000.0

                # ---- FPS ----
                frame_counter += 1
                self.frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    self.fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_start_time = time.time()

                self._draw_hud(annotated_frame)

                if frame_risk == RiskLevel.CRITICAL:
                    self._maybe_save_evidence(annotated_frame, timestamp)

                ok, buf = cv2.imencode(
                    ".jpg", annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY],
                )
                if ok:
                    self._set_latest_jpeg(buf.tobytes())

                # Rate pacing for file sources: maintain smooth target display/processing rate (~12-15 FPS)
                if is_file_source and getattr(config, "TARGET_PROCESSING_FPS", 0) > 0:
                    target_interval = 1.0 / config.TARGET_PROCESSING_FPS
                    proc_duration = time.time() - frame_start
                    sleep_time = target_interval - proc_duration
                    if sleep_time > 0.001:
                        time.sleep(sleep_time)

            video.release()

            if self.running:
                self.reconnect_attempts += 1
                self._publish_status("RECONNECTING")
                print(f"[{self.camera_id}] Reconnecting in {config.RECONNECT_DELAY} seconds...")
                time.sleep(config.RECONNECT_DELAY)

        self._publish_status("OFFLINE")
        print(f"[{self.camera_id}] Worker stopped.")

    def _raise_alert(self, **kwargs):
        try:
            alert_dict, is_new = shared.alert_manager.raise_alert(**kwargs)
            if alert_dict and hasattr(shared, "correlator") and shared.correlator is not None:
                try:
                    inc_dict, is_new_inc = shared.correlator.correlate(
                        camera_id=alert_dict.get("camera_id", self.camera_id),
                        track_id=alert_dict.get("track_id", ""),
                        object_type=alert_dict.get("object_type", "unknown"),
                        event_type=alert_dict.get("event_type", ""),
                        severity=alert_dict.get("severity", "LOW"),
                        description=alert_dict.get("description", ""),
                        zone=alert_dict.get("zone", ""),
                        risk_score=alert_dict.get("risk_score", 0),
                        reasons=alert_dict.get("reasons", []),
                        timestamp=alert_dict.get("timestamp", time.time()),
                    )
                    if is_new_inc:
                        event_bus.publish(IncidentEvent(
                            incident_id=inc_dict["incident_id"],
                            camera_id=inc_dict["camera_id"],
                            primary_track_id=inc_dict["primary_track_id"],
                            object_type=inc_dict["object_type"],
                            severity=inc_dict["severity"],
                            risk_score=inc_dict["risk_score"],
                            event_count=len(inc_dict.get("events", [])),
                            reasons=inc_dict.get("reasons", []),
                            status=inc_dict.get("status", "ACTIVE"),
                        ))
                except Exception as ce:
                    pass
            if is_new and alert_dict and alert_dict.get("severity") == "CRITICAL":
                if hasattr(shared, "event_store") and hasattr(shared.event_store, "record_audit"):
                    try:
                        shared.event_store.record_audit(
                            user="AI_ENGINE", role="OFFICER", action="CRITICAL_THREAT",
                            target=self.camera_id,
                            details=alert_dict.get("description", "Critical threat detected")
                        )
                    except Exception:
                        pass
            return alert_dict, is_new
        except Exception as e:  # alerting must never crash the camera loop
            print(f"[{self.camera_id}] Alert manager error: {e}")
            return None, False

    def _process_anpr(self, det, frame, global_track_id, timestamp):
        """Priority 1. Called only for vehicle-class detections. Real
        pipeline: localize plate -> crop -> OCR -> temporally aggregate.
        Only fires an event/evidence/alert the single frame a track's
        plate first becomes stable — see anpr/aggregator.py."""
        if not self.anpr_enabled:
            return
        try:
            result = self.anpr_pipeline.process(
                self.camera_id, global_track_id, frame, det["box"], timestamp,
            )
        except Exception as e:  # ANPR must never crash the camera loop
            print(f"[{self.camera_id}] ANPR pipeline error: {e}")
            return
        if result is None:
            return

        evidence_filename = None
        if config.ANPR_SAVE_EVIDENCE and result.evidence_crop is not None and result.evidence_crop.size > 0:
            config.ANPR_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            ts_label = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
            bare_filename = f"{self.camera_id}_{ts_label}_{result.plate_text}.jpg"
            try:
                cv2.imwrite(str(config.ANPR_EVIDENCE_DIR / bare_filename), result.evidence_crop)
                # Stored relative to config.EVIDENCE_DIR (not ANPR_EVIDENCE_DIR)
                # since that's the root the dashboard's /evidence/<path:filename>
                # route serves from — the "anpr/" prefix keeps this file
                # correctly reachable there.
                evidence_filename = f"anpr/{bare_filename}"
            except Exception as e:
                print(f"[{self.camera_id}] Failed to save ANPR evidence: {e}")
                evidence_filename = None

        ts_label = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
        event_id = f"ANPR-{self.camera_id}-{ts_label}-{global_track_id.replace(':', '_')}"
        state_tag = f" [{result.state_name}]" if getattr(result, "state_name", "") else ""
        description = (
            f"ANPR: vehicle {global_track_id} plate read as '{result.plate_text}'{state_tag} "
            f"(confidence {result.confidence:.0f})"
        )
        event_bus.publish(ANPREvent(
            camera_id=self.camera_id, vehicle_track_id=global_track_id,
            plate_text=result.plate_text, confidence=result.confidence,
            event_id=event_id,
            plate_box=getattr(result, "plate_box", ()),
            supporting_frames=getattr(result, "supporting_frames", 1),
            is_indian=getattr(result, "is_indian", False),
            state_name=getattr(result, "state_name", ""),
            evidence_path=evidence_filename or "",
        ))
        shared.event_store.record(
            camera_id=self.camera_id, track_id=global_track_id, object_type="vehicle",
            event_type="ANPR_PLATE_READ", severity="LOW", description=description,
            timestamp=timestamp, plate_text=result.plate_text,
            plate_confidence=result.confidence, evidence_path=evidence_filename,
            event_id=event_id,
            plate_box=getattr(result, "plate_box", ()),
            supporting_frames=getattr(result, "supporting_frames", 1),
            is_indian=getattr(result, "is_indian", False),
            state_name=getattr(result, "state_name", ""),
        )

        record = {
            "camera_id": self.camera_id, "vehicle_track_id": global_track_id,
            "plate_text": result.plate_text, "confidence": result.confidence,
            "timestamp": timestamp, "evidence_path": evidence_filename,
            "event_id": event_id,
            "is_indian": getattr(result, "is_indian", False),
            "state_name": getattr(result, "state_name", ""),
            "supporting_frames": getattr(result, "supporting_frames", 1),
        }
        self.recent_anpr_reads.append(record)
        self.recent_anpr_reads = self.recent_anpr_reads[-50:]

    def _process_face(self, det, frame, global_track_id, timestamp):
        """Priority 2. Called only for person-class detections. Real
        pipeline: detect face -> quality filter -> LBP embedding ->
        gallery match -> temporally stabilize. Only fires an
        event/evidence when a track's identity first becomes stable, or
        changes to a different stable identity — see face/pipeline.py."""
        try:
            result = self.face_pipeline.process(
                self.camera_id, global_track_id, frame, det["box"], timestamp,
            )
        except Exception as e:  # face recognition must never crash the camera loop
            print(f"[{self.camera_id}] Face recognition pipeline error: {e}")
            return
        if result is None:
            return

        ts_label = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
        event_id = f"FRS-{self.camera_id}-{ts_label}-{global_track_id.replace(':', '_')}"
        description = (
            f"Face recognition: person {global_track_id} identified as "
            f"'{result.identity}' (confidence {result.confidence:.2f})"
        )

        evidence_filename = None
        # Only save evidence for KNOWN matches — per the brief, "don't
        # create thousands of duplicate screenshots"; an Unknown result
        # is common and not itself evidence-worthy, a Known match is.
        if (config.FACE_SAVE_EVIDENCE and result.is_known
                and result.evidence_crop is not None and result.evidence_crop.size > 0):
            config.FACE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
            bare_filename = f"{self.camera_id}_{ts_label}_{result.identity}.jpg"
            try:
                cv2.imwrite(str(config.FACE_EVIDENCE_DIR / bare_filename), result.evidence_crop)
                evidence_filename = f"face/{bare_filename}"
            except Exception as e:
                print(f"[{self.camera_id}] Failed to save face evidence: {e}")

        event_bus.publish(FaceRecognitionEvent(
            camera_id=self.camera_id, person_track_id=global_track_id,
            identity=result.identity, confidence=result.confidence, is_known=result.is_known,
            event_id=event_id,
            similarity=getattr(result, "similarity", 0.0),
            threshold=getattr(result, "threshold", 0.42),
            face_box=getattr(result, "face_box", ()),
            evidence_path=evidence_filename or "",
        ))
        shared.event_store.record(
            camera_id=self.camera_id, track_id=global_track_id, object_type="person",
            event_type="FACE_RECOGNITION", severity="MEDIUM" if result.is_known else "LOW",
            description=description, timestamp=timestamp,
            identity=result.identity, recognition_confidence=result.confidence,
            recognition_status="KNOWN" if result.is_known else "UNKNOWN",
            evidence_path=evidence_filename,
            event_id=event_id,
            similarity=getattr(result, "similarity", 0.0),
            threshold=getattr(result, "threshold", 0.42),
            face_box=getattr(result, "face_box", ()),
        )

        record = {
            "camera_id": self.camera_id, "person_track_id": global_track_id,
            "identity": result.identity, "confidence": result.confidence,
            "is_known": result.is_known, "timestamp": timestamp,
            "event_id": event_id,
            "similarity": getattr(result, "similarity", 0.0),
            "evidence_path": evidence_filename,
        }
        self.recent_face_events.append(record)
        self.recent_face_events = self.recent_face_events[-50:]

        # A recognized (known) identity lingering/entering a restricted
        # zone is exactly the kind of "combine observations into an
        # incident" case the brief calls for (Section 6/7); the zone/risk
        # engine already raises the underlying intrusion alert, this just
        # makes sure the alert has the person's identity attached when
        # one is known, rather than only a bare track id.
        if result.is_known:
            self._raise_alert(
                camera_id=self.camera_id, track_id=global_track_id,
                object_type="person", event_type="FACE_RECOGNITION_KNOWN",
                severity="MEDIUM", description=description,
                zone=None, risk_score=30, reasons=[description],
                timestamp=timestamp,
            )

    def _accumulate_heatmap(self, center, frame_w, frame_h):
        if frame_w <= 0 or frame_h <= 0:
            return
        x, y = center
        col = min(self.heatmap_cols - 1, max(0, int((x / frame_w) * self.heatmap_cols)))
        row = min(self.heatmap_rows - 1, max(0, int((y / frame_h) * self.heatmap_rows)))
        with self._heatmap_lock:
            self.heatmap[row][col] += 1

    def _maybe_save_evidence(self, frame, timestamp):
        if not config.SAVE_EVIDENCE_ON_CRITICAL:
            return
        if (timestamp - self._last_evidence_saved_at) < config.EVIDENCE_MIN_INTERVAL_SECONDS:
            return

        self._last_evidence_saved_at = timestamp
        config.EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        ts_label = time.strftime("%Y%m%d-%H%M%S", time.localtime(timestamp))
        filename = f"{self.camera_id}_{ts_label}_CRITICAL.jpg"
        filepath = config.EVIDENCE_DIR / filename
        try:
            cv2.imwrite(str(filepath), frame)
            self.last_evidence_path = filename
            shared.event_store.record(
                camera_id=self.camera_id, event_type="EVIDENCE_CAPTURED",
                severity="CRITICAL", description=f"Evidence snapshot saved: {filename}",
                timestamp=timestamp, evidence_path=filename,
            )
        except Exception as e:
            print(f"[{self.camera_id}] Failed to save evidence snapshot: {e}")

    # =========================================================
    # DRAWING HELPERS
    # =========================================================

    def _draw_zones(self, frame, zones, frame_w, frame_h):
        for zone in zones:
            pts = zone.pixel_polygon(frame_w, frame_h)
            pts_int = [(int(x), int(y)) for x, y in pts]
            color = ZONE_COLORS.get(zone.zone_type, (255, 255, 255))
            for i in range(len(pts_int)):
                cv2.line(frame, pts_int[i], pts_int[(i + 1) % len(pts_int)], color, 2)
            if pts_int:
                cv2.putText(
                    frame, zone.name, pts_int[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )

    def _draw_lines(self, frame, lines, frame_w, frame_h):
        for line in lines:
            point_a, point_b = line.pixel_points(frame_w, frame_h)
            pa = (int(point_a[0]), int(point_a[1]))
            pb = (int(point_b[0]), int(point_b[1]))
            cv2.line(frame, pa, pb, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame, line.name, (pa[0] + 4, pa[1] + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA,
            )

    def _draw_track_label(self, frame, det):
        x1, y1, x2, y2 = det["box"]
        # Explicit bounding box around detected and tracked person/object
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        speed_val = det.get("speed", 0.0)
        dir_val = det.get("direction", "STATIONARY")
        motion_tag = f" [{dir_val} {speed_val:.0f}px/s]" if speed_val >= 8.0 else " [STATIONARY]"

        global_tid = f"{self.camera_id}:{det['track_id']}"
        plate_tag = ""
        if self.anpr_pipeline:
            cur_p = self.anpr_pipeline.current_plate_for(global_tid)
            if cur_p and cur_p[0]:
                plate_tag = f" [{cur_p[0]}]"

        face_tag = ""
        if self.face_pipeline:
            cur_f = self.face_pipeline.current_identity_for(global_tid)
            if cur_f and cur_f[0]:
                face_tag = f" [{cur_f[0]}]"

        label = f"{det['class_name']} #{det['track_id']} {det['confidence']:.2f}{motion_tag}{plate_tag}{face_tag}"
        label_pos = (int(x1), max(int(y1) - 8, 14))
        cv2.putText(
            frame, label, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_hud(self, frame):
        frame_h, frame_w = frame.shape[:2]
        # Compact adaptive font scale (unobtrusive, reduced footprint)
        scale = max(0.30, min(0.40, frame_h / 700.0))
        thick = 1
        line_h = int(scale * 34)
        y_start = int(scale * 32)
        x_pad = 8

        risk_color = {
            RiskLevel.LOW: (0, 220, 0),
            RiskLevel.MEDIUM: (0, 200, 255),
            RiskLevel.HIGH: (0, 120, 255),
            RiskLevel.CRITICAL: (0, 0, 255),
        }.get(self.risk_level, (255, 255, 255))

        lines = [
            (f"FPS: {self.fps:.1f}", (0, 255, 0)),
            (f"STATUS: {self.status}", (0, 255, 0)),
            (f"RISK: {self.risk_level}", risk_color),
        ]
        if self.tamper_state.get("is_tampered"):
            lines.append((f"TAMPER: {self.tamper_state.get('tamper_type')}", (0, 0, 255)))

        # Draw neat compact semi-transparent backing box so text is readable without cluttering video
        box_w = int(scale * 300)
        box_h = y_start + len(lines) * line_h + 4
        overlay = frame.copy()
        cv2.rectangle(overlay, (4, 4), (box_w, box_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        for i, (text, col) in enumerate(lines):
            y = y_start + i * line_h
            cv2.putText(frame, text, (x_pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)
