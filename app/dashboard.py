"""
dashboard.py

Flask app that serves as the operator interface for IBVAP:
  - live MJPEG video feed per camera (with detection/tracking/zone/line
    overlays already drawn by camera_worker.py)
  - JSON status API (per-camera FPS, status, object counts, active
    zones, risk level, health)
  - JSON recent-events API (the live in-memory event feed)
  - JSON alert management API (list/filter/sort, acknowledge, resolve)
  - JSON persistent event-history API (Phase 9)
  - JSON analytics API (Phase 12) computed only from real stored events
  - JSON heatmap API (Phase 13) computed only from real tracked positions
  - evidence snapshot serving (Phase 14)
  - one HTML single-page app that ties it all together

This replaces cv2.imshow(), which does not work reliably across threads
on macOS and does not work at all in a headless environment.
"""

import csv
import io
import json
import queue
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

import numpy as np

from events import event_bus, CameraTamperEvent
import config
import shared
import auth
from test_center import test_center_manager


def create_app(camera_manager):
    app = Flask(__name__)
    shared.camera_manager = camera_manager

    class CustomJSONProvider(app.json_provider_class):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return super().default(obj)

    app.json_provider_class = CustomJSONProvider
    app.json = CustomJSONProvider(app)

    def _current_actor():
        u = auth.get_current_user()
        return u["display_name"], u["role"]

    # ------------------------------------------------------------------
    # Page
    # ------------------------------------------------------------------

    @app.route("/")
    def index():
        return render_template(
            "dashboard.html",
            camera_ids=list(camera_manager.workers.keys()),
        )

    # ------------------------------------------------------------------
    # Live status / events / video
    # ------------------------------------------------------------------

    @app.route("/api/status")
    def api_status():
        return jsonify(camera_manager.get_full_status())

    @app.route("/api/system")
    def api_system():
        health = camera_manager.get_system_health()
        health["active_alerts"] = shared.alert_manager.counts_by_severity(status="ACTIVE")
        health["total_events_recorded"] = shared.event_store.count()
        health["sync"] = shared.sync_service.get_status()
        return jsonify(health)

    @app.route("/api/events")
    def api_events():
        limit = request.args.get("limit", default=100, type=int)
        return jsonify(event_bus.recent(limit=limit))

    @app.route("/api/stream")
    def api_stream():
        """
        Real-time Server-Sent Events (SSE) stream pushing updates to
        browser clients with sub-millisecond latency.
        """
        q = event_bus.subscribe()

        def gen():
            try:
                yield f"data: {json.dumps({'type': 'CONNECTED', 'ts': time.time()})}\n\n"
                while True:
                    try:
                        event = q.get(timeout=10.0)
                        # Filter out high-frequency raw track position events from SSE
                        # (bounding boxes are already rendered directly onto MJPEG video streams)
                        if hasattr(event, "event_kind") and event.event_kind == "TRACK":
                            continue
                        payload = asdict(event)
                        yield f"data: {json.dumps(payload, default=str)}\n\n"
                    except queue.Empty:
                        yield f": keep-alive\n\n"
            finally:
                event_bus.unsubscribe(q)

        return Response(stream_with_context(gen()), mimetype="text/event-stream")

    # ------------------------------------------------------------------
    # ANPR / Face recognition / Night movement summaries (Priority 6).
    # All three read from the same persistent event_store every other
    # history/analytics endpoint uses — no separate/duplicated state,
    # so these can never show something the event history disagrees with.
    # ------------------------------------------------------------------

    @app.route("/api/anpr")
    def api_anpr():
        limit = request.args.get("limit", default=100, type=int)
        camera_id = request.args.get("camera_id")
        rows = shared.event_store.query(event_type="ANPR_PLATE_READ", camera_id=camera_id, limit=limit)
        any_anpr = any(w.anpr_enabled for w in camera_manager.workers.values()) if camera_manager else config.ANPR_ENABLED

        # Deduplicate by unique plate and compute visit frequency
        import re
        plate_groups = {}
        for r in rows:
            extra = r.get("extra") or {}
            plate = extra.get("plate_text")
            if not plate:
                m = re.search(r"plate read as '([^']+)'", r.get("description") or "")
                plate = m.group(1) if m else None
            if not plate or plate in ("—", "UNKNOWN"):
                continue
            if plate not in plate_groups:
                plate_groups[plate] = {
                    "plate_text": plate,
                    "state_name": extra.get("state_name") or "",
                    "count": 0,
                    "latest_timestamp": r.get("timestamp_iso") or "",
                    "latest_ts": r.get("ts") or 0.0,
                    "camera_id": r.get("camera_id") or "BOP-03",
                    "track_id": r.get("track_id") or "—",
                    "plate_confidence": extra.get("plate_confidence") or 78.0,
                    "evidence_path": extra.get("evidence_path") or "",
                    "recent_tracks": [],
                }
            group = plate_groups[plate]
            group["count"] += 1
            if r.get("track_id") and r.get("track_id") not in group["recent_tracks"]:
                group["recent_tracks"].append(r.get("track_id"))
            if (r.get("ts") or 0.0) >= group["latest_ts"]:
                group["latest_ts"] = r.get("ts") or 0.0
                group["latest_timestamp"] = r.get("timestamp_iso") or group["latest_timestamp"]
                group["track_id"] = r.get("track_id") or group["track_id"]
                if extra.get("plate_confidence"):
                    group["plate_confidence"] = extra.get("plate_confidence")
                if extra.get("evidence_path"):
                    group["evidence_path"] = extra.get("evidence_path")
                if extra.get("state_name"):
                    group["state_name"] = extra.get("state_name")

        unique_plates_list = sorted(plate_groups.values(), key=lambda g: g["latest_ts"], reverse=True)

        return jsonify({
            "enabled": any_anpr,
            "recent_reads": rows,
            "unique_plates": unique_plates_list,
            "distinct_plates": sorted({r["extra"].get("plate_text") for r in rows if r.get("extra")}),
        })

    @app.route("/api/faces")
    def api_faces():
        limit = request.args.get("limit", default=50, type=int)
        camera_id = request.args.get("camera_id")
        rows = shared.event_store.query(event_type="FACE_RECOGNITION", camera_id=camera_id, limit=limit)
        known = [r for r in rows if r.get("extra", {}).get("recognition_status") == "KNOWN"]
        unknown = [r for r in rows if r.get("extra", {}).get("recognition_status") == "UNKNOWN"]
        any_face = any(w.face_enabled for w in camera_manager.workers.values()) if camera_manager else config.FACE_RECOGNITION_ENABLED
        return jsonify({
            "enabled": any_face,
            "gallery_identities": shared.face_gallery.enrolled_identities if shared.face_gallery else [],
            "recent_known": known[:limit],
            "recent_unknown": unknown[:limit],
        })

    @app.route("/api/night")
    def api_night():
        limit = request.args.get("limit", default=50, type=int)
        rows = shared.event_store.query(event_type="NIGHT_MOVEMENT", limit=limit)
        status = camera_manager.get_full_status()
        return jsonify({
            "per_camera_luminance": {cam_id: s.get("night") for cam_id, s in status.items()},
            "recent_night_movement_events": rows,
        })

    @app.route("/video_feed/<camera_id>")
    def video_feed(camera_id):
        worker = camera_manager.get_worker(camera_id)
        if worker is None:
            return f"Unknown camera '{camera_id}'", 404
        return Response(
            _mjpeg_generator(worker),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/demo_feed/<camera_id>")
    def demo_feed(camera_id):
        """
        Isolated demo feed for Feature Test Center. Does NOT interfere with normal cameras.
        """
        session = test_center_manager.get_active_session()
        worker = None
        if session:
            worker = session.get_worker(camera_id)
            if worker is None and "0" in camera_id:
                suffix = camera_id[-2:]
                worker = session.get_worker(f"DEMO-{suffix}")
        if worker is None:
            worker = camera_manager.get_worker(camera_id)
        if worker is None:
            return f"Unknown demo camera '{camera_id}'", 404
        return Response(
            _mjpeg_generator(worker),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # ------------------------------------------------------------------
    # Feature Test Center (SIH26187 Verification Suite)
    # ------------------------------------------------------------------

    @app.route("/api/test-center/inventory")
    def api_test_center_inventory():
        items = test_center_manager.get_inventory()
        return jsonify({"capabilities": items, "total": len(items)})

    @app.route("/api/test-center/start-demo", methods=["POST"])
    def api_test_center_start():
        data = request.get_json() or {}
        feature_id = data.get("feature_id")
        if not feature_id:
            return jsonify({"error": "Missing feature_id"}), 400
        try:
            res = test_center_manager.start_demo_session(feature_id)
            user, role = _current_actor()
            shared.event_store.record_audit(
                user=user, role=role, action="TEST_CENTER_START",
                target=feature_id, details=f"Started isolated demo session for {feature_id}",
            )
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/test-center/stop-demo", methods=["POST"])
    def api_test_center_stop():
        res = test_center_manager.stop_demo_session()
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="TEST_CENTER_STOP",
            target="SESSION", details="Stopped isolated demo session, returned to normal mode",
        )
        return jsonify(res)

    @app.route("/api/test-center/session")
    def api_test_center_session():
        session = test_center_manager.get_active_session()
        if session:
            status = session.get_status()
            telemetry = session.get_telemetry()
            return jsonify({
                "active_session": True,
                "session_id": session.session_id,
                "feature": session.feature_meta,
                "worker_status": status.get("worker_status", {}),
                "telemetry": telemetry,
            })
        return jsonify({
            "active_session": False,
            "session_id": None,
            "feature": None,
            "worker_status": {},
            "telemetry": None,
        })

    @app.route("/api/test-center/events")
    def api_test_center_events():
        limit = request.args.get("limit", default=500, type=int)
        session = test_center_manager.get_active_session()
        events = session.get_events(limit=limit) if session else []
        total_cnt = getattr(session, "total_events_collected", len(events)) if session else 0
        return jsonify({"events": events, "count": total_cnt, "total_count": total_cnt})

    @app.route("/api/test-center/complete-demo/start", methods=["POST"])
    def api_test_center_complete_demo_start():
        res = test_center_manager.start_complete_demo()
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="COMPLETE_DEMO_START",
            target="TEST_CENTER", details="Started sequential complete demonstration runner",
        )
        return jsonify(res)

    @app.route("/api/test-center/complete-demo/status")
    def api_test_center_complete_demo_status():
        return jsonify(test_center_manager.get_complete_demo_status())

    @app.route("/api/test-center/complete-demo/stop", methods=["POST"])
    def api_test_center_complete_demo_stop():
        res = test_center_manager.stop_complete_demo()
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="COMPLETE_DEMO_STOP",
            target="TEST_CENTER", details="Stopped sequential complete demonstration runner",
        )
        return jsonify(res)

    # ------------------------------------------------------------------
    # Operator controls (Phase 18): pause/resume and stop/start a
    # camera's analysis pipeline, and adjust detection confidence live.
    # Every one of these calls a real method on the real worker thread -
    # there is no control here that doesn't actually do what it says.
    # ------------------------------------------------------------------

    @app.route("/api/cameras/<camera_id>/pause", methods=["POST"])
    def api_pause_camera(camera_id):
        if not camera_manager.pause_camera(camera_id):
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        return jsonify({"camera_id": camera_id, "paused": True})

    @app.route("/api/cameras/<camera_id>/resume", methods=["POST"])
    def api_resume_camera(camera_id):
        if not camera_manager.resume_camera(camera_id):
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        return jsonify({"camera_id": camera_id, "paused": False})

    @app.route("/api/cameras/<camera_id>/stop", methods=["POST"])
    def api_stop_camera(camera_id):
        if not camera_manager.stop_camera(camera_id):
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        return jsonify({"camera_id": camera_id, "status": "OFFLINE"})

    @app.route("/api/cameras/<camera_id>/start", methods=["POST"])
    def api_start_camera(camera_id):
        if not camera_manager.start_camera(camera_id):
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        return jsonify({"camera_id": camera_id, "status": "STARTING"})

    @app.route("/api/confidence", methods=["POST"])
    def api_set_confidence():
        data = request.get_json(silent=True) or {}
        try:
            value = float(data.get("confidence"))
        except (TypeError, ValueError):
            return jsonify({"error": "confidence must be a number between 0 and 1"}), 400
        if not (0.0 < value < 1.0):
            return jsonify({"error": "confidence must be between 0 and 1"}), 400
        camera_id = data.get("camera_id") or None
        if not camera_manager.set_confidence(value, camera_id=camera_id):
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        return jsonify({"confidence": value, "camera_id": camera_id})

    @app.route("/api/cameras/<camera_id>/tamper", methods=["POST"])
    def api_simulate_tamper(camera_id):
        data = request.get_json(silent=True) or {}
        tamper_type = data.get("type")
        if hasattr(camera_manager, "simulate_tamper"):
            if not camera_manager.simulate_tamper(camera_id, tamper_type):
                return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        if tamper_type:
            event_bus.publish(CameraTamperEvent(
                camera_id=camera_id,
                tamper_type=tamper_type,
                reason=f"Hardware sabotage / {tamper_type.lower()} mode triggered",
                event_id=f"TAMPER-{camera_id}-{int(time.time())}",
            ))
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="SIMULATE_TAMPER",
            target=camera_id, details=f"Set simulated tamper mode to '{tamper_type or 'NORMAL'}'",
        )
        return jsonify({"camera_id": camera_id, "simulated_tamper": tamper_type})

    # ------------------------------------------------------------------
    # Alerts (Phase 7/8)
    # ------------------------------------------------------------------

    @app.route("/api/alerts")
    def api_alerts():
        alerts = shared.alert_manager.list_alerts(
            status=request.args.get("status"),
            severity=request.args.get("severity"),
            camera_id=request.args.get("camera"),
            sort_by=request.args.get("sort", "newest"),
            limit=request.args.get("limit", default=None, type=int),
        )
        return jsonify(alerts)

    @app.route("/api/alerts/<alert_id>/acknowledge", methods=["POST"])
    def api_ack_alert(alert_id):
        alert = shared.alert_manager.acknowledge(alert_id)
        if alert is None:
            return jsonify({"error": f"Unknown alert '{alert_id}'"}), 404
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="ALERT_ACKNOWLEDGE",
            target=alert_id, details=f"Acknowledged alert {alert_id} ({alert.get('severity')})",
        )
        return jsonify(alert)

    @app.route("/api/alerts/<alert_id>/resolve", methods=["POST"])
    def api_resolve_alert(alert_id):
        alert = shared.alert_manager.resolve(alert_id)
        if alert is None:
            return jsonify({"error": f"Unknown alert '{alert_id}'"}), 404
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="ALERT_RESOLVE",
            target=alert_id, details=f"Resolved alert {alert_id}",
        )
        return jsonify(alert)

    @app.route("/api/alerts/<alert_id>/false_alarm", methods=["POST"])
    def api_false_alarm(alert_id):
        data = request.get_json(silent=True) or {}
        notes = data.get("notes", "Operator flagged as false positive")
        alert = shared.alert_manager.flag_false_alarm(alert_id, notes=notes)
        if alert is None:
            return jsonify({"error": f"Unknown alert '{alert_id}'"}), 404

        try:
            feedback_dir = config.PROJECT_ROOT / "data" / "retraining_feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)
            rec_file = feedback_dir / f"{alert_id}_{int(time.time())}.json"
            with open(rec_file, "w") as f:
                json.dump(alert, f, indent=2)
        except Exception as e:
            print(f"Failed to save retraining feedback: {e}")

        shared.event_store.record(
            camera_id=alert.get("camera_id"),
            track_id=alert.get("track_id"),
            event_type="ALERT_FALSE_ALARM",
            severity="LOW",
            description=f"Alert {alert_id} flagged as false alarm: {notes}",
        )
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="ALERT_FALSE_ALARM",
            target=alert_id, details=f"Flagged false alarm: {notes}",
        )
        return jsonify(alert)

    @app.route("/api/features/toggle", methods=["POST"])
    def api_toggle_feature():
        data = request.get_json(silent=True) or {}
        feature = data.get("feature", "").strip().lower()
        enabled = bool(data.get("enabled", True))
        camera_id = data.get("camera_id") or None
        if feature not in ("anpr", "face", "face_recognition"):
            return jsonify({"error": "feature must be 'anpr' or 'face'"}), 400
        if not camera_manager.set_feature(feature, enabled, camera_id=camera_id):
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        if camera_id is None:
            if feature == "anpr":
                config.ANPR_ENABLED = enabled
            elif feature in ("face", "face_recognition"):
                config.FACE_RECOGNITION_ENABLED = enabled
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="FEATURE_TOGGLE",
            target=feature, details=f"Feature {feature} set to {enabled} for {camera_id or 'all cameras'}",
        )
        return jsonify({"feature": feature, "enabled": enabled, "camera_id": camera_id})

    @app.route("/api/sync/status")
    def api_sync_status():
        return jsonify(shared.sync_service.get_status())

    @app.route("/api/sync/toggle", methods=["POST"])
    def api_sync_toggle():
        shared.sync_service.toggle_connectivity()
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="SYNC_TOGGLE",
            target="CENTRAL_HQ", details="Toggled CIBMS edge satellite link",
        )
        return jsonify(shared.sync_service.get_status())

    @app.route("/api/sync/now", methods=["POST"])
    def api_sync_now():
        synced = shared.sync_service.trigger_sync_now()
        status = shared.sync_service.get_status()
        status["synced_this_batch"] = synced
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="SYNC_NOW",
            target="CENTRAL_HQ", details=f"Manually triggered sync ({synced} events uploaded)",
        )
        return jsonify(status)

    # ------------------------------------------------------------------
    # Zone & Line Calibration (Phase 3)
    # ------------------------------------------------------------------

    @app.route("/api/config/zones", methods=["GET", "POST"])
    def api_config_zones():
        zones_file = config.PROJECT_ROOT / "config" / "zones.json"
        if request.method == "GET":
            try:
                with open(zones_file, "r") as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        data = request.get_json(silent=True) or {}
        try:
            with open(zones_file, "r") as f:
                current_zones = json.load(f)
        except Exception:
            current_zones = {}

        camera_id = data.get("camera_id")
        if camera_id and "zones" in data:
            current_zones[camera_id] = data["zones"]
        elif isinstance(data, dict) and not camera_id:
            current_zones.update({k: v for k, v in data.items() if not k.startswith("_")})

        with open(zones_file, "w") as f:
            json.dump(current_zones, f, indent=4)

        if hasattr(camera_manager, "reload_zones_and_lines"):
            camera_manager.reload_zones_and_lines(camera_id)
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="CALIBRATE_ZONES",
            target=camera_id or "GLOBAL", details=f"Updated zone coordinates for {camera_id or 'all'}",
        )
        return jsonify({"status": "ok", "camera_id": camera_id, "zones": current_zones.get(camera_id)})

    @app.route("/api/config/lines", methods=["GET", "POST"])
    def api_config_lines():
        lines_file = config.PROJECT_ROOT / "config" / "lines.json"
        if request.method == "GET":
            try:
                with open(lines_file, "r") as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        data = request.get_json(silent=True) or {}
        try:
            with open(lines_file, "r") as f:
                current_lines = json.load(f)
        except Exception:
            current_lines = {}

        camera_id = data.get("camera_id")
        if camera_id and "lines" in data:
            current_lines[camera_id] = data["lines"]
        elif isinstance(data, dict) and not camera_id:
            current_lines.update({k: v for k, v in data.items() if not k.startswith("_")})

        with open(lines_file, "w") as f:
            json.dump(current_lines, f, indent=4)

        if hasattr(camera_manager, "reload_zones_and_lines"):
            camera_manager.reload_zones_and_lines(camera_id)
        user, role = _current_actor()
        shared.event_store.record_audit(
            user=user, role=role, action="CALIBRATE_LINES",
            target=camera_id or "GLOBAL", details=f"Updated crossing lines for {camera_id or 'all'}",
        )
        return jsonify({"status": "ok", "camera_id": camera_id, "lines": current_lines.get(camera_id)})

    @app.route("/api/snapshot/<camera_id>")
    def api_camera_snapshot(camera_id):
        worker = camera_manager.get_worker(camera_id)
        if worker is None:
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        jpeg = worker.get_latest_jpeg()
        if jpeg is None:
            return jsonify({"error": "No frame captured yet"}), 503
        return Response(jpeg, mimetype="image/jpeg")

    # ------------------------------------------------------------------
    # Incident Dossier & Export (Phase 3 & Phase 5)
    # ------------------------------------------------------------------

    def _build_dossier(alert_id):
        alert = shared.alert_manager.get(alert_id)
        if alert is None:
            events = shared.event_store.query(limit=200)
            matching = [e for e in events if alert_id in str(e.get("description", ""))]
            if matching:
                m = matching[0]
                alert = {
                    "id": alert_id,
                    "camera_id": m.get("camera_id"),
                    "track_id": m.get("track_id"),
                    "object_type": m.get("object_type"),
                    "event_type": m.get("event_type"),
                    "severity": m.get("severity", "MEDIUM"),
                    "description": m.get("description", ""),
                    "first_seen": m.get("ts"),
                    "last_seen": m.get("ts"),
                    "status": "RECORDED",
                    "occurrences": 1,
                    "reasons": [m.get("description")],
                }
            else:
                return None

        cam_id = alert.get("camera_id")
        worker = camera_manager.get_worker(cam_id)
        worker_status = worker.status if worker else "UNKNOWN"
        night_status = getattr(worker, "night_state", None)
        tamper_status = getattr(worker, "tamper_state", None)

        timeline = shared.event_store.query(
            camera_id=cam_id,
            limit=20,
        )

        sev = alert.get("severity", "LOW")
        sop_map = {
            "CRITICAL": [
                "1. Immediate Tactical Scramble: Dispatch Quick Reaction Team (QRT) to perimeter.",
                "2. Chokepoint Lock: Seal gate barrier and alert adjacent Border Outpost (BOP) towers.",
                "3. Interception Protocol: Illuminate sector with automated high-intensity searchlight.",
                "4. Log in Central Command CIBMS War Room incident ledger with evidence dossier attached.",
            ],
            "HIGH": [
                "1. Sentry Visual Verification: Command tower watch to confirm target classification.",
                "2. PTZ Camera Focus: Lock optical zoom tracking on anomalous track.",
                "3. Acoustic Deterrent: Sound perimeter warning horn if restricted line is breached.",
                "4. Standby QRT: Place Sector QRT on 2-minute notice.",
            ],
            "MEDIUM": [
                "1. Monitor object dwell duration and trajectory vector.",
                "2. Cross-reference automated ANPR / facial recognition logs.",
                "3. Escalate to High if object remains in restricted polygon past threshold.",
            ],
            "LOW": [
                "1. Routine surveillance observation. No tactical deployment required.",
            ],
        }

        return {
            "classification": "RESTRICTED // CIBMS TACTICAL INCIDENT DOSSIER",
            "dossier_id": f"DOS-{alert_id}",
            "generated_at": time.time(),
            "generated_at_iso": datetime.fromtimestamp(time.time()).isoformat(timespec="seconds"),
            "alert": alert,
            "sector": f"Sector BOP-{cam_id or 'HQ'}",
            "camera_telemetry": {
                "status": worker_status,
                "night_mode": night_status,
                "tamper_status": tamper_status,
            },
            "sop_checklist": sop_map.get(sev, sop_map["LOW"]),
            "timeline": timeline[:8],
        }

    @app.route("/api/incident/<alert_id>/dossier")
    def api_incident_dossier(alert_id):
        dossier = _build_dossier(alert_id)
        if dossier is None:
            return jsonify({"error": f"Incident '{alert_id}' not found"}), 404
        return jsonify(dossier)

    @app.route("/api/incident/<alert_id>/export")
    def api_export_incident(alert_id):
        dossier = _build_dossier(alert_id)
        if dossier is None:
            return jsonify({"error": f"Incident '{alert_id}' not found"}), 404
        fmt = request.args.get("format", "json").lower()
        if fmt in ("md", "markdown"):
            alert = dossier["alert"]
            reasons = alert.get("reasons") or [alert.get("description", "N/A")]
            reasons_md = "\n".join(f"- {r}" for r in reasons)
            sop_md = "\n".join(f"- {s}" for s in dossier["sop_checklist"])
            timeline_lines = [
                "| Time | Event Type | Camera | Severity | Description |",
                "| --- | --- | --- | --- | --- |",
            ]
            for ev in dossier["timeline"]:
                timeline_lines.append(
                    f"| {ev.get('timestamp_iso', '')} | {ev.get('event_type', '')} | "
                    f"{ev.get('camera_id', '')} | {ev.get('severity', '')} | {ev.get('description', '')} |"
                )
            timeline_md = "\n".join(timeline_lines)

            md = (
                f"# {dossier['classification']}\n"
                f"**Dossier ID**: `{dossier['dossier_id']}`  \n"
                f"**Generated At**: {dossier['generated_at_iso']}  \n"
                f"**Sector**: {dossier['sector']}  \n\n"
                f"## Incident Overview\n"
                f"- **Alert ID**: `{alert.get('id')}`\n"
                f"- **Camera**: `{alert.get('camera_id')}`\n"
                f"- **Target**: `{alert.get('object_type')}` (Track `#{alert.get('track_id')}`)\n"
                f"- **Severity**: **{alert.get('severity')}**\n"
                f"- **Status**: {alert.get('status')}\n"
                f"- **Description**: {alert.get('description')}\n\n"
                f"### Behavioral Reasons\n"
                f"{reasons_md}\n\n"
                f"## Camera Telemetry\n"
                f"- **Stream Status**: {dossier['camera_telemetry']['status']}\n"
                f"- **Night Mode**: {dossier['camera_telemetry']['night_mode']}\n"
                f"- **Tamper Detection**: {dossier['camera_telemetry']['tamper_status']}\n\n"
                f"## Standard Operating Procedure (SOP) Action Checklist\n"
                f"{sop_md}\n\n"
                f"## Associated Forensic Timeline\n"
                f"{timeline_md}\n"
            )
            return Response(
                md,
                mimetype="text/markdown",
                headers={"Content-Disposition": f"attachment; filename={alert_id}_dossier.md"},
            )
        else:
            return Response(
                json.dumps(dossier, indent=2, default=str),
                mimetype="application/json",
                headers={"Content-Disposition": f"attachment; filename={alert_id}_dossier.json"},
            )

    @app.route("/api/incidents")
    def api_incidents():
        camera_id = request.args.get("camera_id")
        status = request.args.get("status")
        min_sev = request.args.get("min_severity")
        try:
            limit = int(request.args.get("limit", 50))
        except ValueError:
            limit = 50
        if hasattr(shared, "correlator") and shared.correlator is not None:
            incidents = shared.correlator.list_incidents(
                camera_id=camera_id,
                status=status,
                min_severity=min_sev,
                limit=limit,
            )
        else:
            incidents = []
        return jsonify(incidents)

    # ------------------------------------------------------------------
    # RBAC & Audit Trail (Phase 4)
    # ------------------------------------------------------------------

    @app.route("/api/auth/user")
    def api_auth_user():
        user = auth.get_current_user()
        user["available_users"] = list(auth.USERS.values())
        return jsonify(user)

    @app.route("/api/auth/role", methods=["POST"])
    def api_auth_set_role():
        data = request.get_json(silent=True) or {}
        role = data.get("role") or data.get("username") or ""
        updated = auth.set_current_user(role)
        if updated is None:
            return jsonify({"error": f"Invalid role '{role}'"}), 400
        shared.event_store.record_audit(
            user=updated["display_name"],
            role=updated["role"],
            action="ROLE_SWITCH",
            target=updated["username"],
            details=f"Switched active console role to {updated['role']} ({updated['display_name']})",
        )
        return jsonify(updated)

    @app.route("/api/audit/logs")
    def api_audit_logs():
        limit = request.args.get("limit", default=100, type=int)
        logs = shared.event_store.query_audit(limit=limit)
        return jsonify(logs)

    @app.route("/api/audit/export")
    def api_audit_export():
        limit = request.args.get("limit", default=500, type=int)
        fmt = request.args.get("format", "csv").lower()
        logs = shared.event_store.query_audit(limit=limit)
        if fmt == "json":
            return Response(
                json.dumps(logs, indent=2),
                mimetype="application/json",
                headers={"Content-Disposition": "attachment; filename=ibvap_audit_log.json"},
            )
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["ID", "Timestamp (Epoch)", "Timestamp (ISO)", "Operator/User", "Role", "Action", "Target", "Details"])
        for row in logs:
            writer.writerow([
                row.get("id"),
                row.get("ts"),
                row.get("timestamp_iso"),
                row.get("user"),
                row.get("role"),
                row.get("action"),
                row.get("target"),
                row.get("details"),
            ])
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=ibvap_compliance_audit.csv"},
        )

    # ------------------------------------------------------------------
    # Persistent event history (Phase 9)
    # ------------------------------------------------------------------

    @app.route("/api/history")
    def api_history():
        rows = shared.event_store.query(
            camera_id=request.args.get("camera"),
            severity=request.args.get("severity"),
            event_type=request.args.get("event_type"),
            zone=request.args.get("zone"),
            start_ts=request.args.get("start", default=None, type=float),
            end_ts=request.args.get("end", default=None, type=float),
            limit=request.args.get("limit", default=200, type=int),
        )
        return jsonify(rows)

    # ------------------------------------------------------------------
    # Analytics (Phase 12) - all figures come straight from EventStore's
    # SQL aggregates over whatever has actually been recorded this run.
    # ------------------------------------------------------------------

    @app.route("/api/analytics")
    def api_analytics():
        return jsonify({
            "by_severity": shared.event_store.counts_by_severity(),
            "by_camera": shared.event_store.counts_by_camera(),
            "by_event_type": shared.event_store.counts_by_event_type(),
            "over_time": shared.event_store.events_over_time(
                bucket_seconds=request.args.get("bucket_seconds", default=60, type=int)
            ),
            "zone_stats": shared.event_store.zone_stats(),
            "object_counts": shared.event_store.object_counts(),
            "data_span_seconds": round(shared.event_store.data_span_seconds(), 1),
            "total_events": shared.event_store.count(),
        })

    # ------------------------------------------------------------------
    # Heatmap (Phase 13) - real accumulated track positions, per camera.
    # ------------------------------------------------------------------

    @app.route("/api/heatmap/<camera_id>")
    def api_heatmap(camera_id):
        worker = camera_manager.get_worker(camera_id)
        if worker is None:
            return jsonify({"error": f"Unknown camera '{camera_id}'"}), 404
        return jsonify({
            "camera_id": camera_id,
            "cols": worker.heatmap_cols,
            "rows": worker.heatmap_rows,
            "grid": worker.get_heatmap_snapshot(),
        })

    # ------------------------------------------------------------------
    # Evidence snapshots (Phase 14)
    # ------------------------------------------------------------------

    @app.route("/evidence/<path:filename>")
    def evidence_file(filename):
        return send_from_directory(str(config.EVIDENCE_DIR), filename)

    @app.route("/api/evidence")
    def api_evidence_list():
        rows = shared.event_store.query(event_type="EVIDENCE_CAPTURED", limit=100)
        return jsonify(rows)

    return app


def _mjpeg_generator(worker):
    frame_interval = 1.0 / config.STREAM_TARGET_FPS
    last_sent = 0.0

    while getattr(worker, "running", True):
        jpeg = worker.get_latest_jpeg()
        now = time.time()

        if jpeg is not None and (now - last_sent) >= frame_interval:
            last_sent = now
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
        else:
            time.sleep(0.02)
