"""
Integration tests for dashboard.py's Flask API: status, system health,
alerts (list/ack/resolve), event history, analytics, and heatmap
endpoints. Uses a lightweight fake CameraManager/CameraWorker so this
runs without OpenCV video I/O or the YOLO model - it's testing the API
wiring, not the CV pipeline (which test_pipeline_smoke.py covers).
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config

# Point the shared EventStore at a throwaway temp DB for this test run,
# before importing shared/dashboard (which create the singleton on import).
_fd, _tmp_db = tempfile.mkstemp(suffix=".db")
os.close(_fd)
os.remove(_tmp_db)
config.EVENTS_DB_PATH = Path(_tmp_db)

import shared  # noqa: E402  (must import after overriding EVENTS_DB_PATH)
import dashboard  # noqa: E402


class FakeWorker:
    def __init__(self, camera_id, status="ONLINE"):
        self.camera_id = camera_id
        self.status = status
        self.fps = 12.3
        self.frame_count = 500
        self.object_counts = {"person": 2, "car": 1}
        self.active_zones = {"restricted_zone"}
        self.active_track_count = 3
        self.risk_level = "HIGH"
        self.active_alerts = ["person #4: Present in restricted zone 'restricted_zone'"]
        self.error_message = ""
        self.frame_failures = 0
        self.reconnect_attempts = 0
        self.processing_latency_ms = 42.0
        self.last_frame_time = 0.0
        self.last_evidence_path = None
        self.heatmap_cols = 4
        self.heatmap_rows = 2
        self.heatmap = [[1, 0, 0, 2], [0, 3, 0, 0]]
        self.paused = False
        self.confidence = 0.35

    def reload_zones(self, zones=None):
        pass

    def reload_lines(self, lines=None):
        pass

    def get_latest_jpeg(self):
        return None

    def get_heatmap_snapshot(self):
        return [row[:] for row in self.heatmap]

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def start(self):
        self.status = "ONLINE"

    def stop(self):
        self.status = "OFFLINE"
        self.paused = False

    def set_confidence(self, value):
        self.confidence = float(value)


class FakeCameraManager:
    def __init__(self):
        self.workers = {
            "BOP-01": FakeWorker("BOP-01", status="ONLINE"),
            "BOP-02": FakeWorker("BOP-02", status="OFFLINE"),
        }

    def get_worker(self, camera_id):
        return self.workers.get(camera_id)

    def get_full_status(self):
        return {cid: {"status": w.status, "fps": w.fps} for cid, w in self.workers.items()}

    def get_system_health(self):
        online = sum(1 for w in self.workers.values() if w.status == "ONLINE")
        total = len(self.workers)
        label = "GOOD" if online == total else ("DEGRADED" if online else "CRITICAL")
        return {"cameras_online": online, "cameras_total": total, "health": label}

    # ---- Operator controls (Phase 18) ----

    def pause_camera(self, camera_id):
        w = self.workers.get(camera_id)
        if w is None:
            return False
        w.pause()
        return True

    def resume_camera(self, camera_id):
        w = self.workers.get(camera_id)
        if w is None:
            return False
        w.resume()
        return True

    def stop_camera(self, camera_id):
        w = self.workers.get(camera_id)
        if w is None:
            return False
        w.stop()
        return True

    def start_camera(self, camera_id):
        w = self.workers.get(camera_id)
        if w is None:
            return False
        w.start()
        return True

    def set_confidence(self, value, camera_id=None):
        if camera_id and camera_id not in self.workers:
            return False
        targets = [self.workers[camera_id]] if camera_id else list(self.workers.values())
        for w in targets:
            w.set_confidence(value)
        return True

    def set_feature(self, feature, enabled, camera_id=None):
        if camera_id and camera_id not in self.workers:
            return False
        return True

    def reload_zones_and_lines(self, camera_id=None):
        if camera_id and camera_id not in self.workers:
            return False
        return True

    def simulate_tamper(self, camera_id, tamper_type=None):
        if camera_id and camera_id not in self.workers:
            return False
        return True


def _client():
    manager = FakeCameraManager()
    app = dashboard.create_app(manager)
    app.testing = True
    return app.test_client()


def test_index_page_loads():
    client = _client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"IBVAP" in resp.data


def test_status_endpoint():
    client = _client()
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["BOP-01"]["status"] == "ONLINE"
    assert data["BOP-02"]["status"] == "OFFLINE"


def test_system_health_reflects_degraded_when_one_camera_offline():
    client = _client()
    resp = client.get("/api/system")
    data = resp.get_json()
    assert data["cameras_online"] == 1
    assert data["cameras_total"] == 2
    assert data["health"] == "DEGRADED"


def test_unknown_camera_video_feed_returns_404():
    client = _client()
    resp = client.get("/video_feed/NOT-A-CAMERA")
    assert resp.status_code == 404


def test_valid_camera_video_feed_streams_mjpeg():
    mgr = FakeCameraManager()
    mgr.workers["BOP-01"].get_latest_jpeg = lambda: b"\xff\xd8\xff\xe0mock_jpeg"
    app = dashboard.create_app(mgr)
    client = app.test_client()
    resp = client.get("/video_feed/BOP-01")
    assert resp.status_code == 200
    assert "multipart/x-mixed-replace" in resp.headers.get("Content-Type", "")
    first_chunk = next(resp.response)
    assert b"--frame" in first_chunk
    assert b"Content-Type: image/jpeg" in first_chunk


def test_heatmap_endpoint_returns_real_grid():
    client = _client()
    resp = client.get("/api/heatmap/BOP-01")
    data = resp.get_json()
    assert data["cols"] == 4 and data["rows"] == 2
    assert data["grid"] == [[1, 0, 0, 2], [0, 3, 0, 0]]


def test_heatmap_unknown_camera_404():
    client = _client()
    resp = client.get("/api/heatmap/NOPE")
    assert resp.status_code == 404


def test_alert_lifecycle_via_api():
    client = _client()
    shared.alert_manager.raise_alert(
        camera_id="BOP-01", track_id="BOP-01:9", object_type="person",
        event_type="LOITERING", severity="CRITICAL",
        description="person loitering in restricted_zone",
    )
    resp = client.get("/api/alerts?status=ACTIVE")
    alerts = resp.get_json()
    assert len(alerts) >= 1
    alert_id = alerts[0]["id"]

    ack_resp = client.post(f"/api/alerts/{alert_id}/acknowledge")
    assert ack_resp.get_json()["status"] == "ACKNOWLEDGED"

    resolve_resp = client.post(f"/api/alerts/{alert_id}/resolve")
    assert resolve_resp.get_json()["status"] == "RESOLVED"

    still_active = client.get("/api/alerts?status=ACTIVE").get_json()
    assert alert_id not in [a["id"] for a in still_active]


def test_acknowledge_unknown_alert_returns_404():
    client = _client()
    resp = client.post("/api/alerts/ALERT-999999/acknowledge")
    assert resp.status_code == 404


def test_history_and_analytics_reflect_persisted_events():
    client = _client()
    shared.event_store.record(camera_id="BOP-01", event_type="ZONE_ENTER", severity="MEDIUM")
    shared.event_store.record(camera_id="BOP-01", event_type="LOITERING", severity="CRITICAL")

    history = client.get("/api/history?camera=BOP-01").get_json()
    assert len(history) >= 2

    analytics = client.get("/api/analytics").get_json()
    assert analytics["total_events"] >= 2
    assert "CRITICAL" in analytics["by_severity"]


def test_pause_and_resume_camera():
    client = _client()
    resp = client.post("/api/cameras/BOP-01/pause")
    assert resp.status_code == 200
    assert resp.get_json()["paused"] is True

    resp = client.post("/api/cameras/BOP-01/resume")
    assert resp.status_code == 200
    assert resp.get_json()["paused"] is False


def test_pause_unknown_camera_404():
    client = _client()
    resp = client.post("/api/cameras/NOPE/pause")
    assert resp.status_code == 404


def test_stop_and_start_camera():
    client = _client()
    resp = client.post("/api/cameras/BOP-01/stop")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "OFFLINE"

    resp = client.post("/api/cameras/BOP-01/start")
    assert resp.status_code == 200


def test_set_confidence_global_and_per_camera():
    client = _client()
    resp = client.post("/api/confidence", json={"confidence": 0.5})
    assert resp.status_code == 200
    assert resp.get_json()["confidence"] == 0.5

    resp = client.post("/api/confidence", json={"confidence": 0.6, "camera_id": "BOP-02"})
    assert resp.status_code == 200
    assert resp.get_json()["camera_id"] == "BOP-02"


def test_set_confidence_rejects_out_of_range():
    client = _client()
    resp = client.post("/api/confidence", json={"confidence": 1.5})
    assert resp.status_code == 400

    resp = client.post("/api/confidence", json={"confidence": "not-a-number"})
    assert resp.status_code == 400


def test_set_confidence_unknown_camera_404():
    client = _client()
    resp = client.post("/api/confidence", json={"confidence": 0.4, "camera_id": "NOPE"})
    assert resp.status_code == 404


def test_false_alarm_workflow():
    client = _client()
    alert_dict, _ = shared.alert_manager.raise_alert(
        camera_id="BOP-01", track_id="BOP-01:99", object_type="person",
        event_type="ZONE_RISK", severity="MEDIUM", description="Test alert for false alarm",
    )
    alert_id = alert_dict["id"]
    resp = client.post(f"/api/alerts/{alert_id}/false_alarm", json={"notes": "Legitimate maintenance staff"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "FALSE_ALARM"
    assert resp.get_json()["false_alarm_notes"] == "Legitimate maintenance staff"


def test_feature_toggle_endpoint():
    client = _client()
    resp = client.post("/api/features/toggle", json={"feature": "anpr", "enabled": True})
    assert resp.status_code == 200
    assert resp.get_json()["enabled"] is True

    resp = client.post("/api/features/toggle", json={"feature": "invalid_feature", "enabled": True})
    assert resp.status_code == 400


def test_sync_endpoints():
    client = _client()
    resp = client.get("/api/sync/status")
    assert resp.status_code == 200
    assert "connectivity" in resp.get_json()

    resp = client.post("/api/sync/toggle")
    assert resp.status_code == 200
    assert "is_connected" in resp.get_json()

    resp = client.post("/api/sync/now")
    assert resp.status_code == 200


def test_config_zones_and_lines():
    client = _client()
    resp = client.get("/api/config/zones")
    assert resp.status_code == 200
    assert "default" in resp.get_json()

    resp = client.get("/api/config/lines")
    assert resp.status_code == 200
    assert "default" in resp.get_json()


def test_incident_dossier():
    client = _client()
    alert, is_new = shared.alert_manager.raise_alert(
        camera_id="BOP-01",
        track_id="trk-99",
        object_type="person",
        event_type="UNAUTHORIZED_ZONE_ENTRY",
        severity="CRITICAL",
        description="Breach into restricted border sector",
    )
    resp = client.get(f"/api/incident/{alert['id']}/dossier")
    assert resp.status_code == 200
    dossier = resp.get_json()
    assert "classification" in dossier
    assert "sop_checklist" in dossier
    assert len(dossier["sop_checklist"]) > 0


def test_auth_and_audit_endpoints():
    client = _client()
    resp = client.get("/api/auth/user")
    assert resp.status_code == 200
    user = resp.get_json()
    assert "role" in user

    resp = client.post("/api/auth/role", json={"role": "JAWAN"})
    assert resp.status_code == 200
    assert resp.get_json()["role"] == "JAWAN"

    resp = client.get("/api/audit/logs")
    assert resp.status_code == 200
    logs = resp.get_json()
    assert any(log["action"] == "ROLE_SWITCH" for log in logs)

    # Switch back to OFFICER
    client.post("/api/auth/role", json={"role": "OFFICER"})


def test_simulate_tamper_endpoint():
    client = _client()
    resp = client.post("/api/cameras/BOP-01/tamper", json={"type": "OCCLUSION"})
    assert resp.status_code == 200
    assert resp.get_json()["simulated_tamper"] == "OCCLUSION"

    resp = client.post("/api/cameras/BOP-01/tamper", json={"type": None})
    assert resp.status_code == 200
    assert resp.get_json()["simulated_tamper"] is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} dashboard API tests passed.")
    shared.event_store.close()
    if os.path.exists(_tmp_db):
        os.remove(_tmp_db)
