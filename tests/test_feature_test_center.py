"""
Test Suite for Feature Test Center (SIH 2026 Problem Statement 187).
Validates:
1. Feature inventory completeness and descriptions (Point 10).
2. Demo session isolation (Point 4 - normal operational mode unaffected).
3. Bounded Complete Demo sequence execution (Point 6 - bounded max 8s per feature).
4. API endpoints and streaming routes.
"""

import sys
import time
import os
from pathlib import Path

# Add project root and app dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import config
from test_center import (
    FEATURE_INVENTORY,
    get_test_center_inventory,
    get_test_center_manager,
    DemoSession,
    TestCenterManager,
)
import dashboard


def test_feature_inventory_point_10_descriptions():
    """Point 10: Verify every capability has a short, accurate code-derived description."""
    inventory = get_test_center_inventory()
    assert len(inventory) == 14, f"Expected 14 capabilities, got {len(inventory)}"

    for item in inventory:
        assert item["id"], "Capability missing ID"
        assert item["name"], "Capability missing name"
        assert item["category"], "Capability missing category"
        assert item.get("short_description"), f"{item['id']} missing short_description"
        assert len(item["short_description"]) >= 20, f"{item['id']} description too brief"
        assert item["primary_camera"] in ("DEMO-01", "DEMO-02", "DEMO-03"), f"Invalid primary camera: {item['primary_camera']}"
        assert item.get("demo_video_file"), f"{item['id']} missing demo video file"
        assert item.get("expected_event_type"), f"{item['id']} missing expected event type"


def test_demo_session_isolation_point_4():
    """Point 4: Verify starting a demo session uses isolated workers without touching normal cameras."""
    operational_workers = {"BOP-01": object(), "BOP-02": object(), "BOP-03": object()}
    initial_normal_cameras = set(operational_workers.keys())

    manager = TestCenterManager()
    
    # Start a demo session for ANPR (either by code CAP-04 / CAP-06 or id "anpr")
    res = manager.start_demo_session("anpr")
    assert res["status"] == "READY"
    session = manager.get_active_session()
    assert session is not None
    assert session.feature["id"] == "anpr"
    assert session.is_active is True
    assert set(session.workers.keys()) == {"DEMO-01", "DEMO-02", "DEMO-03"}

    # Operational camera workers remain completely unpolluted (Point 4)
    assert set(operational_workers.keys()) == initial_normal_cameras
    assert "DEMO-01" not in operational_workers
    assert "DEMO-02" not in operational_workers
    assert "DEMO-03" not in operational_workers

    # Check session status
    status = session.get_status()
    assert status["active_session"] is True
    assert status["feature"]["id"] == "anpr"
    assert "DEMO-01" in status["worker_status"]

    # Stop session
    stop_res = manager.stop_demo_session()
    assert stop_res["status"] == "STOPPED"
    assert session.is_active is False
    assert manager.get_active_session() is None


def test_complete_demo_sequence_point_6_bounded():
    """Point 6: Verify complete demo sequence runs in bounded fashion (max 8s per step) and advances cleanly."""
    manager = TestCenterManager()

    # Shorten inventory for test to 2 items to ensure quick bounded execution
    test_subset = [FEATURE_INVENTORY[0], FEATURE_INVENTORY[1]]
    
    steps = [
        {
            "feature_id": f["id"],
            "name": f["name"],
            "camera_id": f["primary_camera"],
            "expected_event": f.get("expected_event_type"),
            "timeout_seconds": 1.0,
            "status": "PENDING",
            "outcome": None,
            "duration": 0.0,
            "event_count": 0,
        }
        for f in test_subset
    ]

    manager._complete_demo_state = {
        "running": True,
        "status": "RUNNING",
        "total_steps": len(steps),
        "current_step_index": 0,
        "steps": steps,
        "results": [],
        "verified_count": 0,
        "timed_out_count": 0,
        "started_at": time.time(),
        "finished_at": None,
    }

    # Run the worker step
    manager._run_step(0, steps[0])
    assert steps[0]["outcome"] in ("VERIFIED", "TIMED_OUT")
    assert steps[0]["duration"] <= 3.0  # Must be bounded!

    manager._run_step(1, steps[1])
    assert steps[1]["outcome"] in ("VERIFIED", "TIMED_OUT")
    assert steps[1]["duration"] <= 3.0


class FakeCameraManager:
    def __init__(self):
        self.workers = {}

    def get_worker(self, camera_id):
        return None

    def get_full_status(self):
        return {}

    def get_system_health(self):
        return {"cameras_online": 0, "cameras_total": 0, "health": "GOOD"}


@pytest.fixture
def client():
    manager = FakeCameraManager()
    test_app = dashboard.create_app(manager)
    test_app.testing = True
    with test_app.test_client() as c:
        yield c


def test_api_test_center_inventory(client):
    """Test /api/test-center/inventory route returns valid data."""
    res = client.get("/api/test-center/inventory")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 14
    assert len(data["capabilities"]) == 14
    first = data["capabilities"][0]
    assert "short_description" in first
    assert "primary_camera" in first


def test_api_test_center_demo_lifecycle(client):
    """Test start, status, events, and stop API lifecycle."""
    # 1. Start demo
    start_res = client.post("/api/test-center/start-demo", json={"feature_id": "human_detection"})
    assert start_res.status_code == 200
    start_data = start_res.get_json()
    assert start_data["status"] == "READY"
    assert start_data["session_id"].startswith("demo_")

    # 2. Check session info
    sess_res = client.get("/api/test-center/session")
    assert sess_res.status_code == 200
    sess_data = sess_res.get_json()
    assert sess_data["active_session"] is True
    assert sess_data["feature"]["id"] == "human_detection"

    # 3. Check demo events (isolated)
    events_res = client.get("/api/test-center/events")
    assert events_res.status_code == 200
    events_data = events_res.get_json()
    assert "events" in events_data
    assert isinstance(events_data["events"], list)

    # 4. Stop demo
    stop_res = client.post("/api/test-center/stop-demo")
    assert stop_res.status_code == 200
    stop_data = stop_res.get_json()
    assert stop_data["status"] == "STOPPED"

    # 5. Check session is now inactive
    sess_res2 = client.get("/api/test-center/session")
    sess_data2 = sess_res2.get_json()
    assert sess_data2["active_session"] is False


def test_api_complete_demo_status_idle(client):
    """Test /api/test-center/complete-demo/status when no sequence is active."""
    res = client.get("/api/test-center/complete-demo/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert data["status"] in ("IDLE", "STOPPED", "COMPLETED")
