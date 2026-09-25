"""
test_sync_service.py

Unit tests for app/sync_service.py:
- Offline queuing
- Sync advances when connected
- Toggle connectivity works
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from event_store import EventStore
from sync_service import SyncService


def test_sync_service_lifecycle():
    store = EventStore(":memory:")
    # Seed 10 events
    for i in range(10):
        store.record(camera_id="TEST-01", event_type="OBJECT_DETECTED", description=f"Event {i}")

    sync = SyncService(store)
    status = sync.get_status()
    assert status["connectivity"] == "ONLINE"
    assert status["total_events"] == 10

    # Add 5 more events
    for i in range(5):
        store.record(camera_id="TEST-01", event_type="OBJECT_DETECTED", description=f"New event {i}")

    status_before = sync.get_status()
    assert status_before["pending_events"] == 5

    # Trigger sync step
    synced = sync.trigger_sync_now()
    assert synced == 5

    status_after = sync.get_status()
    assert status_after["pending_events"] == 0

    # Toggle offline
    sync.set_connectivity(False)
    assert sync.get_status()["connectivity"] == "OFFLINE"

    # Add 3 events while offline
    for i in range(3):
        store.record(camera_id="TEST-01", event_type="ZONE_ENTER", description=f"Offline event {i}")

    # Should not sync while offline
    assert sync.trigger_sync_now() == 0
    assert sync.get_status()["pending_events"] == 3

    # Toggle back online
    sync.set_connectivity(True)
    assert sync.trigger_sync_now() == 3
    assert sync.get_status()["pending_events"] == 0


if __name__ == "__main__":
    test_sync_service_lifecycle()
    print("PASS: test_sync_service_lifecycle")
    print("\n1 sync service test passed.")
