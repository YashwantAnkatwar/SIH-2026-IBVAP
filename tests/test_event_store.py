"""
Unit tests for event_store.py — SQLite-backed event persistence and
analytics aggregation. Uses a temporary on-disk DB file per test (via
tempfile) so tests don't interfere with each other or the real app data.
"""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from event_store import EventStore


def _fresh_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let EventStore create it fresh
    return EventStore(path), path


def test_record_and_count():
    store, path = _fresh_store()
    store.record(camera_id="BOP-01", event_type="OBJECT_DETECTED", severity="LOW")
    store.record(camera_id="BOP-01", event_type="ZONE_ENTER", severity="MEDIUM")
    assert store.count() == 2
    store.close()


def test_query_filters_by_camera_and_severity():
    store, path = _fresh_store()
    store.record(camera_id="BOP-01", event_type="RISK", severity="HIGH")
    store.record(camera_id="BOP-02", event_type="RISK", severity="HIGH")
    store.record(camera_id="BOP-01", event_type="RISK", severity="LOW")

    by_cam = store.query(camera_id="BOP-01")
    assert len(by_cam) == 2
    assert all(r["camera_id"] == "BOP-01" for r in by_cam)

    by_sev = store.query(severity="HIGH")
    assert len(by_sev) == 2
    store.close()


def test_query_orders_newest_first():
    store, path = _fresh_store()
    store.record(event_type="A", timestamp=100.0)
    store.record(event_type="B", timestamp=200.0)
    rows = store.query()
    assert rows[0]["event_type"] == "B"
    assert rows[1]["event_type"] == "A"
    store.close()


def test_counts_by_severity_and_camera():
    store, path = _fresh_store()
    store.record(camera_id="BOP-01", event_type="RISK", severity="CRITICAL")
    store.record(camera_id="BOP-01", event_type="RISK", severity="CRITICAL")
    store.record(camera_id="BOP-02", event_type="RISK", severity="LOW")

    sev_counts = store.counts_by_severity()
    assert sev_counts["CRITICAL"] == 2
    assert sev_counts["LOW"] == 1

    cam_counts = store.counts_by_camera()
    assert cam_counts["BOP-01"] == 2
    assert cam_counts["BOP-02"] == 1
    store.close()


def test_counts_by_event_type():
    store, path = _fresh_store()
    store.record(event_type="ZONE_ENTER")
    store.record(event_type="ZONE_ENTER")
    store.record(event_type="LOITERING")
    counts = store.counts_by_event_type()
    assert counts["ZONE_ENTER"] == 2
    assert counts["LOITERING"] == 1
    store.close()


def test_events_over_time_buckets():
    store, path = _fresh_store()
    store.record(event_type="X", timestamp=0.0)
    store.record(event_type="X", timestamp=5.0)
    store.record(event_type="X", timestamp=65.0)
    buckets = store.events_over_time(bucket_seconds=60)
    assert len(buckets) == 2
    assert buckets[0]["count"] == 2
    assert buckets[1]["count"] == 1
    store.close()


def test_object_counts_distinct_tracks():
    store, path = _fresh_store()
    store.record(object_type="person", track_id="t1", event_type="OBJECT_DETECTED")
    store.record(object_type="person", track_id="t1", event_type="OBJECT_DETECTED")
    store.record(object_type="person", track_id="t2", event_type="OBJECT_DETECTED")
    store.record(object_type="car", track_id="t3", event_type="OBJECT_DETECTED")
    counts = store.object_counts()
    assert counts["person"] == 2  # distinct tracks t1, t2
    assert counts["car"] == 1
    store.close()


def test_data_span_seconds_reflects_real_range():
    store, path = _fresh_store()
    assert store.data_span_seconds() == 0.0
    store.record(event_type="X", timestamp=1000.0)
    store.record(event_type="X", timestamp=1300.0)
    assert store.data_span_seconds() == 300.0
    store.close()


def test_zone_stats_counts_entries_exits_violations():
    store, path = _fresh_store()
    store.record(zone="restricted_zone", event_type="ZONE_ENTER")
    store.record(zone="restricted_zone", event_type="ZONE_EXIT", dwell_seconds=4.0)
    store.record(zone="restricted_zone", event_type="LOITERING")
    stats = {row["zone"]: row for row in store.zone_stats()}
    assert stats["restricted_zone"]["entries"] == 1
    assert stats["restricted_zone"]["exits"] == 1
    assert stats["restricted_zone"]["violations"] == 1
    store.close()


def test_store_persists_across_reopen():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    store1 = EventStore(path)
    store1.record(event_type="PERSISTED")
    store1.close()

    store2 = EventStore(path)
    assert store2.count() == 1
    store2.close()
    os.remove(path)


def test_audit_logs_record_and_query():
    store, path = _fresh_store()
    store.record_audit(user="Capt. Yashwant", role="OFFICER", action="CALIBRATE_ZONE", target="BOP-01", details="Updated restricted zone")
    store.record_audit(user="Sepoy Rajesh", role="JAWAN", action="ALERT_ACKNOWLEDGE", target="ALT-001", details="Sentry acknowledged alert")
    logs = store.query_audit()
    assert len(logs) == 2
    assert logs[0]["user"] == "Sepoy Rajesh"
    assert logs[0]["action"] == "ALERT_ACKNOWLEDGE"
    assert logs[1]["user"] == "Capt. Yashwant"
    assert logs[1]["action"] == "CALIBRATE_ZONE"
    store.close()
    os.remove(path)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} event-store tests passed.")
