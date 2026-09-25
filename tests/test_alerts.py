"""
Unit tests for alerts.py — alert creation, deduplication, severity
escalation, acknowledgement, and resolution. No YOLO/model dependency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from alerts import AlertManager


def _raise(mgr, t=0.0, severity="MEDIUM", camera="BOP-01", track="BOP-01:1", event="ZONE_ENTER"):
    return mgr.raise_alert(
        camera_id=camera, track_id=track, object_type="person",
        event_type=event, severity=severity,
        description="person entered restricted zone",
        zone="restricted_zone", timestamp=t,
    )


def test_creates_new_alert_on_first_occurrence():
    mgr = AlertManager()
    alert, is_new = _raise(mgr, t=0.0)
    assert is_new is True
    assert alert["occurrences"] == 1
    assert alert["status"] == "ACTIVE"


def test_deduplicates_repeated_alerts_within_window():
    mgr = AlertManager(dedup_window_seconds=10.0)
    first, _ = _raise(mgr, t=0.0)
    second, is_new = _raise(mgr, t=2.0)
    assert is_new is False
    assert second["id"] == first["id"]
    assert second["occurrences"] == 2


def test_does_not_deduplicate_after_window_expires():
    mgr = AlertManager(dedup_window_seconds=5.0)
    first, _ = _raise(mgr, t=0.0)
    second, is_new = _raise(mgr, t=20.0)
    assert is_new is True
    assert second["id"] != first["id"]


def test_dedup_key_is_specific_to_camera_track_and_event_type():
    mgr = AlertManager()
    a1, _ = _raise(mgr, t=0.0, camera="BOP-01", track="BOP-01:1", event="ZONE_ENTER")
    a2, is_new_track = _raise(mgr, t=0.0, camera="BOP-01", track="BOP-01:2", event="ZONE_ENTER")
    a3, is_new_event = _raise(mgr, t=0.0, camera="BOP-01", track="BOP-01:1", event="LOITERING")
    assert is_new_track is True and a2["id"] != a1["id"]
    assert is_new_event is True and a3["id"] != a1["id"]


def test_severity_escalates_to_higher_on_repeat():
    mgr = AlertManager()
    _raise(mgr, t=0.0, severity="MEDIUM")
    updated, _ = _raise(mgr, t=1.0, severity="CRITICAL")
    assert updated["severity"] == "CRITICAL"
    # Escalation should not downgrade back on a lower-severity repeat.
    updated2, _ = _raise(mgr, t=2.0, severity="LOW")
    assert updated2["severity"] == "CRITICAL"


def test_resolved_alert_is_not_deduplicated_into():
    mgr = AlertManager()
    alert, _ = _raise(mgr, t=0.0)
    mgr.resolve(alert["id"])
    new_alert, is_new = _raise(mgr, t=1.0)
    assert is_new is True
    assert new_alert["id"] != alert["id"]


def test_acknowledge_and_resolve_update_status():
    mgr = AlertManager()
    alert, _ = _raise(mgr, t=0.0)
    acked = mgr.acknowledge(alert["id"])
    assert acked["status"] == "ACKNOWLEDGED"
    assert acked["acknowledged_at"] is not None

    resolved = mgr.resolve(alert["id"])
    assert resolved["status"] == "RESOLVED"
    assert resolved["resolved_at"] is not None


def test_acknowledge_unknown_alert_returns_none():
    mgr = AlertManager()
    assert mgr.acknowledge("ALERT-999999") is None
    assert mgr.resolve("ALERT-999999") is None


def test_list_alerts_filters_by_status_and_severity():
    mgr = AlertManager()
    a1, _ = _raise(mgr, t=0.0, severity="HIGH", track="BOP-01:1")
    a2, _ = _raise(mgr, t=0.0, severity="CRITICAL", track="BOP-01:2")
    mgr.resolve(a1["id"])

    active = mgr.list_alerts(status="ACTIVE")
    assert all(a["status"] == "ACTIVE" for a in active)
    assert a2["id"] in [a["id"] for a in active]
    assert a1["id"] not in [a["id"] for a in active]

    critical = mgr.list_alerts(severity="CRITICAL")
    assert all(a["severity"] == "CRITICAL" for a in critical)


def test_list_alerts_sort_by_severity_puts_critical_first():
    mgr = AlertManager()
    _raise(mgr, t=0.0, severity="LOW", track="BOP-01:1")
    _raise(mgr, t=0.0, severity="CRITICAL", track="BOP-01:2")
    _raise(mgr, t=0.0, severity="MEDIUM", track="BOP-01:3")

    ordered = mgr.list_alerts(sort_by="severity")
    assert ordered[0]["severity"] == "CRITICAL"


def test_counts_by_severity():
    mgr = AlertManager()
    _raise(mgr, t=0.0, severity="HIGH", track="BOP-01:1")
    _raise(mgr, t=0.0, severity="HIGH", track="BOP-01:2")
    _raise(mgr, t=0.0, severity="CRITICAL", track="BOP-01:3")

    counts = mgr.counts_by_severity(status="ACTIVE")
    assert counts["HIGH"] == 2
    assert counts["CRITICAL"] == 1
    assert counts["LOW"] == 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} alert-management tests passed.")
