"""
alerts.py

Turns raw per-frame risk events into a manageable list of operator-facing
alerts (Phase 7/8): duplicate detections of the same underlying situation
(same camera + same track + same event type, seen again a moment later)
update one alert's timestamp/count instead of spawning a new alert every
frame, and every alert carries a severity plus operator actions
(acknowledge / resolve).

This module has no dependency on OpenCV/YOLO/Flask, so it is fully unit
testable on its own.
"""

import itertools
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

_id_counter = itertools.count(1)


def _next_id():
    return f"ALERT-{next(_id_counter):06d}"


@dataclass
class Alert:
    id: str
    camera_id: str
    track_id: str
    object_type: str
    event_type: str
    severity: str
    description: str
    zone: Optional[str] = None
    risk_score: Optional[int] = None
    reasons: List[str] = field(default_factory=list)
    evidence_path: Optional[str] = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    occurrences: int = 1
    status: str = "ACTIVE"  # ACTIVE | ACKNOWLEDGED | RESOLVED
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class AlertManager:
    """
    Thread-safe alert store with deduplication.

    Camera worker threads call `raise_alert(...)` for every risky
    track/event on every frame; the manager collapses repeats of the
    same (camera_id, track_id, event_type) into one alert as long as
    they keep recurring within `dedup_window_seconds` of each other.
    """

    def __init__(self, dedup_window_seconds=8.0, history_limit=5000):
        self.dedup_window_seconds = dedup_window_seconds
        self.history_limit = history_limit
        self._lock = threading.Lock()
        self._alerts = {}       # alert_id -> Alert
        self._dedup_index = {}  # (camera_id, track_id, event_type) -> alert_id

    def raise_alert(self, camera_id, track_id, object_type, event_type,
                     severity, description, zone=None, risk_score=None,
                     reasons=None, evidence_path=None, timestamp=None):
        """Create a new alert, or update a recent matching one (dedup)."""
        timestamp = timestamp if timestamp is not None else time.time()
        key = (camera_id, track_id, event_type)

        with self._lock:
            existing_id = self._dedup_index.get(key)
            existing = self._alerts.get(existing_id) if existing_id else None

            if (
                existing is not None
                and existing.status == "ACTIVE"
                and (timestamp - existing.last_seen) <= self.dedup_window_seconds
            ):
                existing.last_seen = timestamp
                existing.occurrences += 1
                existing.severity = _max_severity(existing.severity, severity)
                existing.description = description
                existing.risk_score = risk_score if risk_score is not None else existing.risk_score
                if reasons:
                    existing.reasons = reasons
                if evidence_path:
                    existing.evidence_path = evidence_path
                self._enforce_limit()
                return existing.to_dict(), False

            alert = Alert(
                id=_next_id(),
                camera_id=camera_id,
                track_id=track_id,
                object_type=object_type,
                event_type=event_type,
                severity=severity,
                description=description,
                zone=zone,
                risk_score=risk_score,
                reasons=list(reasons or []),
                evidence_path=evidence_path,
                first_seen=timestamp,
                last_seen=timestamp,
            )
            self._alerts[alert.id] = alert
            self._dedup_index[key] = alert.id
            self._enforce_limit()
            return alert.to_dict(), True

    def _enforce_limit(self):
        if len(self._alerts) <= self.history_limit:
            return
        # Drop the oldest resolved alerts first, then oldest overall.
        ordered = sorted(
            self._alerts.values(),
            key=lambda a: (a.status != "RESOLVED", a.last_seen),
        )
        for alert in ordered[: len(self._alerts) - self.history_limit]:
            self._alerts.pop(alert.id, None)

    # ------------------------------------------------------------------
    # Operator actions
    # ------------------------------------------------------------------

    def acknowledge(self, alert_id):
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            alert.status = "ACKNOWLEDGED"
            alert.acknowledged_at = time.time()
            return alert.to_dict()

    def resolve(self, alert_id):
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            alert.status = "RESOLVED"
            alert.resolved_at = time.time()
            return alert.to_dict()

    def flag_false_alarm(self, alert_id, notes=""):
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            alert.status = "FALSE_ALARM"
            alert.resolved_at = time.time()
            data = alert.to_dict()
            data["false_alarm_notes"] = notes
            return data

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_alerts(self, status=None, severity=None, camera_id=None,
                     sort_by="newest", limit=None):
        """
        Returns alerts as plain dicts, optionally filtered by status,
        severity, or camera, and sorted by 'newest' (last_seen desc,
        default) or 'severity' (severity desc, then newest).
        """
        with self._lock:
            items = list(self._alerts.values())

        if status:
            items = [a for a in items if a.status == status]
        if severity:
            items = [a for a in items if a.severity == severity]
        if camera_id:
            items = [a for a in items if a.camera_id == camera_id]

        if sort_by == "severity":
            items.sort(key=lambda a: (SEVERITY_ORDER.get(a.severity, 0), a.last_seen), reverse=True)
        else:
            items.sort(key=lambda a: a.last_seen, reverse=True)

        if limit:
            items = items[:limit]

        return [a.to_dict() for a in items]

    def get(self, alert_id):
        with self._lock:
            alert = self._alerts.get(alert_id)
            return alert.to_dict() if alert else None

    def counts_by_severity(self, status="ACTIVE"):
        with self._lock:
            items = list(self._alerts.values())
        counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        for a in items:
            if status is None or a.status == status:
                counts[a.severity] = counts.get(a.severity, 0) + 1
        return counts


def _max_severity(a, b):
    return a if SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0) else b
