"""
correlator.py

Event Correlation and Tactical Incident Relationship Model (Phase 4).

Combines multi-modal, temporally related events (e.g. zone intrusion,
plate reading, facial recognition, line crossing, loitering) occurring
on the same camera stream or involving the same target track/vehicle
within a correlation time window into a single unified Tactical Incident.

Avoids creating dozens of disconnected alerts for the same underlying occurrence
and maintains an explainable timeline of events suitable for command reporting.
"""

import itertools
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

from risk_engine import RiskEngine, RiskLevel


SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
_incident_counter = itertools.count(1)


def _next_incident_id(camera_id: str = "BOP"):
    clean_cam = camera_id.replace(":", "_").replace("-", "_")
    return f"INC-{clean_cam}-{next(_incident_counter):06d}"


def _max_severity(a: str, b: str) -> str:
    return a if SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0) else b


@dataclass
class Incident:
    incident_id: str
    camera_id: str
    primary_track_id: str
    object_type: str
    status: str = "ACTIVE"  # ACTIVE | ACKNOWLEDGED | RESOLVED
    severity: str = "LOW"
    risk_score: int = 10
    start_time: float = field(default_factory=time.time)
    last_time: float = field(default_factory=time.time)
    associated_tracks: Set[str] = field(default_factory=set)
    zones: Set[str] = field(default_factory=set)
    reasons: List[str] = field(default_factory=list)
    evidence_paths: List[str] = field(default_factory=list)
    plate_text: Optional[str] = None
    identity: Optional[str] = None
    events: List[dict] = field(default_factory=list)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, round(self.last_time - self.start_time, 1))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_seconds"] = self.duration_seconds
        d["associated_tracks"] = list(self.associated_tracks)
        d["zones"] = list(self.zones)
        return d


class EventCorrelator:
    """
    Thread-safe event correlator that groups related observations
    into an ongoing incident dossier.
    """

    def __init__(self, correlation_window_seconds: float = 30.0, max_incidents: int = 5000):
        self.correlation_window_seconds = float(correlation_window_seconds)
        self.max_incidents = max_incidents
        self._lock = threading.Lock()
        self._incidents: Dict[str, Incident] = {}

    def correlate(
        self,
        camera_id: str,
        track_id: str,
        object_type: str,
        event_type: str,
        severity: str,
        description: str,
        zone: Optional[str] = None,
        risk_score: Optional[int] = None,
        reasons: Optional[List[str]] = None,
        evidence_path: Optional[str] = None,
        plate_text: Optional[str] = None,
        identity: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> tuple:
        """
        Correlate an incoming event into an ongoing active Incident
        or spawn a new Incident.
        Returns: (incident_dict, is_new_incident: bool)
        """
        now = timestamp if timestamp is not None else time.time()
        reasons = reasons or [description] if description else []

        with self._lock:
            # 1. Search for an existing active incident on this camera within the time window
            matched_inc: Optional[Incident] = None
            for inc in reversed(list(self._incidents.values())):
                if inc.camera_id != camera_id or inc.status != "ACTIVE":
                    continue
                if (now - inc.last_time) > self.correlation_window_seconds:
                    continue

                # Association match criteria:
                # a) Same primary or associated track ID
                same_track = (track_id == inc.primary_track_id or track_id in inc.associated_tracks)
                # b) Same registered license plate
                same_plate = bool(plate_text and inc.plate_text and plate_text == inc.plate_text)
                # c) Same verified person identity
                same_person = bool(identity and inc.identity and identity == inc.identity and identity != "Unknown")
                # d) Group incursion in the same zone
                same_group_zone = bool(object_type == "group" and zone and zone in inc.zones)

                if same_track or same_plate or same_person or same_group_zone:
                    matched_inc = inc
                    break

            event_summary = {
                "event_type": event_type,
                "timestamp": now,
                "description": description,
                "severity": severity,
                "risk_score": risk_score or 0,
                "track_id": track_id,
                "zone": zone,
                "evidence_path": evidence_path,
            }

            if matched_inc is not None:
                # Merge into ongoing incident
                matched_inc.last_time = now
                matched_inc.associated_tracks.add(track_id)
                if zone:
                    matched_inc.zones.add(zone)
                if plate_text:
                    matched_inc.plate_text = plate_text
                if identity and identity != "Unknown":
                    matched_inc.identity = identity
                if evidence_path and evidence_path not in matched_inc.evidence_paths:
                    matched_inc.evidence_paths.append(evidence_path)

                matched_inc.severity = _max_severity(matched_inc.severity, severity)
                if risk_score is not None:
                    matched_inc.risk_score = max(matched_inc.risk_score, risk_score)

                for r in reasons:
                    if r not in matched_inc.reasons:
                        matched_inc.reasons.append(r)

                matched_inc.events.append(event_summary)
                self._enforce_limit()
                return matched_inc.to_dict(), False

            # Spawn new Incident
            inc_id = _next_incident_id(camera_id)
            new_inc = Incident(
                incident_id=inc_id,
                camera_id=camera_id,
                primary_track_id=track_id,
                object_type=object_type,
                severity=severity,
                risk_score=risk_score if risk_score is not None else 20,
                start_time=now,
                last_time=now,
                associated_tracks={track_id},
                zones={zone} if zone else set(),
                reasons=list(reasons),
                evidence_paths=[evidence_path] if evidence_path else [],
                plate_text=plate_text,
                identity=identity if identity != "Unknown" else None,
                events=[event_summary],
            )
            self._incidents[inc_id] = new_inc
            self._enforce_limit()
            return new_inc.to_dict(), True

    def get_incident(self, incident_id: str) -> Optional[dict]:
        with self._lock:
            inc = self._incidents.get(incident_id)
            return inc.to_dict() if inc else None

    def list_incidents(
        self,
        camera_id: Optional[str] = None,
        status: Optional[str] = None,
        min_severity: Optional[str] = None,
        limit: int = 50,
    ) -> List[dict]:
        with self._lock:
            items = list(self._incidents.values())

        if camera_id:
            items = [inc for inc in items if inc.camera_id == camera_id]
        if status:
            items = [inc for inc in items if inc.status == status]
        if min_severity:
            min_rank = SEVERITY_ORDER.get(min_severity, 0)
            items = [inc for inc in items if SEVERITY_ORDER.get(inc.severity, 0) >= min_rank]

        # Newest last_seen first
        items.sort(key=lambda inc: inc.last_time, reverse=True)
        return [inc.to_dict() for inc in items[:limit]]

    def acknowledge(self, incident_id: str) -> Optional[dict]:
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                return None
            inc.status = "ACKNOWLEDGED"
            inc.acknowledged_at = time.time()
            return inc.to_dict()

    def resolve(self, incident_id: str) -> Optional[dict]:
        with self._lock:
            inc = self._incidents.get(incident_id)
            if inc is None:
                return None
            inc.status = "RESOLVED"
            inc.resolved_at = time.time()
            return inc.to_dict()

    def _enforce_limit(self):
        if len(self._incidents) <= self.max_incidents:
            return
        # Drop oldest resolved incidents first, then oldest overall
        ordered = sorted(
            self._incidents.values(),
            key=lambda inc: (inc.status != "RESOLVED", inc.last_time),
        )
        excess = len(self._incidents) - self.max_incidents
        for inc in ordered[:excess]:
            self._incidents.pop(inc.incident_id, None)
