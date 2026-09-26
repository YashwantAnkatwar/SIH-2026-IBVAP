"""
events.py

Defines all event types the pipeline can produce (detections, zone
transitions, risk alerts, camera status changes) and a small
thread-safe EventBus that camera workers publish into and the
dashboard reads from.

The bus keeps only a bounded number of recent events in memory (this is
a demo prototype, not a persistent audit log), which is sufficient to
show a live "recent events" feed in the UI.
"""

import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class TrackEvent:
    """A single detection/track observation in one frame."""
    camera_id: str
    track_id: str
    object_type: str
    confidence: float
    center_x: int
    center_y: int
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "DETECTION"
    event_type: str = "OBJECT_DETECTED"


@dataclass
class ZoneEvent:
    """A zone entry / exit / dwell-update for a track."""
    camera_id: str
    track_id: str
    object_type: str
    zone: str
    zone_type: str
    dwell_seconds: float
    kind: str  # ZONE_ENTER | ZONE_EXIT | ZONE_DWELL
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "ZONE"
    event_id: str = ""
    direction: str = ""
    box: tuple = ()
    severity: str = ""
    event_type: str = ""

    def __post_init__(self):
        if not self.event_type:
            self.event_type = self.kind


@dataclass
class RiskEvent:
    """A risk assessment / alert for a track."""
    camera_id: str
    track_id: str
    object_type: str
    risk_level: str
    reasons: list
    zone: str = ""
    risk_score: int = 0
    event_type: str = "ZONE_RISK"  # e.g. LOITERING, GROUP_INCURSION, LINE_CROSSING, UNUSUAL_MOVEMENT
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "RISK"


@dataclass
class LineCrossingEvent:
    """A track crossing a configured virtual line."""
    camera_id: str
    track_id: str
    object_type: str
    line: str
    direction: str
    from_label: str
    to_label: str
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "LINE"
    event_type: str = "LINE_CROSSING"
    event_id: str = ""
    box: tuple = ()
    severity: str = ""


@dataclass
class IntrusionEvent:
    """Unified intrusion / geofence / tripwire event structure for border surveillance."""
    event_id: str
    camera_id: str
    timestamp: str
    track_id: str
    object_class: str
    zone_or_line: str
    direction: str
    bounding_box: tuple
    event_type: str
    severity: str
    event_kind: str = "INTRUSION"
    metadata: dict = field(default_factory=dict)


def create_intrusion_event(
    camera_id, track_id, object_class, zone_or_line,
    direction="INTRUSION", bounding_box=(), event_type="ZONE_INTRUSION",
    severity="HIGH", event_id=None, timestamp=None, **metadata
):
    import uuid
    eid = event_id or f"EVT-{camera_id}-{uuid.uuid4().hex[:8]}"
    ts = timestamp or _now_iso()
    return IntrusionEvent(
        event_id=eid,
        camera_id=camera_id,
        timestamp=ts,
        track_id=track_id,
        object_class=object_class,
        zone_or_line=zone_or_line,
        direction=direction,
        bounding_box=tuple(bounding_box) if bounding_box else (),
        event_type=event_type,
        severity=severity,
        metadata=metadata,
    )


@dataclass
class StatusEvent:
    """A camera coming online/offline/erroring/reconnecting."""
    camera_id: str
    status: str
    detail: str = ""
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "STATUS"
    event_type: str = "CAMERA_STATUS"


@dataclass
class ANPREvent:
    """A stabilized ANPR read: one real vehicle track's plate, after
    temporal aggregation — not a per-frame OCR attempt."""
    camera_id: str
    vehicle_track_id: str
    plate_text: str
    confidence: float
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "ANPR"
    event_type: str = "ANPR_PLATE_READ"
    event_id: str = ""
    plate_box: tuple = ()
    supporting_frames: int = 1
    is_indian: bool = False
    state_name: str = ""
    evidence_path: str = ""


@dataclass
class FaceRecognitionEvent:
    """A stabilized face-recognition outcome for one real person track —
    either a gallery match (Known) or a confirmed Unknown, after
    temporal aggregation across multiple frames."""
    camera_id: str
    person_track_id: str
    identity: str          # gallery identity string, or "Unknown"
    confidence: float
    is_known: bool
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "FACE"
    event_type: str = "FACE_RECOGNITION"
    event_id: str = ""
    similarity: float = 0.0
    threshold: float = 0.42
    face_box: tuple = ()
    evidence_path: str = ""


@dataclass
class NightMovementEvent:
    """A movement detection event under low-light/night conditions inside a zone."""
    camera_id: str
    track_id: str
    zone: str
    speed: float
    luminance: float
    threshold: float
    is_night: bool
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "NIGHT"
    event_type: str = "NIGHT_MOVEMENT"
    event_id: str = ""
    evidence_path: str = ""


@dataclass
class CameraTamperEvent:
    """Camera health / anti-sabotage event (occlusion, blinding, defocus, no-signal)."""
    camera_id: str
    tamper_type: str  # OCCLUSION | BLINDING | DEFOCUS | NO_SIGNAL
    reason: str
    metrics: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "TAMPER"
    event_type: str = "CAMERA_TAMPERED"
    event_id: str = ""


@dataclass
class DwellEvent:
    """Loitering / dwell threshold crossing event in restricted zones."""
    camera_id: str
    track_id: str
    zone: str
    entry_time: float
    current_dwell_time: float
    threshold_crossed: str  # WARNING | CRITICAL
    threshold_seconds: float
    object_type: str = "person"
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "BEHAVIOR"
    event_type: str = "LOITERING"
    event_id: str = ""


@dataclass
class SuddenMovementEvent:
    """Sudden acceleration, sprint, or rapid approach toward a boundary."""
    camera_id: str
    track_id: str
    previous_velocity: float
    current_velocity: float
    direction: str
    acceleration: float
    boundary_relationship: str
    resulting_event: str = "SUDDEN_ACCELERATION"
    object_type: str = "person"
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "BEHAVIOR"
    event_type: str = "SUDDEN_MOVEMENT"
    event_id: str = ""


@dataclass
class GroupIncursionEvent:
    """Multi-person cluster moving together inside a restricted zone."""
    camera_id: str
    zone: str
    track_ids: list = field(default_factory=list)
    count: int = 3
    center: tuple = ()
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "GROUP"
    event_type: str = "GROUP_INCURSION"
    event_id: str = ""


@dataclass
class IncidentEvent:
    """Correlated multi-signal incident update."""
    incident_id: str
    camera_id: str
    primary_track_id: str
    object_type: str
    severity: str
    risk_score: int
    event_count: int
    reasons: list = field(default_factory=list)
    status: str = "ACTIVE"
    timestamp: str = field(default_factory=_now_iso)
    event_kind: str = "INCIDENT"
    event_type: str = "INCIDENT_UPDATE"


def create_track_event(camera_id, track_id, object_type, confidence, center_x, center_y):
    """Kept for backward compatibility with the original API."""
    return TrackEvent(
        camera_id=camera_id,
        track_id=track_id,
        object_type=object_type,
        confidence=confidence,
        center_x=center_x,
        center_y=center_y,
    )


class EventBus:
    """
    Thread-safe event bus with bounded recent history and non-blocking
    subscriber queues for real-time Server-Sent Events (SSE).
    """

    def __init__(self, max_events=5000):
        self._lock = threading.Lock()
        self._events = deque(maxlen=max_events)
        self._subscribers = set()

    def subscribe(self):
        import queue
        q = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            self._subscribers.discard(q)

    def publish(self, event):
        with self._lock:
            self._events.append(event)
            subs = list(self._subscribers)
        # Non-blocking push to all active SSE subscriber queues
        for q in subs:
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def recent(self, limit=50, event_kinds=None):
        with self._lock:
            items = list(self._events)

        if event_kinds:
            items = [e for e in items if e.event_kind in event_kinds]

        items = items[-limit:]
        items.reverse()  # newest first
        return [asdict(e) for e in items]


# A single process-wide bus shared by all camera workers and the dashboard.
event_bus = EventBus()
