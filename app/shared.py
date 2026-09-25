"""
shared.py

Process-wide singletons that every camera worker thread and the Flask
dashboard both need to see the same instance of: the alert manager,
the persistent event store, and (Priority 2) the face recognition
gallery. Kept in one small module (mirroring the existing event_bus
pattern in events.py) to avoid circular imports between
camera_worker.py and dashboard.py.
"""

import config
from alerts import AlertManager
from event_store import EventStore
from sync_service import SyncService
from correlator import EventCorrelator

alert_manager = AlertManager(
    dedup_window_seconds=config.ALERT_DEDUP_WINDOW_SECONDS,
    history_limit=config.ALERT_HISTORY_LIMIT,
)

correlator = EventCorrelator(
    correlation_window_seconds=config.ALERT_DEDUP_WINDOW_SECONDS * 3.0,
    max_incidents=config.ALERT_HISTORY_LIMIT,
)

event_store = EventStore(config.EVENTS_DB_PATH)
sync_service = SyncService(event_store)
sync_service.start()

# The face gallery is loaded on-demand or at startup so every camera sees
# the same set of enrolled identities.
face_gallery = None
def get_face_gallery():
    global face_gallery
    if face_gallery is None:
        from face.gallery import FaceGallery
        face_gallery = FaceGallery(config.GALLERY_DIR, match_threshold=config.FACE_MATCH_THRESHOLD)
    return face_gallery

if config.FACE_RECOGNITION_ENABLED:
    get_face_gallery()
