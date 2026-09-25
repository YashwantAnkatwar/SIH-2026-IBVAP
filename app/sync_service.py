"""
sync_service.py

Simulates the Edge-to-Central Command synchronization claimed in SIH26187:
- Border Outposts (BOPs) operate fully offline at the edge.
- Local events are recorded persistently in SQLite.
- A background synchronization worker monitors network connectivity to the
  central command center (e.g. CIBMS / Army BSS HQ).
- When connectivity is ONLINE, queued events are packaged and synchronized in
  batches, advancing the sync bookmark.
- When OFFLINE, events safely accumulate in local SQLite without loss or performance degradation.
"""

import threading
import time
from typing import Dict, Any


class SyncService:
    """Manages event synchronization between local Edge outpost and Central HQ."""

    def __init__(self, event_store, hq_url="https://cibms-hq.border.gov.in/api/v1/sync"):
        self.event_store = event_store
        self.hq_url = hq_url
        self.is_connected = True  # Simulated edge connectivity status
        self.last_synced_id = 0
        self.last_sync_time = 0.0
        self.batch_size = 50
        self.sync_interval = 3.0  # seconds between sync attempts
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Initialize last_synced_id from highest existing event id if desired,
        # or start from 0 so historical events get marked synced.
        self._init_bookmark()

    def _init_bookmark(self):
        try:
            with self.event_store._cursor() as cur:
                cur.execute("SELECT MAX(id) FROM events")
                row = cur.fetchone()
                self.last_synced_id = row[0] or 0
                self.last_sync_time = time.time()
        except Exception:
            self.last_synced_id = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def set_connectivity(self, connected: bool):
        with self._lock:
            self.is_connected = bool(connected)

    def toggle_connectivity(self) -> bool:
        with self._lock:
            self.is_connected = not self.is_connected
            return self.is_connected

    def trigger_sync_now(self) -> int:
        """Manually trigger a sync cycle. Returns number of records synced."""
        return self._do_sync_step()

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            total_events = self.event_store.count()
            pending = max(0, total_events - self.last_synced_id)
            return {
                "connectivity": "ONLINE" if self.is_connected else "OFFLINE",
                "is_connected": self.is_connected,
                "hq_url": self.hq_url,
                "total_events": total_events,
                "synced_events": self.last_synced_id,
                "pending_events": pending,
                "unsynced_events_count": pending,
                "last_sync_time": self.last_sync_time,
                "last_sync_iso": (
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_sync_time))
                    if self.last_sync_time else None
                ),
            }

    def _do_sync_step(self) -> int:
        if not self.is_connected:
            return 0

        synced_count = 0
        try:
            with self.event_store._cursor() as cur:
                cur.execute("SELECT MAX(id) FROM events")
                max_row = cur.fetchone()
                max_id = max_row[0] if max_row and max_row[0] else 0

                if max_id > self.last_synced_id:
                    # Fetch next batch
                    target_id = min(max_id, self.last_synced_id + self.batch_size)
                    cur.execute(
                        "SELECT id, ts, camera_id, event_type, severity, description FROM events WHERE id > ? AND id <= ?",
                        (self.last_synced_id, target_id),
                    )
                    rows = cur.fetchall()
                    synced_count = len(rows)
                    with self._lock:
                        self.last_synced_id = target_id
                        self.last_sync_time = time.time()
        except Exception as e:
            # Sync failures must never crash the application
            pass

        return synced_count

    def _sync_loop(self):
        while self._running:
            time.sleep(self.sync_interval)
            self._do_sync_step()
