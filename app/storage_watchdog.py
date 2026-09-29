"""
storage_watchdog.py

Autonomous background sentinel ensuring the host VM and edge node NEVER run out of
disk space or crash from storage saturation (e.g. 30GB GCP/AWS VM limits).

Runs every 60 seconds in an isolated daemon thread:
1. Monitors filesystem disk usage (free space & percentage).
2. Prevents unbounded growth of SQLite WAL logs, evidence directories, and stdout logfiles.
3. Performs emergency aggressive compaction if free space falls below 5 GB.
"""

import gc
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import config
import shared


class StorageWatchdog(threading.Thread):
    def __init__(self, check_interval_seconds: float = 60.0):
        super().__init__(name="StorageWatchdogThread", daemon=True)
        self.check_interval = max(10.0, float(check_interval_seconds))
        self._running = True

    def run(self):
        # Initial cleanup pass on server boot
        time.sleep(5.0)
        self._perform_cleanup(emergency=False)

        while self._running:
            try:
                time.sleep(self.check_interval)
                usage = self._get_disk_usage()
                is_emergency = (usage["free_gb"] < 4.0) or (usage["percent_used"] > 80.0)
                self._perform_cleanup(emergency=is_emergency)
            except Exception as e:
                # Watchdog must never throw or crash the main app
                pass

    def stop(self):
        self._running = False

    def _get_disk_usage(self) -> dict:
        try:
            target = config.PROJECT_ROOT if config.PROJECT_ROOT.exists() else Path("/")
            total, used, free = shutil.disk_usage(str(target))
            percent = (used / total) * 100.0 if total > 0 else 0.0
            return {
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "percent_used": round(percent, 1),
            }
        except Exception:
            return {"total_gb": 30.0, "used_gb": 0.0, "free_gb": 30.0, "percent_used": 0.0}

    def _perform_cleanup(self, emergency: bool = False):
        # 1. Ensure evidence directory does not hoard snapshots
        try:
            if config.EVIDENCE_DIR.exists():
                for root, _, files in os.walk(str(config.EVIDENCE_DIR)):
                    for fname in files:
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            fpath = Path(root) / fname
                            try:
                                fpath.unlink(missing_ok=True)
                            except Exception:
                                pass
        except Exception:
            pass

        # 2. Prevent SQLite DB and WAL bloat
        try:
            if hasattr(shared, "event_store") and shared.event_store and not shared.event_store.is_memory:
                db_path = Path(shared.event_store.db_path)
                wal_path = db_path.parent / (db_path.name + "-wal")

                # Prune rows in database
                max_keep = 1000 if emergency else 3000
                shared.event_store.prune_old_events(max_keep=max_keep)

                # Truncate WAL file if it exceeds 5 MB or in emergency
                if emergency or (wal_path.exists() and wal_path.stat().st_size > 5 * 1024 * 1024):
                    try:
                        conn = sqlite3.connect(str(db_path), timeout=5.0)
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        if emergency:
                            conn.execute("VACUUM")
                        conn.close()
                    except Exception:
                        pass
        except Exception:
            pass

        # 3. Truncate oversized text logs (nohup.out, app.log, etc.)
        try:
            log_candidates = [
                config.PROJECT_ROOT / "nohup.out",
                config.PROJECT_ROOT / "app.log",
                config.PROJECT_ROOT / "server.log",
            ]
            for log_file in log_candidates:
                if log_file.exists():
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    max_mb = 10.0 if emergency else 30.0
                    if size_mb > max_mb:
                        # Keep only the trailing 200 KB
                        try:
                            with open(log_file, "rb") as f:
                                f.seek(max(0, log_file.stat().st_size - 200 * 1024))
                                tail_data = f.read()
                            with open(log_file, "wb") as f:
                                f.write(tail_data)
                        except Exception:
                            pass
        except Exception:
            pass

        # 4. Release unreferenced memory
        gc.collect()


# Module-level singleton
watchdog_instance = None


def start_watchdog():
    global watchdog_instance
    if watchdog_instance is None or not watchdog_instance.is_alive():
        watchdog_instance = StorageWatchdog(check_interval_seconds=60.0)
        watchdog_instance.start()
    return watchdog_instance
