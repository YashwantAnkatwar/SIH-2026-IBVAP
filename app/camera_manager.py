"""
camera_manager.py

Owns the collection of CameraWorker threads (one per configured camera),
and exposes simple start/stop/status operations used by main.py and the
dashboard.
"""

import time

from camera_worker import CameraWorker


class CameraManager:

    def __init__(self, camera_config, model_path, confidence):
        self.workers = {}

        for camera_id, cam_cfg in camera_config.items():
            self.workers[camera_id] = CameraWorker(
                camera_id=camera_id,
                source=cam_cfg["source"],
                model_path=model_path,
                confidence=confidence,
                anpr_enabled=cam_cfg.get("anpr_enabled"),
                face_enabled=cam_cfg.get("face_enabled"),
                name=cam_cfg.get("name"),
                purpose=cam_cfg.get("purpose"),
            )

    # ======================================
    # START / STOP
    # ======================================

    def start_all(self):
        for worker in self.workers.values():
            worker.start()

    def stop_all(self):
        for worker in self.workers.values():
            worker.stop()

    # ======================================
    # OPERATOR CONTROLS (Phase 18)
    # ======================================

    def pause_camera(self, camera_id):
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.pause()
        return True

    def resume_camera(self, camera_id):
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.resume()
        return True

    def stop_camera(self, camera_id):
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.stop()
        return True

    def start_camera(self, camera_id):
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.start()
        return True

    def set_confidence(self, value, camera_id=None):
        """Apply a new detection confidence threshold live. If camera_id
        is None, applies to every camera (a global operator control)."""
        targets = (
            [self.workers[camera_id]] if camera_id else list(self.workers.values())
        )
        if camera_id and camera_id not in self.workers:
            return False
        for worker in targets:
            worker.set_confidence(value)
        return True

    def set_feature(self, feature, enabled, camera_id=None):
        """Toggle a specialized pipeline (anpr, face) live on cameras."""
        targets = (
            [self.workers[camera_id]] if camera_id else list(self.workers.values())
        )
        if camera_id and camera_id not in self.workers:
            return False
        for worker in targets:
            worker.set_feature(feature, enabled)
        return True

    def reload_zones_and_lines(self, camera_id=None):
        """Reload zones and lines from config files on the workers."""
        targets = (
            [self.workers[camera_id]] if camera_id else list(self.workers.values())
        )
        if camera_id and camera_id not in self.workers:
            return False
        for worker in targets:
            worker.reload_zones()
            worker.reload_lines()
        return True

    def simulate_tamper(self, camera_id, tamper_type=None):
        """Forward tamper simulation to the target camera worker."""
        worker = self.workers.get(camera_id)
        if worker is None:
            return False
        worker.simulate_tamper(tamper_type)
        return True

    # ======================================
    # STATUS
    # ======================================

    def get_status(self):
        """Lightweight status (used by the console/main.py loop)."""
        status = {}
        for camera_id, worker in self.workers.items():
            status[camera_id] = {
                "status": worker.status,
                "fps": worker.fps,
            }
        return status

    def get_full_status(self):
        """Richer status, used by the dashboard's JSON API."""
        status = {}
        now = time.time()
        for camera_id, worker in self.workers.items():
            status[camera_id] = {
                "name": worker.name,
                "purpose": worker.purpose,
                "status": worker.status,
                "paused": worker.paused,
                "confidence": round(worker.confidence, 2),
                "fps": round(worker.fps, 1),
                "frame_count": worker.frame_count,
                "object_counts": worker.object_counts,
                "active_zones": sorted(worker.active_zones),
                "active_track_count": worker.active_track_count,
                "risk_level": worker.risk_level,
                "active_alerts": worker.active_alerts,
                "error_message": worker.error_message,
                "health": {
                    "frame_failures": worker.frame_failures,
                    "reconnect_attempts": worker.reconnect_attempts,
                    "processing_latency_ms": round(worker.processing_latency_ms, 1),
                    "seconds_since_last_frame": (
                        round(now - worker.last_frame_time, 1)
                        if worker.last_frame_time else None
                    ),
                },
                "last_evidence_path": worker.last_evidence_path,
                "night": worker.night_state,
                "tamper": worker.tamper_state,
                "anpr_enabled": worker.anpr_enabled,
                "face_enabled": worker.face_enabled,
                "recent_anpr_reads": worker.recent_anpr_reads[-10:],
                "recent_face_events": worker.recent_face_events[-10:],
            }
        return status

    def get_system_health(self):
        """Overall multi-camera system health (Phase 16)."""
        total = len(self.workers)
        online = sum(1 for w in self.workers.values() if w.status == "ONLINE")
        if total == 0:
            label = "UNKNOWN"
        elif online == total:
            label = "GOOD"
        elif online == 0:
            label = "CRITICAL"
        else:
            label = "DEGRADED"
        return {"cameras_online": online, "cameras_total": total, "health": label}

    def get_worker(self, camera_id):
        return self.workers.get(camera_id)
