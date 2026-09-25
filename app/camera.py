"""
camera.py

Thin abstraction around cv2.VideoCapture that isolates all the
video-source-specific concerns (opening, reading frames, reporting
resolution/fps, releasing resources) away from the worker/pipeline logic.

Keeping this separate makes it trivial to later swap in a real RTSP/USB
camera source instead of a video file, since only this file would need
to change.
"""

import cv2


class VideoSource:
    """Wraps a single video/camera source (file path, RTSP URL, or device index)."""

    def __init__(self, source):
        self.source = source
        self.cap = None

    def open(self):
        """Attempt to open the underlying source. Returns True on success."""
        # A numeric string like "0" means "use webcam device 0".
        source = self.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)
        return self.cap is not None and self.cap.isOpened()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """Read a single frame. Returns (success, frame)."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    @property
    def width(self):
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def source_fps(self):
        if self.cap is None:
            return 0.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 25.0

    def rewind(self):
        """Rewind back to frame 0 for looping. Reopens source if seek fails."""
        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, _ = self.cap.read()
                if ok:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return True
            except Exception:
                pass
        self.release()
        return self.open()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

