"""
tamper_detector.py

Detects physical camera tampering in real time:
1. Lens Occlusion (covered by cloth, bag, hand, or spray paint):
   Frame becomes almost uniformly dark or low contrast across the whole sensor.
2. Lens Blinding (laser, flashlight, spotlight shining directly into lens):
   Frame becomes uniformly saturated / blown out.
3. Severe Defocus / Blurring:
   Sharpness (Laplacian variance) collapses compared to the scene's normal baseline.

Uses temporal smoothing (short rolling window) so momentary lighting changes or
transient shadows don't trigger false tamper alarms.
"""

from collections import deque
import cv2
import numpy as np


class TamperDetector:
    """Monitors incoming frames for signs of occlusion, blinding, or defocus."""

    def __init__(self, history_len=10):
        self.history_len = history_len
        self._luminance_history = deque(maxlen=history_len)
        self._std_history = deque(maxlen=history_len)
        self._sharpness_history = deque(maxlen=history_len)
        self.is_tampered = False
        self.tamper_type = "NONE"
        self.tamper_reason = ""

    def update(self, frame) -> dict:
        """Evaluate one frame. Returns tamper diagnosis dictionary."""
        if frame is None or frame.size == 0:
            return {"is_tampered": True, "tamper_type": "NO_SIGNAL", "reason": "Empty frame"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        self._luminance_history.append(mean_lum)
        self._std_history.append(std_lum)
        self._sharpness_history.append(sharpness)

        if len(self._luminance_history) < 3:
            return {"is_tampered": False, "tamper_type": "NONE", "reason": ""}

        avg_mean = sum(self._luminance_history) / len(self._luminance_history)
        avg_std = sum(self._std_history) / len(self._std_history)
        avg_sharpness = sum(self._sharpness_history) / len(self._sharpness_history)

        # 1. Occlusion (lens covered): very low standard deviation and low brightness
        if avg_mean < 15.0 and avg_std < 8.0:
            self.is_tampered = True
            self.tamper_type = "OCCLUSION"
            self.tamper_reason = (
                f"Camera lens obstructed/covered (mean={avg_mean:.1f}, std={avg_std:.1f})"
            )
        # 2. Blinding (laser/floodlight saturation): extreme brightness with low texture variation
        elif avg_mean > 245.0 and avg_std < 10.0:
            self.is_tampered = True
            self.tamper_type = "BLINDING"
            self.tamper_reason = (
                f"Camera lens blinded/saturated (mean={avg_mean:.1f}, std={avg_std:.1f})"
            )
        # 3. Defocus: almost no edge energy despite moderate brightness
        elif avg_sharpness < 6.0 and 20.0 <= avg_mean <= 235.0:
            self.is_tampered = True
            self.tamper_type = "DEFOCUS"
            self.tamper_reason = (
                f"Camera lens severely blurred/defocused (sharpness={avg_sharpness:.1f})"
            )
        else:
            self.is_tampered = False
            self.tamper_type = "NONE"
            self.tamper_reason = ""

        return {
            "is_tampered": self.is_tampered,
            "tamper_type": self.tamper_type,
            "reason": self.tamper_reason,
            "metrics": {
                "mean_luminance": round(avg_mean, 1),
                "std_luminance": round(avg_std, 1),
                "sharpness": round(avg_sharpness, 1),
            },
        }

    def reset(self):
        self._luminance_history.clear()
        self._std_history.clear()
        self._sharpness_history.clear()
        self.is_tampered = False
        self.tamper_type = "NONE"
        self.tamper_reason = ""
