"""
aggregator.py

Reduces many noisy per-frame OCR reads for one vehicle track into a
single stable plate result, per Priority-1 requirement #6/#7 in the
project brief: "avoid generating repeated ANPR events for every frame"
and "use temporal aggregation ... choose the most reliable/stable
result".

Approach: for each vehicle track, keep a short rolling history of
(text, confidence) observations. A track is only reported as "stable"
once the same text has been read at least MIN_AGREEING_READS times
(exact string match) with a mean confidence above MIN_CONFIDENCE, OR
once ENOUGH total observations have accumulated that the plurality
text clearly dominates. This is plain confidence-weighted voting, not
a learned model — deliberately simple and auditable.

A track is only reported once as newly-stable; subsequent frames for
the same track with the same stabilized text produce no further
events (the caller/pipeline enforces this via `is_new_result`).
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional


MIN_AGREEING_READS = 2       # identical reads needed before we trust it
MIN_CONFIDENCE = 45.0        # Tesseract mean confidence, 0..100
MAX_HISTORY_PER_TRACK = 20   # bound memory; old noisy reads age out


@dataclass
class TrackPlateState:
    observations: List[tuple] = field(default_factory=list)  # (text, confidence)
    stable_text: Optional[str] = None
    stable_confidence: float = 0.0
    supporting_frames: int = 0
    total_frames_seen: int = 0
    last_box: tuple = ()
    last_timestamp: float = 0.0
    reported: bool = False  # whether the current stable_text has already been emitted


class PlateTrackAggregator:
    def __init__(self):
        self._tracks: Dict[str, TrackPlateState] = defaultdict(TrackPlateState)

    def observe(self, track_key: str, text: str, confidence: float, box=None, timestamp=None):
        """Record one OCR read for a vehicle track. `text` may be empty
        (no usable read this frame) — empty reads are recorded as
        evidence of instability but never become the stable result."""
        state = self._tracks[track_key]
        state.total_frames_seen += 1
        if box:
            state.last_box = tuple(box)
        if timestamp is not None:
            state.last_timestamp = float(timestamp)

        if text:
            state.observations.append((text, confidence))
            if len(state.observations) > MAX_HISTORY_PER_TRACK:
                state.observations.pop(0)
        self._recompute(state)

    def _recompute(self, state: TrackPlateState):
        if not state.observations:
            return

        from anpr.ocr import validate_indian_plate

        # Canonicalize and filter observations to valid Indian HSRP plates with raw fallback
        valid_obs = []
        raw_obs = []
        for text, conf in state.observations:
            val = validate_indian_plate(text)
            if val.is_valid:
                valid_obs.append((val.canonical_plate or text, conf))
            else:
                raw_obs.append((text, conf))

        active_obs = valid_obs if valid_obs else raw_obs
        if not active_obs:
            return

        counts = Counter(text for text, _ in active_obs)
        best_text, best_count = counts.most_common(1)[0]
        confidences = [c for t, c in active_obs if t == best_text]
        mean_conf = sum(confidences) / len(confidences)

        newly_qualifies = (
            best_count >= MIN_AGREEING_READS
            and mean_conf >= MIN_CONFIDENCE
        )
        if not newly_qualifies:
            return

        if state.stable_text is None:
            state.stable_text = best_text
            state.stable_confidence = mean_conf
            state.supporting_frames = best_count
            state.reported = False
        elif state.stable_text != best_text:
            # Switch stable plate if candidate has strictly more support than established plate
            if best_count > state.supporting_frames:
                state.stable_text = best_text
                state.stable_confidence = mean_conf
                state.supporting_frames = best_count
                state.reported = False
        else:
            # Same plate, refine the confidence estimate and support as more reads arrive.
            state.supporting_frames = max(state.supporting_frames, best_count)
            state.stable_confidence = mean_conf

    def get_stable_unreported(self, track_key: str):
        """Returns (text, confidence) once, the first time a track's
        result becomes stable (or changes to a new stable value), then
        None on every subsequent call until something changes again."""
        state = self._tracks.get(track_key)
        if state is None or state.stable_text is None or state.reported:
            return None
        state.reported = True
        return state.stable_text, state.stable_confidence

    def get_stable_details(self, track_key: str):
        """Returns full temporal details: (text, confidence, supporting_frames, box, timestamp)."""
        state = self._tracks.get(track_key)
        if state is None or state.stable_text is None:
            return None
        return (
            state.stable_text,
            state.stable_confidence,
            state.supporting_frames,
            state.last_box,
            state.last_timestamp,
        )

    def get_current(self, track_key: str):
        """Read-only peek at a track's current stable result, whether or
        not it has already been reported. None if not yet stable."""
        state = self._tracks.get(track_key)
        if state is None or state.stable_text is None:
            return None
        return state.stable_text, state.stable_confidence

    def forget(self, track_key: str):
        self._tracks.pop(track_key, None)

    def active_track_keys(self):
        return list(self._tracks.keys())
