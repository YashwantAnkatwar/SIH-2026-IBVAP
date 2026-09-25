"""
pipeline.py

FaceRecognitionPipeline: the object camera_worker.py calls once per
person detection per frame.

Full chain, matching the brief exactly:

    person detection (upstream, YOLO)
        -> person tracking (upstream, ByteTrack)
        -> face detection (face_detector.py)
        -> face quality filtering (face_detector.py)
        -> face crop (no separate "alignment" step: LBP is computed on
           the raw detected face box, which is a documented
           simplification vs. landmark-based alignment)
        -> face embedding (embedder.py)
        -> gallery matching (gallery.py)
        -> Known / Unknown + confidence
        -> per-track temporal stabilization (this file)
        -> identity + confidence event, at most once per track per
           distinct stabilized outcome — not once per frame.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from face.face_detector import FaceDetector
from face.gallery import FaceGallery

MIN_OBSERVATIONS_TO_STABILIZE = 3   # e.g. "Frame1..3 -> Person_01" in the brief's own example
STABILITY_VOTE_FRACTION = 0.6        # this fraction of recent observations must agree
HISTORY_PER_TRACK = 15


@dataclass
class FaceTrackState:
    observations: List[tuple] = field(default_factory=list)  # (identity, confidence)
    stable_identity: Optional[str] = None
    stable_confidence: float = 0.0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    reported_identity: Optional[str] = None  # last identity already emitted as an event
    last_face_crop: Optional["any"] = None    # most recent face crop, for evidence saving
    last_face_box: tuple = ()
    last_similarity: float = 0.0
    last_threshold: float = 0.42


@dataclass
class FaceRecognitionResult:
    camera_id: str
    person_track_id: str
    identity: str            # "Unknown" or a gallery identity like "person_01"
    confidence: float
    is_known: bool
    first_seen: float
    last_seen: float
    evidence_crop: "any" = None  # BGR numpy array of the face crop, for evidence saving
    similarity: float = 0.0
    threshold: float = 0.42
    face_box: tuple = ()
    observations_count: int = 0


class FaceRecognitionPipeline:
    def __init__(self, gallery: FaceGallery):
        self._detector = FaceDetector()
        self._gallery = gallery
        self._tracks: Dict[str, FaceTrackState] = defaultdict(FaceTrackState)
        self.frames_processed = 0
        self.faces_detected = 0
        self.faces_passed_quality = 0

    def process(self, camera_id, global_track_id, frame, person_box, timestamp) -> Optional[FaceRecognitionResult]:
        self.frames_processed += 1
        state = self._tracks[global_track_id]
        if state.first_seen is None:
            state.first_seen = timestamp
        state.last_seen = timestamp

        x1, y1, x2, y2 = [int(v) for v in person_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return self._maybe_emit(camera_id, global_track_id, state)
        person_crop = frame[y1:y2, x1:x2]

        candidate = self._detector.detect(person_crop)
        if candidate is None:
            return self._maybe_emit(camera_id, global_track_id, state)
        self.faces_detected += 1

        fx1, fy1, fx2, fy2 = candidate.box
        face_crop = person_crop[fy1:fy2, fx1:fx2]
        if self._detector.quality_check(face_crop) is None:
            return self._maybe_emit(camera_id, global_track_id, state)
        self.faces_passed_quality += 1

        match = self._gallery.match(face_crop)
        state.observations.append((match.identity, match.confidence))
        if len(state.observations) > HISTORY_PER_TRACK:
            state.observations.pop(0)
        state.last_face_crop = face_crop
        state.last_face_box = (x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2)
        state.last_similarity = round(max(0.0, 1.0 - match.distance), 3)
        state.last_threshold = self._gallery.match_threshold

        self._recompute(state)
        return self._maybe_emit(camera_id, global_track_id, state)

    def _recompute(self, state: FaceTrackState):
        if len(state.observations) < MIN_OBSERVATIONS_TO_STABILIZE:
            return
        recent = state.observations[-HISTORY_PER_TRACK:]
        counts: Dict[str, list] = defaultdict(list)
        for identity, conf in recent:
            counts[identity].append(conf)

        best_identity, confs = max(counts.items(), key=lambda kv: len(kv[1]))
        if len(confs) / len(recent) < STABILITY_VOTE_FRACTION:
            return  # too inconsistent yet — no stable call this frame

        state.stable_identity = best_identity
        state.stable_confidence = sum(confs) / len(confs)

    def _maybe_emit(self, camera_id, global_track_id, state: FaceTrackState) -> Optional[FaceRecognitionResult]:
        if state.stable_identity is None:
            return None
        if state.stable_identity == state.reported_identity:
            return None  # already reported this exact stabilized outcome for this track

        state.reported_identity = state.stable_identity
        return FaceRecognitionResult(
            camera_id=camera_id,
            person_track_id=global_track_id,
            identity=state.stable_identity,
            confidence=state.stable_confidence,
            is_known=(state.stable_identity != "Unknown"),
            first_seen=state.first_seen,
            last_seen=state.last_seen,
            evidence_crop=state.last_face_crop,
            similarity=state.last_similarity,
            threshold=state.last_threshold,
            face_box=state.last_face_box,
            observations_count=len(state.observations),
        )

    def forget_track(self, global_track_id):
        self._tracks.pop(global_track_id, None)

    def current_identity_for(self, global_track_id):
        state = self._tracks.get(global_track_id)
        if state is None or state.stable_identity is None:
            return None
        return state.stable_identity, state.stable_confidence
