"""
gallery.py

A small, file-backed gallery of enrolled identities for demonstration,
matching the brief's required layout exactly:

    gallery/
        person_01/   (one or more face images of that person)
        person_02/
        ...

Each subject's embeddings are averaged into one reference vector per
image (kept as a list, not merged into a single centroid, so a subject
with varied poses/lighting is still matchable). Matching is real
nearest-neighbor search by embedding "distance" (see note below) with
a threshold — "Unknown" is a genuine outcome, not a placeholder,
whenever nothing in the gallery is close enough.

This module is intentionally the only place that knows about
filesystem layout / identity strings, so the gallery can later be
swapped for a real identity-management system (per the brief) without
touching face_detector.py, embedder.py/embedder_deep.py, or
pipeline.py. As of this change, it is ALSO embedder-agnostic: it does
not import a concrete embedder module itself. Which backend is active
(classical LBP, or the new pretrained deep FaceNet/VGGFace2 model) is
decided once, by face.embedder_factory.create_embedder(), based on
config.FACE_EMBEDDER — everything below just calls `.embed()` and
`.compare()` on whatever embedder object it was given.

`.compare(a, b)` is deliberately NOT necessarily a literal distance —
for the LBP embedder it is a Euclidean distance; for the deep embedder
it is `1 - cosine_similarity`. Both are on the same "0.0 = identical,
larger = more different" scale, which is all this module relies on, so
the matching/threshold/confidence logic here is identical either way.
The *threshold value* that counts as "close enough" is NOT shared
between backends (see config.py) — LBP's 0.55 has no meaning for the
deep embedder's very differently-scaled output.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

DEFAULT_MATCH_THRESHOLD = 0.55  # LBP-only default: L2 distance on normalized LBP vectors; tuned empirically, see README
DEFAULT_MATCH_THRESHOLD_DEEP = 0.42  # deep-embedder-only default: 1 - cosine_similarity; see scripts/evaluate_face_embedders.py


@dataclass
class MatchResult:
    identity: str          # e.g. "person_01", or "Unknown"
    confidence: float       # 0..1, derived from distance; higher = more confident
    distance: float         # raw embedder.compare() score, for debugging/tuning (smaller = closer)
    is_known: bool


class FaceGallery:
    def __init__(self, gallery_dir, match_threshold: Optional[float] = None, embedder=None):
        """
        embedder: an object with `.embed(face_crop) -> np.ndarray` and
            `.compare(vec_a, vec_b) -> float` (smaller = more alike).
            If not given, built from config.FACE_EMBEDDER via
            face.embedder_factory — i.e. the deep FaceNet embedder by
            default in this build, unless IBVAP_FACE_EMBEDDER=lbp.
        match_threshold: if not given, uses the threshold that matches
            whichever embedder ended up being used (see config.py) —
            NOT gallery.DEFAULT_MATCH_THRESHOLD, which is LBP-specific.
        """
        self.gallery_dir = Path(gallery_dir)

        if embedder is None:
            import config
            from face.embedder_factory import create_embedder
            embedder = create_embedder(config.FACE_EMBEDDER)
        self._embedder = embedder

        if match_threshold is None:
            match_threshold = _default_threshold_for(self._embedder)
        self.match_threshold = match_threshold

        self._identities: dict[str, List[np.ndarray]] = {}
        self.reload()

    def reload(self):
        """(Re)scans gallery_dir/<identity>/*.jpg|png and computes
        embeddings for every enrollment image found. Safe to call with
        an empty or missing directory — an empty gallery just means
        every face is reported Unknown, which is honest, not an error."""
        self._identities = {}
        if not self.gallery_dir.exists():
            return

        for person_dir in sorted(self.gallery_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            embeddings = []
            for img_path in sorted(person_dir.glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                embeddings.append(self._embedder.embed(image))
            if embeddings:
                self._identities[person_dir.name] = embeddings

    def enroll_from_crop(self, identity: str, face_crop: np.ndarray, save_path: Optional[Path] = None):
        """Enroll one face crop under `identity` at runtime (used by the
        demo-gallery-building script). Optionally also persists the crop
        to disk under gallery_dir/<identity>/ so it survives restarts."""
        embedding = self._embedder.embed(face_crop)
        self._identities.setdefault(identity, []).append(embedding)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), face_crop)

    def match(self, face_crop: np.ndarray) -> MatchResult:
        query = self._embedder.embed(face_crop)

        best_identity = None
        best_distance = float("inf")
        for identity, embeddings in self._identities.items():
            for ref in embeddings:
                d = self._embedder.compare(query, ref)
                if d < best_distance:
                    best_distance = d
                    best_identity = identity

        if best_identity is None or best_distance > self.match_threshold:
            confidence = max(0.0, 1.0 - (best_distance / (self.match_threshold * 2))) if best_identity else 0.0
            return MatchResult(identity="Unknown", confidence=confidence, distance=best_distance, is_known=False)

        confidence = max(0.0, min(1.0, 1.0 - (best_distance / self.match_threshold)))
        return MatchResult(identity=best_identity, confidence=confidence, distance=best_distance, is_known=True)

    @property
    def enrolled_identities(self):
        return list(self._identities.keys())

    def is_empty(self):
        return len(self._identities) == 0


def _default_threshold_for(embedder) -> float:
    """Picks the right default `.compare()` threshold for whichever
    embedder instance is active, since LBP's Euclidean-distance scale
    and the deep embedder's (1 - cosine similarity) scale are not
    comparable numbers. See config.FACE_MATCH_THRESHOLD_LBP /
    config.FACE_MATCH_THRESHOLD_DEEP for where these are actually
    tuned (empirically, via scripts/evaluate_face_embedders.py), and
    config.py's docstring for the full genuine/impostor numbers behind
    them -- these two constants here are just the LAST-RESORT fallback
    used when a FaceGallery is constructed directly (e.g. in a unit
    test) without going through config.py at all.
    """
    name = getattr(embedder, "NAME", "")
    if name == "facenet_vggface2":
        return DEFAULT_MATCH_THRESHOLD_DEEP
    return DEFAULT_MATCH_THRESHOLD
