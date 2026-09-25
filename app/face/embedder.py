"""
embedder.py

Computes a real feature vector ("embedding") for an aligned face crop
using a grid of Local Binary Pattern (LBP) histograms — a classical,
well-established texture descriptor (Ahonen et al., 2006, "Face
Recognition with Local Binary Patterns") predating deep embeddings but
still a genuine, non-fabricated, distance-comparable representation:
faces of the same person produce nearby vectors, faces of different
people produce distant vectors.

This is an explicit, documented accuracy/engineering trade-off: a deep
embedding network (FaceNet/ArcFace/InsightFace) would perform
substantially better, but needs a downloaded pretrained model. LBP
needs none — it runs fully offline with only numpy/OpenCV, which is
why it's used for this prototype. `Embedder` is a small, swappable
interface (`.embed(face_crop) -> np.ndarray`) specifically so a deep
model can be dropped in later without touching gallery.py or
pipeline.py.
"""

import cv2
import numpy as np

FACE_SIZE = (100, 100)   # all faces are normalized to this size before LBP
GRID = (8, 8)             # spatial grid; histogram computed per cell
LBP_RADIUS = 1
LBP_NEIGHBORS = 8
LBP_BINS = 2 ** LBP_NEIGHBORS  # 256 possible uniform-radius patterns (not reduced to uniform patterns, for simplicity)


def _lbp_image(gray: np.ndarray) -> np.ndarray:
    """Standard 8-neighbor, radius-1 LBP, vectorized with numpy shifts
    (no per-pixel Python loop, so this stays fast enough for real-time
    use on person-crop-sized images)."""
    h, w = gray.shape
    padded = np.pad(gray, 1, mode="edge").astype(np.int16)

    center = padded[1:-1, 1:-1]
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
    ]
    lbp = np.zeros((h, w), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        lbp |= ((neighbor >= center).astype(np.uint8)) << bit
    return lbp


class Embedder:
    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Returns a 1-D float32 vector (L2-normalized). Deterministic:
        the same crop always produces the same vector."""
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, FACE_SIZE, interpolation=cv2.INTER_LINEAR)
        gray = cv2.equalizeHist(gray)

        lbp = _lbp_image(gray)

        gh, gw = GRID
        ch, cw = FACE_SIZE[1] // gh, FACE_SIZE[0] // gw
        histograms = []
        for gy in range(gh):
            for gx in range(gw):
                cell = lbp[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
                hist, _ = np.histogram(cell, bins=LBP_BINS, range=(0, LBP_BINS))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-6)  # normalize within-cell first
                histograms.append(hist)

        vector = np.concatenate(histograms)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)

    @staticmethod
    def distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Euclidean distance between two L2-normalized vectors — 0.0 is
        identical, larger is more different. Bounded in [0, 2]."""
        return float(np.linalg.norm(vec_a - vec_b))

    @staticmethod
    def compare(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Generic embedder-agnostic hook used by gallery.py: a
        "dissimilarity" score where SMALLER = more alike (same
        orientation/scale as the historical `.distance()` threshold of
        0.55 in gallery.py). For LBP this is just `.distance()` under a
        name that also makes sense for other embedders (see
        embedder_deep.DeepEmbedder.compare, which is NOT a Euclidean
        distance but is on the same "smaller = closer" scale)."""
        return Embedder.distance(vec_a, vec_b)
