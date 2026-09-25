"""
embedder_deep.py

A genuine pretrained deep face-embedding backend, replacing the LBP
histogram embedder (embedder.py) as the DEFAULT embedder for this
prototype, to fix weak recognition under pose/scale change.

MODEL CHOICE
------------
InceptionResnetV1(pretrained="vggface2") from facenet-pytorch
(https://github.com/timesler/facenet-pytorch, MIT license).
This is a faithful, widely-used PyTorch port of the FaceNet
architecture (Schroff et al., 2015), fine-tuned on VGGFace2. It
outputs a 512-dim embedding.

Why this model and not InsightFace/ArcFace (buffalo_l), which is
generally reported to be the stronger of the two:
  - ArcFace via `insightface` needs onnxruntime plus insightface's own
    dependency chain (scikit-image, easydict, cython builds in some
    versions) and a separate model-zoo download. In the actual
    development sandbox used to build and test this change, disk
    space was extremely constrained (a fresh CPU-only PyTorch install
    alone repeatedly exhausted it), and the added, less predictable
    insightface dependency tree was not something that installed
    reproducibly under that constraint.
  - facenet-pytorch needed only itself + torchvision on top of the
    torch dependency this project already carries for YOLO
    (ultralytics), has a single pip install, and bundles its own
    face aligner (MTCNN, weights shipped in the package itself, no
    extra download) which this module uses for real landmark-based
    alignment (see `_align` below) rather than skipping alignment.
  - This is a documented trade-off, not a claim that FaceNet is the
    best possible choice. If, after evaluation (see
    tests/manual_face_chokepoint_check.py and
    scripts/evaluate_face_embedders.py), FaceNet/VGGFace2's
    improvement over LBP is insufficient for the SIH demo, swapping
    in ArcFace/insightface is the recommended next step, and requires
    changing only this one file plus the FACE_EMBEDDER value in
    config.py — the rest of the pipeline (face_detector.py,
    gallery.py, pipeline.py) does not need to change, by design.

WEIGHTS / LICENSING
--------------------
  - MTCNN's P/R/O-net weights ship inside the facenet-pytorch package
    itself (app/face -> facenet_pytorch/data/*.pt); no network access
    needed for those.
  - The InceptionResnetV1 vggface2 weights (~107MB) are downloaded
    once, on first use, directly from facenet-pytorch's own GitHub
    release asset and cached under ~/.cache/torch/checkpoints/. No
    other network access is performed by this module.
  - facenet-pytorch is MIT-licensed. The underlying VGGFace2 training
    data has its own (research-oriented) terms; a production/
    government deployment should independently confirm those terms
    are acceptable before any operational use — this is flagged here,
    not resolved.

PREPROCESSING / ALIGNMENT
--------------------------
Real face alignment is implemented, not skipped:
  1. Run MTCNN.detect(..., landmarks=True) on the already-detected
     face crop (from face_detector.py) to get 5 facial landmarks
     (eyes, nose, mouth corners) if MTCNN can find them.
  2. If found, compute a similarity transform (uniform scale +
     rotation + translation, via cv2.estimateAffinePartial2D) mapping
     those 5 points onto the standard ArcFace-style reference
     landmark positions, scaled to a 160x160 output — this is the
     same reference-point convention widely used for ArcFace/FaceNet
     alignment (a canonical frontal-face template).
  3. If MTCNN cannot find confident landmarks in the crop (can happen
     on partial occlusion, extreme blur, or a very small/poor crop
     that still passed face_detector.py's coarser Haar-cascade-based
     quality check), we do NOT crash and we do NOT fabricate
     landmarks: we fall back to a plain resize of the existing crop to
     160x160. `last_alignment_used` records, per call, whether real
     alignment ran, so this is honestly observable rather than
     silently assumed.
  4. Standardize with facenet-pytorch's own convention,
     (pixel - 127.5) / 128.0, then a single forward pass through
     InceptionResnetV1 in eval mode (no dropout/batchnorm randomness,
     so embeddings are deterministic for a given input).
  5. L2-normalize the resulting 512-dim embedding.

SIMILARITY METRIC
------------------
Cosine similarity (dot product of L2-normalized vectors) — the metric
FaceNet/VGGFace2-style embeddings are meant to be compared with. This
is intentionally NOT the LBP embedder's Euclidean distance; gallery.py
is generic over "higher similarity = closer" so it does not care which
metric is behind it, but the LBP embedder's distance-based threshold
(0.55) is meaningless for this embedding space and must not be reused
here (see config.py / gallery.py for the deep-model-specific
threshold, chosen empirically in scripts/evaluate_face_embedders.py).

RUNTIME
-------
CPU-only in this deployment (torch.cuda.is_available() is False in the
sandbox this was built/tested in). Latency is measured directly in
scripts/evaluate_face_embedders.py and reported in the final report,
rather than asserted here.
"""

import threading

import cv2
import numpy as np
import torch

_MODEL_LOCK = threading.Lock()  # model construction/download is not thread-safe; guard it

# Standard 5-point reference landmarks for a 112x112 aligned face crop
# (left eye, right eye, nose tip, left mouth corner, right mouth
# corner) -- the widely-used ArcFace/insightface alignment template.
# Scaled to whatever OUTPUT_SIZE we actually feed the embedder.
_REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

OUTPUT_SIZE = 160  # InceptionResnetV1(pretrained="vggface2") expects 160x160 input


def _reference_landmarks(output_size: int) -> np.ndarray:
    scale = output_size / 112.0
    return _REFERENCE_LANDMARKS_112 * scale


class DeepEmbedder:
    """Drop-in replacement for embedder.Embedder with the same
    `.embed(face_crop) -> np.ndarray` contract, plus a `similarity()`
    metric appropriate for this embedding space (see module docstring)."""

    EMBEDDING_DIM = 512
    NAME = "facenet_vggface2"

    def __init__(self, device: str = "cpu"):
        # Imported lazily so environments that only need the LBP
        # fallback are never forced to have torch/facenet-pytorch
        # installed just to import this package.
        from facenet_pytorch import MTCNN, InceptionResnetV1

        self.device = torch.device(device)
        with _MODEL_LOCK:
            self._mtcnn = MTCNN(
                image_size=OUTPUT_SIZE, margin=0, min_face_size=20,
                post_process=False, select_largest=True, keep_all=False,
                device=self.device,
            )
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self._reference = _reference_landmarks(OUTPUT_SIZE)
        # Diagnostics only, updated by the most recent embed() call --
        # lets callers/tests/evaluation scripts observe honestly
        # whether real landmark alignment ran or the plain-resize
        # fallback was used, without embed() needing a richer return type.
        self.last_alignment_used = False

    @torch.no_grad()
    def embed(self, face_crop: np.ndarray) -> np.ndarray:
        """Returns a 512-D float32 L2-normalized embedding. Deterministic
        for a given input (model is in eval mode)."""
        if face_crop is None or face_crop.size == 0:
            self.last_alignment_used = False
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        aligned = self._align(rgb)
        if aligned is not None:
            input_rgb = aligned
            self.last_alignment_used = True
        else:
            input_rgb = cv2.resize(rgb, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            self.last_alignment_used = False

        tensor = self._to_tensor(input_rgb).unsqueeze(0).to(self.device)
        embedding = self._model(tensor)[0].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _align(self, rgb: np.ndarray):
        """Real 5-point similarity-transform alignment. Returns an
        aligned OUTPUT_SIZE x OUTPUT_SIZE RGB image, or None if MTCNN
        found no usable landmarks in this crop (honest fallback, not
        a crash and not a fabricated alignment)."""
        try:
            boxes, probs, points = self._mtcnn.detect(rgb, landmarks=True)
        except Exception:
            return None
        if boxes is None or points is None or len(points) == 0 or points[0] is None:
            return None

        src = np.array(points[0], dtype=np.float32)  # 5x2
        if src.shape != (5, 2):
            return None

        transform, inliers = cv2.estimateAffinePartial2D(
            src, self._reference, method=cv2.LMEDS,
        )
        if transform is None:
            return None

        aligned = cv2.warpAffine(rgb, transform, (OUTPUT_SIZE, OUTPUT_SIZE), borderValue=0.0)
        return aligned

    @staticmethod
    def _to_tensor(rgb_img: np.ndarray) -> torch.Tensor:
        img = rgb_img.astype(np.float32)
        img = (img - 127.5) / 128.0  # facenet-pytorch's fixed_image_standardization convention
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    @staticmethod
    def similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity of two L2-normalized vectors: 1.0 =
        identical direction, 0.0 = orthogonal, -1.0 = opposite. This is
        the metric this embedding space should be compared with."""
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def compare(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Generic embedder-agnostic hook used by gallery.py, on the
        same "smaller = more alike" scale as embedder.Embedder.compare
        (which is a Euclidean distance): 1 - cosine_similarity, so 0.0
        is identical and larger is more different (bounded [0, 2]).
        gallery.py's threshold/confidence logic is therefore identical
        for either embedder backend; only the number that counts as a
        good threshold differs, which is why config.py keeps a
        separate default per FACE_EMBEDDER value instead of reusing
        the LBP-tuned 0.55."""
        return 1.0 - DeepEmbedder.similarity(vec_a, vec_b)
