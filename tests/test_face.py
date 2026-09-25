"""
Unit tests for the face recognition pipeline (app/face/) — embedding
distance behavior, gallery matching (known/unknown), and per-track
temporal stabilization. Uses small synthetic face-like images (not real
photos of real people, deliberately, to keep this suite offline and
privacy-clean) so these run fast and deterministically; real-footage
validation against the ChokePoint clip lives separately in
tests/manual_face_chokepoint_check.py.

NOTE on embedder choice in this file: the gallery-mechanics tests below
(reload/enroll/threshold/unknown logic) explicitly construct
`FaceGallery(..., embedder=Embedder())` to pin them to the classical
LBP embedder. That's deliberate, not an oversight: the synthetic
random-noise "faces" here were designed to exercise LBP's texture
statistics, not a face manifold, and are not a fair or meaningful input
for the pretrained deep model (which is the new DEFAULT embedder for
this prototype — see config.FACE_EMBEDDER). The deep embedder gets its
own dedicated tests, against real face crops, in test_face_deep.py.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import cv2
import numpy as np

from face.embedder import Embedder
from face.gallery import FaceGallery
from face.pipeline import FaceRecognitionPipeline, MIN_OBSERVATIONS_TO_STABILIZE
from face.face_detector import FaceDetector


def _synthetic_face(seed: int, size=(120, 120)):
    """A deterministic, seed-dependent synthetic pattern standing in for
    a face crop — not a real face, but a stable, distinguishable texture
    so we can test that the SAME seed embeds close to itself and
    DIFFERENT seeds embed far apart, which is exactly what an embedder
    needs to do, without using anyone's real photo."""
    rng = np.random.RandomState(seed)
    img = (rng.rand(*size, 3) * 255).astype(np.uint8)
    # Smooth it so it isn't pure noise (closer to real image statistics).
    img = cv2.GaussianBlur(img, (7, 7), 0)
    return img


def test_embedder_same_input_gives_identical_vector():
    embedder = Embedder()
    face = _synthetic_face(seed=1)
    v1 = embedder.embed(face)
    v2 = embedder.embed(face)
    assert np.allclose(v1, v2)


def test_embedder_similar_faces_are_closer_than_dissimilar_ones():
    embedder = Embedder()
    base = _synthetic_face(seed=42)
    # A slightly perturbed version of the same face (small brightness change).
    similar = np.clip(base.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    different = _synthetic_face(seed=999)

    v_base = embedder.embed(base)
    v_similar = embedder.embed(similar)
    v_different = embedder.embed(different)

    d_similar = Embedder.distance(v_base, v_similar)
    d_different = Embedder.distance(v_base, v_different)
    assert d_similar < d_different


def test_gallery_empty_reports_unknown_not_fake_match():
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(Path(tmp) / "gallery", embedder=Embedder())
        assert gallery.is_empty()
        result = gallery.match(_synthetic_face(seed=1))
        assert result.identity == "Unknown"
        assert result.is_known is False


def test_gallery_enrolled_identity_matches_itself():
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(Path(tmp) / "gallery", embedder=Embedder())
        face = _synthetic_face(seed=7)
        gallery.enroll_from_crop("person_01", face)
        result = gallery.match(face)
        assert result.identity == "person_01"
        assert result.is_known is True
        assert result.distance < 1e-3  # identical crop -> ~zero distance


def test_gallery_dissimilar_face_reports_unknown():
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(Path(tmp) / "gallery", embedder=Embedder())
        gallery.enroll_from_crop("person_01", _synthetic_face(seed=7))
        result = gallery.match(_synthetic_face(seed=888))
        assert result.identity == "Unknown"
        assert result.is_known is False


def test_gallery_reload_reads_persisted_enrollment_images():
    with tempfile.TemporaryDirectory() as tmp:
        gallery_dir = Path(tmp) / "gallery"
        gallery = FaceGallery(gallery_dir, embedder=Embedder())
        face = _synthetic_face(seed=3)
        save_path = gallery_dir / "person_02" / "enroll_00.jpg"
        gallery.enroll_from_crop("person_02", face, save_path=save_path)
        assert save_path.exists()

        # Fresh gallery instance, reading only from disk.
        gallery2 = FaceGallery(gallery_dir, embedder=Embedder())
        assert "person_02" in gallery2.enrolled_identities
        result = gallery2.match(face)
        assert result.identity == "person_02"


def test_face_detector_returns_none_on_blank_crop():
    detector = FaceDetector()
    blank = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert detector.detect(blank) is None


def test_face_detector_quality_check_rejects_tiny_crop():
    detector = FaceDetector()
    tiny = np.full((10, 10, 3), 200, dtype=np.uint8)
    assert detector.quality_check(tiny) is None


def test_face_detector_quality_check_rejects_flat_blurry_crop():
    detector = FaceDetector()
    flat = np.full((80, 80, 3), 200, dtype=np.uint8)  # zero texture -> zero Laplacian variance
    assert detector.quality_check(flat) is None


def test_pipeline_requires_multiple_observations_before_stabilizing():
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(Path(tmp) / "gallery", embedder=Embedder())
        pipeline = FaceRecognitionPipeline(gallery)
        # Feed frames with an empty/undetectable "person" crop (a flat
        # color patch — no face inside it) — should never emit a result
        # since no face is ever detected, only "no result yet".
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        for i in range(MIN_OBSERVATIONS_TO_STABILIZE):
            result = pipeline.process("CAM1", "CAM1:1", frame, (0, 0, 100, 100), timestamp=float(i))
            assert result is None


def test_pipeline_forget_track_clears_state():
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(Path(tmp) / "gallery", embedder=Embedder())
        pipeline = FaceRecognitionPipeline(gallery)
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        pipeline.process("CAM1", "CAM1:1", frame, (0, 0, 100, 100), timestamp=0.0)
        assert "CAM1:1" in pipeline._tracks
        pipeline.forget_track("CAM1:1")
        assert "CAM1:1" not in pipeline._tracks


def test_pipeline_tracks_are_independent_per_camera_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        gallery = FaceGallery(Path(tmp) / "gallery", embedder=Embedder())
        pipeline = FaceRecognitionPipeline(gallery)
        frame = np.full((200, 200, 3), 128, dtype=np.uint8)
        pipeline.process("CAM1", "CAM1:5", frame, (0, 0, 100, 100), timestamp=0.0)
        pipeline.process("CAM2", "CAM2:5", frame, (0, 0, 100, 100), timestamp=0.0)
        # Same raw track_id (5) on two cameras must not collide because
        # global_track_id includes the camera prefix.
        assert "CAM1:5" in pipeline._tracks
        assert "CAM2:5" in pipeline._tracks
        assert pipeline._tracks["CAM1:5"] is not pipeline._tracks["CAM2:5"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)} face recognition pipeline tests passed.")
