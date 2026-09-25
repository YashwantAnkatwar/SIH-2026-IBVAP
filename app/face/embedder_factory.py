"""
embedder_factory.py

Single place that turns a FACE_EMBEDDER config value (a plain string,
e.g. from the IBVAP_FACE_EMBEDDER environment variable) into an actual
embedder instance. gallery.py depends on this, not on either concrete
embedder module directly, so switching the active backend is a
one-line config change -- exactly the "replaceable embedder"
abstraction the original brief asked for.
"""

_LBP_NAMES = {"lbp", "classic", "legacy", "histogram"}
_DEEP_NAMES = {"facenet", "deep", "vggface2", "arcface", "insightface"}


def create_embedder(name: str):
    """Returns a new embedder instance for the given backend name.
    Raises ValueError on an unrecognized name rather than silently
    falling back, so a typo'd config value is loud, not a silent
    accuracy regression."""
    key = (name or "").strip().lower()

    if key in _LBP_NAMES:
        from face.embedder import Embedder
        return Embedder()

    if key in _DEEP_NAMES:
        # NOTE: "arcface"/"insightface" are accepted as config values
        # for forward-compatibility with the brief's own example
        # (FACE_EMBEDDER=arcface), but the actual deep backend
        # implemented in this build is FaceNet/VGGFace2
        # (embedder_deep.DeepEmbedder) -- see that module's docstring
        # for why, and for what would need to change to genuinely swap
        # in ArcFace/insightface later.
        from face.embedder_deep import DeepEmbedder
        return DeepEmbedder()

    raise ValueError(
        f"Unknown FACE_EMBEDDER backend {name!r}. "
        f"Expected one of {sorted(_LBP_NAMES)} (classical LBP) or "
        f"{sorted(_DEEP_NAMES)} (pretrained deep FaceNet/VGGFace2)."
    )
