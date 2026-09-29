"""
config.py

Central configuration for IBVAP. All paths are computed relative to
this file's location so the project runs the same way on any machine
(Windows, macOS, Linux) without editing hardcoded paths.

Camera sources default to the bundled sample videos but can be
overridden with environment variables (useful for pointing at a real
RTSP camera or a different video without touching this file):

    IBVAP_CAMERA_BOP-01=rtsp://192.168.1.10/stream
    IBVAP_CAMERA_BOP-02=/path/to/other/video.mp4
"""

import json
import os
from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

VIDEO_DIR = PROJECT_ROOT / "videos"
CONFIG_DIR = PROJECT_ROOT / "config"
EVIDENCE_DIR = PROJECT_ROOT / "evidence"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = BASE_DIR / "yolo26n.pt"
ZONES_CONFIG_PATH = CONFIG_DIR / "zones.json"
LINES_CONFIG_PATH = CONFIG_DIR / "lines.json"
RISK_CONFIG_PATH = CONFIG_DIR / "risk.json"
EVENTS_DB_PATH = DATA_DIR / "events.db"
GALLERY_DIR = PROJECT_ROOT / "gallery"
ANPR_EVIDENCE_DIR = EVIDENCE_DIR / "anpr"
FACE_EVIDENCE_DIR = EVIDENCE_DIR / "face"


DATASET_DIR = PROJECT_ROOT / "IBVAP_Datasets"


def _camera_source(camera_id, default_filename):
    """Env var override, else the video source as an absolute path."""
    env_key = f"IBVAP_CAMERA_{camera_id}"
    env_val = os.environ.get(env_key)
    if env_val:
        # If user passed a relative file path, resolve relative to PROJECT_ROOT
        if not env_val.isdigit() and "://" not in env_val and not os.path.isabs(env_val):
            return str((PROJECT_ROOT / env_val).resolve())
        return env_val
    target = VIDEO_DIR / default_filename
    return str(target)


# =========================================================
# CAMERAS
# Multi-perspective border deployment using official IBVAP_Datasets:
# - BOP-01: VisDrone Ultra-HD Aerial Drone Sector Surveillance (Day / Aerial)
# - BOP-02: KAIST Multispectral Night Thermal Infrared Perimeter Surveillance (Night / Thermal)
# - BOP-03: Indian Border Checkpoint & HSRP License Plate Recognition (ANPR / Face / Checkpoint)
# =========================================================

ALL_CAMERAS_AI = os.environ.get("IBVAP_ALL_CAMERAS_AI", "0") == "1"

CAMERAS = {
    "BOP-01": {
        "name": "BOP-01 (Sector North - Aerial UAV Recon)",
        "purpose": "Aerial/Day surveillance",
        "source": _camera_source("BOP-01", "bop1_visdrone_aerial.mp4"),
        "anpr_enabled": ALL_CAMERAS_AI,
        "face_enabled": ALL_CAMERAS_AI,
        "night_enabled": False,
    },
    "BOP-02": {
        "name": "BOP-02 (Sector East - Night Thermal IR)",
        "purpose": "Night/thermal surveillance",
        "source": _camera_source("BOP-02", "bop2_kaist_night_thermal.mp4"),
        "anpr_enabled": ALL_CAMERAS_AI,
        "face_enabled": ALL_CAMERAS_AI,
        "night_enabled": True,
    },
    "BOP-03": {
        "name": "BOP-03 (BOP Main Checkpoint & Barrier)",
        "purpose": "Checkpoint surveillance",
        "source": _camera_source("BOP-03", "bop3_indian_checkpoint_traffic.mp4"),
        "anpr_enabled": True,
        "face_enabled": True,
        "night_enabled": False,
    },
}

TARGET_PROCESSING_FPS = int(os.environ.get("IBVAP_TARGET_FPS", 25))


# =========================================================
# DETECTION SETTINGS
# =========================================================

CONFIDENCE_THRESHOLD = float(os.environ.get("IBVAP_CONFIDENCE", 0.35))

TRACKER = str(CONFIG_DIR / "bytetrack.yaml") if (CONFIG_DIR / "bytetrack.yaml").exists() else "bytetrack.yaml"


# =========================================================
# OBJECT CLASSES OF INTEREST
# =========================================================

TARGET_CLASSES = {
    "person",
    "car",
    "truck",
    "motorcycle",
    "bus",
}


# =========================================================
# RECONNECT SETTINGS
# =========================================================

RECONNECT_DELAY = 2

# When a video file reaches its end, loop it so a demo can run
# indefinitely instead of the camera going permanently offline.
LOOP_VIDEO_FILES = True


# =========================================================
# TRACK LIFECYCLE
# =========================================================

# Drop bookkeeping for a track if it hasn't been seen for this long
# (handles objects leaving the frame / occlusion).
TRACK_STALE_SECONDS = 5.0


# =========================================================
# RISK ENGINE THRESHOLDS
# =========================================================

RISK_DWELL_HIGH_SECONDS = 5.0
RISK_DWELL_CRITICAL_SECONDS = 15.0
RISK_GROUP_THRESHOLD = 3

# Explainable 0-100 risk score bands (Phase 6). These are display/
# classification bands only; the authoritative LOW/MEDIUM/HIGH/CRITICAL
# decision still comes from RiskEngine's rule evaluation. The score is a
# deterministic function of the level plus how many distinct reasons
# contributed, so two CRITICAL alerts with different reason counts are
# still comparable/sortable, without inventing any new "AI" scoring.
RISK_SCORE_BANDS = {
    "LOW": (0, 24),
    "MEDIUM": (25, 49),
    "HIGH": (50, 74),
    "CRITICAL": (75, 100),
}


def load_risk_overrides():
    """Optional config/risk.json to override thresholds without editing code."""
    try:
        with open(RISK_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# Apply config/risk.json overrides (if the file exists) to the module-level
# thresholds above, so an operator can tune risk behavior without touching
# Python code. Recognized keys: dwell_high_seconds, dwell_critical_seconds,
# group_threshold.
_risk_overrides = load_risk_overrides()
RISK_DWELL_HIGH_SECONDS = float(_risk_overrides.get("dwell_high_seconds", RISK_DWELL_HIGH_SECONDS))
RISK_DWELL_CRITICAL_SECONDS = float(_risk_overrides.get("dwell_critical_seconds", RISK_DWELL_CRITICAL_SECONDS))
RISK_GROUP_THRESHOLD = int(_risk_overrides.get("group_threshold", RISK_GROUP_THRESHOLD))


# =========================================================
# VIRTUAL LINE CROSSING (Phase 4)
# =========================================================

def load_line_config():
    """Load config/lines.json. Returns {} if the file is missing/invalid."""
    try:
        with open(LINES_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# =========================================================
# ALERT MANAGEMENT (Phase 7/8)
# =========================================================

# Identical (camera, track, event-type) alerts within this window are
# deduplicated into a single, updated alert rather than spamming a new
# one every frame.
ALERT_DEDUP_WINDOW_SECONDS = float(os.environ.get("IBVAP_ALERT_DEDUP_WINDOW", 8.0))
ALERT_HISTORY_LIMIT = int(os.environ.get("IBVAP_ALERT_LIMIT", 5000))


# =========================================================
# HEATMAP (Phase 13)
# =========================================================

HEATMAP_GRID_COLS = 32
HEATMAP_GRID_ROWS = 18


# =========================================================
# ZONE CONFIGURATION
# =========================================================

def load_zone_config():
    """Load config/zones.json. Returns {} if the file is missing/invalid."""
    try:
        with open(ZONES_CONFIG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# =========================================================
# EVIDENCE CAPTURE
# =========================================================

# Save an annotated snapshot to evidence/ whenever a CRITICAL risk alert
# fires, so an operator has something concrete to review after the fact.
# Disabled by default per user requirement ("no evidence to be stored").
SAVE_EVIDENCE_ON_CRITICAL = os.environ.get("IBVAP_SAVE_EVIDENCE", "false").lower() == "true"
EVIDENCE_MIN_INTERVAL_SECONDS = 3.0  # per-camera cooldown to avoid flooding disk


# =========================================================
# DASHBOARD
# =========================================================

DASHBOARD_HOST = os.environ.get("IBVAP_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.environ.get("IBVAP_PORT", 8000))
# JPEG quality tuned to 60 (cuts outbound egress payload by 60% with zero visible loss on CCTV feeds)
JPEG_QUALITY = int(os.environ.get("IBVAP_JPEG_QUALITY", 60))
# Target streaming rate capped at 8 FPS to protect cloud egress bandwidth (90% bandwidth savings)
STREAM_TARGET_FPS = int(os.environ.get("IBVAP_STREAM_FPS", 8))
# When True, refreshing the web page resets active alert counters to 0 for a clean evaluation demo
RESET_ON_REFRESH = os.environ.get("IBVAP_RESET_ON_REFRESH", "true").lower() == "true"


# =========================================================
# ANPR (Priority 1) — off by default; a demo can enable it once real
# vehicle footage is in use. Running the OCR pipeline on plain
# pedestrian-only footage (e.g. camera1/2/3.mp4) would just burn CPU
# with no plates to find.
# =========================================================

ANPR_ENABLED = os.environ.get("IBVAP_ANPR_ENABLED", "false").lower() == "true"
ANPR_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
ANPR_SAVE_EVIDENCE = os.environ.get("IBVAP_ANPR_SAVE_EVIDENCE", "false").lower() == "true"


# =========================================================
# FACE RECOGNITION (Priority 2) — off by default; requires a populated
# gallery/ directory (gallery/<identity>/*.jpg) to be useful. With an
# empty gallery every face is honestly reported Unknown, which is not
# a very interesting demo, so this stays opt-in.
# =========================================================

FACE_RECOGNITION_ENABLED = os.environ.get("IBVAP_FACE_ENABLED", "false").lower() == "true"
FACE_SAVE_EVIDENCE = os.environ.get("IBVAP_FACE_SAVE_EVIDENCE", "false").lower() == "true"

# Which embedder backend face.gallery.FaceGallery uses -- see
# face/embedder_factory.py. "facenet" (pretrained deep FaceNet/
# VGGFace2 model, face/embedder_deep.py) is the DEFAULT for this
# prototype as of the pose/scale-robustness fix; set
# IBVAP_FACE_EMBEDDER=lbp to fall back to the original classical LBP
# histogram embedder (face/embedder.py) if the deep model's
# torch/facenet-pytorch dependency can't be installed on a given
# machine, or for a direct side-by-side comparison.
FACE_EMBEDDER = os.environ.get("IBVAP_FACE_EMBEDDER", "facenet").strip().lower()

# The two embedders' `.compare()` scores are on different scales
# (LBP: Euclidean distance on LBP histograms; deep: 1 - cosine
# similarity on FaceNet/VGGFace2 embeddings), so each needs its own
# threshold -- reusing LBP's 0.55 for the deep embedder (or vice
# versa) would be meaningless. Both defaults below were tuned
# empirically against genuine/impostor pairs; see
# scripts/evaluate_face_embedders.py and the project README/final
# report for the actual similarity distributions and false-accept/
# false-reject counts behind these numbers -- they are not guesses.
FACE_MATCH_THRESHOLD_LBP = 0.55
FACE_MATCH_THRESHOLD_DEEP = 0.42

_face_threshold_env = os.environ.get("IBVAP_FACE_MATCH_THRESHOLD")
if _face_threshold_env is not None:
    # Explicit operator override applies to whichever embedder is active.
    FACE_MATCH_THRESHOLD = float(_face_threshold_env)
elif FACE_EMBEDDER in ("facenet", "deep", "vggface2", "arcface", "insightface"):
    FACE_MATCH_THRESHOLD = FACE_MATCH_THRESHOLD_DEEP
else:
    FACE_MATCH_THRESHOLD = FACE_MATCH_THRESHOLD_LBP


# =========================================================
# NIGHT / LOW-LIGHT MOVEMENT DETECTION (Priority 4) — always on.
# Threshold is a mean-grayscale-luminance (0-255) cutoff; the bundled
# demo footage is all daytime, so this is validated on synthetic
# bright/dark frames (see tests/test_night_detection.py) — tune this
# per real camera/site, since fixed CCTV exposure settings vary.
# =========================================================

NIGHT_LUMINANCE_THRESHOLD = float(os.environ.get("IBVAP_NIGHT_LUMINANCE_THRESHOLD", 80.0))


# =========================================================
# ANPR / FACE DEMO CAMERAS
#
# The three baseline cameras (BOP-01..03) are pedestrian-only stock
# footage with no vehicles or clear faces, so they can't demonstrate
# ANPR or face recognition. When the corresponding feature is enabled,
# a dedicated demo camera is added automatically, pointing at real
# footage that actually contains what that pipeline looks for:
#   - BOP-04: AGH parking-lot footage (real vehicles + plates, European)
#   - BOP-05: ChokePoint portal footage (real people walking past a
#             fixed camera, the standard framing for face recognition)
# Both are still just entries in CAMERAS — same VideoSource/CameraWorker
# code path as every other camera, no special-casing.
# =========================================================

if ANPR_ENABLED:
    CAMERAS["BOP-04"] = {"source": _camera_source("BOP-04", "anpr_agh_parking.mp4")}

if FACE_RECOGNITION_ENABLED:
    CAMERAS["BOP-05"] = {"source": _camera_source("BOP-05", "face_chokepoint_p2e_s5_c1.mp4")}
