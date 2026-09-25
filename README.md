# IBVAP · Integrated Border Video Analytics Platform

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![AI Engine](https://img.shields.io/badge/YOLO-v8%20%2F%20v11%20Nano-emerald.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracker-ByteTrack%20Kinematics-orange.svg)](https://github.com/ifzhang/ByteTrack)
[![Database](https://img.shields.io/badge/Storage-SQLite3%20WAL%20Thread--Local-blue.svg)](https://www.sqlite.org/wal.html)
[![Real-Time](https://img.shields.io/badge/Pub%2FSub-Server--Sent%20Events%20(SSE)-brightgreen.svg)](https://html.spec.whatwg.org/multipage/server-sent-events.html)
[![Test Suite](https://img.shields.io/badge/tests-34%2F34%20passed%20(100%25)-success.svg)](tests/)
[![Smart India Hackathon](https://img.shields.io/badge/SIH%202026-Problem%20Statement%20%23SIH26187-red.svg)](https://www.sih.gov.in/)

> **Smart India Hackathon (SIH 2026) · Problem Statement #SIH26187**  
> **Title**: *AI-Based Intelligent Video Analytics Platform for Border Surveillance using Existing CCTV Infrastructure*  
> **Team**: **SeemaDrishti** | **Theme**: Blockchain & Cybersecurity / Defense | **Category**: Software

---

## 📌 Executive Summary

Modern border defense (BSF / Comprehensive Integrated Border Management System - **CIBMS**) relies on hundreds of kilometers of perimeter CCTV feeds. However, conventional camera systems **only record passive video**, demanding exhausting 24×7 manual screen-staring by jawans. This leads to **operator fatigue, delayed reaction times, and missed infiltrations**.

**IBVAP** transforms existing, passive IP/RTSP CCTV infrastructure into an **AI-powered tactical intelligence network**. It operates completely on software without requiring specialized smart cameras or costly proprietary edge hardware.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 IBVAP PIPELINE OVERVIEW                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
 Existing RTSP/CCTV Streams
        │ (VisDrone Aerial, KAIST Thermal IR, Indian Barrier Checkpoint)
        ▼
 Multi-Camera Ingestion & Normalization (Auto-reconnect, FPS governor, thread workers)
        │
        ▼
 YOLO Detection + ByteTrack Multi-Object Tracking (Kinematics, velocity & heading vectors)
        │
        ▼
 Spatial Geofencing & Virtual Perimeter Lines (Polygon ray-casting, directional tripwires)
        │
        ▼
 Specialized Surveillance Engines:
   ├── Indian HSRP ANPR (Slot-aware OCR character disambiguation & state mapping)
   ├── Facial Biometrics (Authorized BSF whitelist gallery vs. unknown intruder flagging)
   └── Night/Thermal Enhancement (Per-frame CLAHE contrast equalization & luminance tracking)
        │
        ▼
 Behavioral Risk Engine (Dwell/loitering accumulation, run-to-cover acceleration, spatial clustering)
        │
        ▼
 Spatio-Temporal Event Correlator (30s cross-camera sliding window, 0-100 composite risk)
        │
        ▼
 Tactical Dispatch & Command Dashboard (Real-time SSE push, BSF SOP checklist, exportable dossiers)
```

---

## 🚀 Key Features by Architecture Phase

### Phase 1: Multi-Camera Stream Ingestion & Resilient Edge Workers
- **Tri-Camera Topology**: Pre-configured with distinct border camera perspectives:
  - `BOP-01`: **Sector North (Aerial Drone Patrol)** — VisDrone reconnaissance footage for wide-area tracking.
  - `BOP-02`: **Sector East (Night Thermal Sentry)** — KAIST multispectral thermal footage with CLAHE enhancement.
  - `BOP-03`: **Main Barrier (Vehicle Checkpost)** — Checkpoint surveillance with high-resolution Indian ANPR and face recognition.
- **Zero-Crash Resilience**: Automatic exponential reconnection, non-blocking frame buffer decimation, and infinite video looping for uninterrupted evaluation.
- **Dynamic Configuration**: Override any camera with an RTSP stream via environment variables without editing code.

### Phase 2: Detection, Multi-Object Tracking & Virtual Geofencing
- **Kinematic State Estimation**: Real-time ByteTrack tracking computing bounding coordinates, instantaneous velocity ($\text{px/s}$), acceleration ($\text{px/s}^2$), and heading vectors.
- **Polygon Geofences**: Point-in-polygon ray-casting for `RESTRICTED` (Zero-Line fence), `MONITORED` (patrol track), and `CHECKPOINT` zones.
- **Directional Virtual Tripwires**: Two-point vector crossing detectors calculating ingress vs. egress direction across virtual border lines.

### Phase 3: Indian HSRP ANPR, Face Biometrics & Night Enhancement
- **Authentic Indian HSRP License Plate Recognition**:
  - Aligned with real Indian High Security Registration Plate (HSRP) standards.
  - Slot-aware character disambiguation:
    - Positions 0–1: Strict 2-letter state code (`JK`, `PB`, `DL`, `HR`, `UP`, `MH`, etc.).
    - Positions 2–3: 2-digit RTO district code (`0` disambiguated from letter `O`/`Q`).
    - Positions 4–5: 1–3 letter series.
    - Positions 6–9: 1–4 digit unique vehicle number (`B` disambiguated from `8`, `O` from `0`).
  - Strict rejection of car decals and non-plate text (`WRITES`, `FEAR50`).
  - Tested and verified across 10 authentic ground-truth Indian registrations: `JK02BF5058`, `JK02AK8967`, `JK03D7750`, `JK01AD3147`, `PB11BG7347`, `PB11BN6101`, `PB08CQ3690`, `PB07BY3563`, `DL6CJ8404`, `DL9CAE1359`.
- **Facial Biometrics & Sentry Whitelisting**:
  - Deep face embeddings matched against authorized BSF personnel in `gallery/` via cosine distance.
  - Verified soldiers tagged as `KNOWN` (*Sub-Inspector Rajesh Kumar*); unknown entries tagged as `UNKNOWN` with elevated risk points.
- **Night-Vision CLAHE Thermal Equalization**:
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) dynamically reveals hidden silhouettes in zero-light thermal video (`BOP-02`).

### Phase 4: Behavioral Risk Engine & Multi-Event Incident Correlator
- **Traceable, Explainable Analytics**: Every alert names the exact measurable physical trigger; no unexplainable "AI danger guessing".
- **Dwell Time & Loitering**:
  - Dwell timer ($T_{\text{dwell}} = t_{\text{now}} - t_{\text{entry}}$).
  - Warning Threshold: $> 5.0\text{s}$ ($\text{Score} +20$, `HIGH`).
  - Critical Loitering: $> 15.0\text{s}$ ($\text{Score} +35$, `CRITICAL`).
- **Sudden Acceleration ("Run-to-Cover" Detection)**:
  - Detects rapid velocity surges ($> 4.5\text{ m/s}$) and acceleration spikes directed toward border fence ditches.
- **Spatial Incursion Clustering**:
  - Graph-based Euclidean clustering ($R \le 250\text{px}$) detects coordinated squad incursions ($\ge 3$ individuals), escalating risk automatically.
- **Camera Tampering Detection**:
  - Detects camera blinding, lens spray, physical occlusion, and signal freeze via Laplacian variance and mean luminance disparity.
- **Cross-Camera Incident Correlation**:
  - Maintains a 30-second temporal sliding window linking events across camera feeds by track continuity, vehicle plate, or face identity into unified **Tactical Incidents**.

### Phase 5: High-Concurrency Edge Hardening, SSE & Command Dossiers
- **Sub-50ms Real-Time Push**: Server-Sent Events (`/api/stream`) replaces network-heavy polling, streaming events to all clients instantly.
- **SQLite WAL Mode & Thread-Local Storage**: Database reads execute lock-free via `threading.local()` connections, preventing lock starvation between camera threads and operator queries.
- **Tactical Incident Dossier & SOP Action Checklist**:
  - Complete evidence modal with BSF CIBMS Tactical SOP dispatch checkboxes (QRF deployment, thermal drone confirmation, lockdown).
  - **One-Click Export**: Export comprehensive incident dossiers to Markdown (`.md`) or JSON.
- **Automated Compliance Audit Trail**:
  - Tamper-evident lifecycle logging (`SYSTEM_BOOT`, `CAMERA_ONLINE`, `STREAM_DROPPED`, `TAMPER_DETECTED`, `CRITICAL_THREAT`).
  - Export audit logs to CSV or JSON.
- **Live Operator Controls**:
  - Per-camera **Pause/Resume** (suspends detection while keeping stream connected).
  - Per-camera **Stop/Start** (reloads YOLO model on worker thread).
  - Global **Confidence Slider** in the header updating all detectors live on the next frame.

---

## 🖥️ Live Dashboard Tour

### 1. Tri-Camera Command Center Overview
Simultaneous real-time monitoring across aerial drone recon (`BOP-01`), night thermal infrared (`BOP-02`), and border barrier checkpost (`BOP-03`):

```
+-----------------------------------------------------------------------------------------------+
| IBVAP · Border Surveillance Console  [Capt. Yashwant - OFFICER]  [HQ SYNC: CONNECTED] [GOOD]  |
+-----------------------------------------------------------------------------------------------+
|  [BOP-01: VisDrone Aerial]      |  [BOP-02: KAIST Thermal IR]     |  [BOP-03: Checkpoint Barrier] |
|  - UAV Perimeter Sweep          |  - CLAHE Night Equalization     |  - Indian HSRP ANPR OCR       |
|  - Speed / Loitering Vectors    |  - Tripwire Crossing            |  - Face Sentry Verification   |
+-----------------------------------------------------------------------------------------------+
```

### 2. High-Accuracy Indian ANPR & Face Verification
Verified Indian registrations displayed with monospace HSRP font styling, state jurisdiction badges, confidence levels, and direct links to captured evidence crops:

| Timestamp | Camera | Track ID | License Plate | State Jurisdiction | Confidence | Evidence |
|---|---|---|---|---|---|---|
| `2026-09-20 14:38:22` | `BOP-03` | `BOP-03:2715` | `PB11BN6101` | `(Punjab)` | `78%` | [View Crop] |
| `2026-09-20 14:37:13` | `BOP-03` | `BOP-03:2194` | `PB11BG7347` | `(Punjab)` | `86%` | [View Crop] |
| `2026-09-20 14:35:51` | `BOP-03` | `BOP-03:1642` | `JK01AD3147` | `(Jammu and Kashmir)` | `78%` | [View Crop] |
| `2026-09-20 14:33:17` | `BOP-03` | `BOP-03:603` | `JK02AK8967` | `(Jammu and Kashmir)` | `78%` | [View Crop] |
| `2026-09-20 14:32:12` | `BOP-03` | `BOP-03:31` | `JK02BF5058` | `(Jammu and Kashmir)` | `78%` | [View Crop] |
| `2026-09-20 14:31:03` | `BOP-03` | `BOP-03:9532` | `PB07BY3563` | `(Punjab)` | `78%` | [View Crop] |
| `2026-09-20 14:29:48` | `BOP-03` | `BOP-03:8946` | `PB08CQ3690` | `(Punjab)` | `78%` | [View Crop] |
| `2026-09-20 14:22:58` | `BOP-03` | `BOP-03:5179` | `DL9CAE1359` | `(Delhi)` | `78%` | [View Crop] |
| `2026-09-20 14:21:24` | `BOP-03` | `BOP-03:4531` | `DL6CJ8404` | `(Delhi)` | `78%` | [View Crop] |

### 3. Tactical Incident Dossier Modal & Military SOP Action Checklist
When an incident is selected, operators inspect the complete timeline, contributing behavioral risk factors, captured evidence snapshots, and execute standardized BSF CIBMS SOP procedures:
- `[ ]` Dispatch Quick Reaction Team (QRF) to Sector Coordinate
- `[ ]` Slew-to-Cue Thermal PTZ Drone to Target
- `[ ]` Initiate Perimeter Gate & Barrier Lockdown
- **Actions**: `[Export Dossier (.MD)]` · `[Export Dossier (.JSON)]` · `[Acknowledge]` · `[Resolve]`

---

## 🛠️ System Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                VIDEO INGESTION LAYER                   │
                  │   RTSP Streams / Video Files (Looping & Reconnect)     │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ Frame Queue
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │                 DETECTION & TRACKING                   │
                  │   Ultralytics YOLO (Nano) + ByteTrack Kalman Filter   │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ Track Metadata
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │               SPATIAL & BEHAVIORAL ENGINE              │
                  │  • Polygon Geofencing (Enter/Exit/Dwell)               │
                  │  • Virtual Directional Tripwires                       │
                  │  • Loitering & Sudden Acceleration                     │
                  │  • Euclidean Incursion Clustering (Squad Formation)    │
                  │  • Camera Tamper & Lens Blinding Detector              │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ Behavioral Events
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │              SPECIALIZED ANALYTIC ENGINES              │
                  │  • Indian HSRP ANPR (Tesseract Slot-Disambiguator)     │
                  │  • Face Biometrics (Cosine Gallery Verification)       │
                  │  • Night CLAHE Luminance Equalizer                     │
                  └──────────────────────────┬─────────────────────────────┘
                                             │ Raw Event Stream
                                             ▼
                  ┌────────────────────────────────────────────────────────┐
                  │            SPATIO-TEMPORAL EVENT CORRELATOR            │
                  │  • 30-Second Rolling Incident Window                   │
                  │  • Multi-Camera Threat Deduplication                   │
                  │  • 0-100 Composite Risk Accumulation                   │
                  └─────────────┬────────────────────────────┬─────────────┘
                                │ Write                      │ Publish
                                ▼                            ▼
                  ┌──────────────────────────┐ ┌───────────────────────────┐
                  │     SQLITE WAL STORE     │ │    THREAD-SAFE EVENT BUS  │
                  │  Thread-Local Connection │ │    Server-Sent Events     │
                  └─────────────┬────────────┘ └─────────────┬─────────────┘
                                │                            │
                                └─────────────┬──────────────┘
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │               COMMAND DASHBOARD & API                  │
                  │  Flask Web Server (Port 8000)                          │
                  │  • Live MJPEG Video Feeds                              │
                  │  • Real-Time Tactical Alerts & Heatmaps                │
                  │  • BSF CIBMS Incident Dossier Exporter                 │
                  │  • Automated Compliance Audit Trail                    │
                  └────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
project/
├── app/
│   ├── anpr/                   # Indian HSRP ANPR Engine
│   │   ├── aggregator.py       # Multi-frame consensus & plate stabilization
│   │   ├── ocr.py              # Tesseract OCR & slot-aware syntactic disambiguation
│   │   └── pipeline.py         # End-to-end vehicle crop & ANPR pipeline
│   ├── face/                   # Facial Biometric Verification
│   │   ├── embedder.py         # Face embedding extraction
│   │   ├── gallery.py          # Authorized BSF sentry whitelist gallery
│   │   └── pipeline.py         # Cosine distance verification (KNOWN vs UNKNOWN)
│   ├── static/                 # CSS styling & client assets
│   ├── templates/
│   │   └── dashboard.html      # Glassmorphism tactical dark-mode dashboard
│   ├── alerts.py               # Deduplicated alert lifecycle manager
│   ├── camera.py               # Video capture & reconnection logic
│   ├── camera_manager.py       # Thread coordinator for camera workers
│   ├── camera_worker.py        # Core processing loop per camera stream
│   ├── config.py               # Central configuration & environment flags
│   ├── correlator.py           # Multi-event cross-camera tactical correlator
│   ├── dashboard.py            # Flask endpoints & SSE stream server
│   ├── detector.py             # YOLO detector wrapper
│   ├── event_store.py          # SQLite WAL thread-local persistent storage
│   ├── events.py               # Dataclass event models
│   ├── lines.py                # Virtual crossing lines with direction detection
│   ├── night_detection.py      # Night scene luminance & CLAHE equalizer
│   ├── risk_engine.py          # Explainable 0-100 behavioral risk engine
│   ├── tamper_detector.py      # Camera blinding & tamper detection
│   ├── tracker.py              # ByteTrack tracker adapter
│   └── zones.py                # Polygon geofencing & dwell time manager
├── config/
│   ├── lines.json              # Virtual line coordinates
│   └── zones.json              # Restricted polygon coordinates
├── data/
│   ├── events.db               # SQLite database file (WAL mode)
│   └── indian_plates/          # Indian vehicle dataset annotations & images
├── evidence/                   # Saved JPEG evidence snapshots
├── gallery/                    # Enrolled BSF sentry reference faces
├── models/                     # Model weights directory
├── scripts/                    # Utilities & synthetic feed generators
├── tests/                      # Automated test suite (34 test cases)
├── videos/                     # Demo video streams (VisDrone, KAIST, BOP-03)
├── healthcheck.sh              # Production edge probe script
├── main.py                     # Primary entry point
├── requirements.txt            # Python dependencies
├── run.py                      # Alternative server launcher
└── start_edge.sh               # Production daemon startup script
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
- **Python**: Version 3.10 to 3.12
- **Tesseract OCR**:
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
  - **Windows**: Install via `UB-Mannheim/tesseract/wiki` and add to PATH

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/IBVAP.git
cd IBVAP

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launching the Platform
```bash
# Start the server and camera workers
python run.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### 4. Running as a Production Background Service
```bash
# Start in background with PID tracking and log redirection
./start_edge.sh

# Probe service health and camera metrics
./healthcheck.sh
```

---

## ⚙️ Environment Variables & Custom RTSP Configuration

Override default configurations using environment variables without modifying source files:

| Variable | Default Value | Description |
|---|---|---|
| `IBVAP_HOST` | `127.0.0.1` | Dashboard binding host |
| `IBVAP_PORT` | `8000` | Dashboard binding port |
| `IBVAP_CAMERA_BOP-01` | `videos/bop1_visdrone_aerial.mp4` | Video path or RTSP stream URL for `BOP-01` |
| `IBVAP_CAMERA_BOP-02` | `videos/bop2_kaist_night_thermal.mp4` | Video path or RTSP stream URL for `BOP-02` |
| `IBVAP_CAMERA_BOP-03` | `videos/bop3_indian_checkpoint_traffic.mp4` | Video path or RTSP stream URL for `BOP-03` |
| `IBVAP_CONFIDENCE` | `0.35` | Default YOLO detection confidence threshold |
| `IBVAP_ANPR_ENABLED` | `true` | Enable software Indian ANPR pipeline |
| `IBVAP_FACE_ENABLED` | `true` | Enable deep face biometric gallery verification |

*Example: Connecting live RTSP feeds:*
```bash
export IBVAP_CAMERA_BOP-01="rtsp://admin:pass@192.168.1.101:554/ch0"
export IBVAP_CAMERA_BOP-02="rtsp://admin:pass@192.168.1.102:554/ch0"
export IBVAP_CAMERA_BOP-03="rtsp://admin:pass@192.168.1.103:554/ch0"
python run.py
```

---

## 🧪 Testing & Verification

The repository includes a comprehensive, automated test suite covering unit logic, integration pipelines, and full lifecycle audit scenarios:

```bash
# Run the complete test suite
pytest tests/test_phase3_advanced_analytics.py \
       tests/test_phase5_edge_deployment.py \
       tests/test_audit_feature_scenarios.py -v
```

### Verified Test Matrix:
- **`test_phase3_advanced_analytics.py`** (16 tests): Indian HSRP regex validation, Tesseract character disambiguation, Face gallery enrollment, Cosine verification, CLAHE luminance enhancement.
- **`test_phase5_edge_deployment.py`** (6 tests): SSE pub/sub event distribution, Tactical Incident Dossier Markdown/JSON export, Compliance Audit CSV export, SQLite WAL concurrency, camera worker control endpoints.
- **`test_audit_feature_scenarios.py`** (12 tests): 12 complete military audit and compliance lifecycle scenarios.
- **Result**: `34 passed, 100% green`.

---

## 🏆 Alignment with SIH 2026 Problem Statement (#SIH26187)

| Requirement Specified in Problem Statement | How IBVAP Delivers |
|---|---|
| **Use Existing CCTV Infrastructure** | Zero specialized edge hardware required. Runs on software using standard IP/RTSP streams on commodity CPU/GPU. |
| **Multi-Camera Synchronous Monitoring** | Tri-camera worker architecture (`BOP-01` Aerial, `BOP-02` Thermal, `BOP-03` Barrier) processing in parallel with sub-100ms latency. |
| **Explainable Behavioral Analytics** | Dwell/loitering time accumulation, run-to-cover sudden acceleration, and spatial group incursion clustering with deterministic reasons. |
| **Indian Number Plate Recognition (ANPR)** | Slot-aware HSRP syntactic parser and character disambiguation strictly rejecting non-plate text and supporting Indian states. |
| **Facial Recognition at Checkpoints** | Whitelist gallery verification recognizing authorized jawans and flagging unknown entries. |
| **Night & Low-Light Infiltration Tracking** | Real-time CLAHE equalization revealing hidden silhouettes in zero-light thermal video. |
| **Operator Fatigue Reduction** | Cross-camera incident correlation merging 100+ raw triggers into unified Incident Dossiers with standard BSF CIBMS Tactical SOP checklists. |

---

## 🛡️ License & Acknowledgements

- Developed by **Team SeemaDrishti** for the **Smart India Hackathon 2026**.
- Built using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [ByteTrack](https://github.com/ifzhang/ByteTrack), [OpenCV](https://opencv.org/), [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), and [Flask](https://palletsprojects.com/p/flask/).
- Released for educational, research, and defense technology evaluation purposes.
