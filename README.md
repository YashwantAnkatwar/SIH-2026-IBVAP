<div align="center">

# 🛡️ IBVAP · Integrated Border Video Analytics Platform
### *AI-Based Intelligent Video Analytics Platform for Border Surveillance using Existing CCTV Infrastructure*

[![Live Prototype](https://img.shields.io/badge/🌐%20Live%20Prototype-Online%20(Port%208000)-00ffcc?style=for-the-badge&logo=googlechrome&logoColor=black)](http://34.93.16.54:8000/)
[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-SIH%202026%20%7C%20%23SIH26187-orange?style=for-the-badge&logo=target)](https://www.sih.gov.in/)
[![Tests Passing](https://img.shields.io/badge/Tests-214%2F214%20Passing%20(100%25)-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![AI Engine](https://img.shields.io/badge/AI%20Engine-YOLO%20Nano%20%2B%20ByteTrack-FF6F00?style=for-the-badge&logo=yolo)](https://github.com/ultralytics/ultralytics)
[![Storage Architecture](https://img.shields.io/badge/Database-SQLite%20WAL%20(Thread--Local)-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/wal.html)

<p align="center">
  <b>Developed by Team SeemaDrishti</b><br>
  <i>Empowering the Border Security Force (BSF) & Comprehensive Integrated Border Management System (CIBMS)</i>
</p>

---

### 🌐 [Click Here to Access the Live Cloud Prototype (http://34.93.16.54:8000/)](http://34.93.16.54:8000/)

<br>

<p align="center">
  <img src="assets/dashboard_overview.png" alt="IBVAP Tactical Command Console" width="95%" style="border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.5); border: 1px solid #1f293d;">
</p>

> 💡 **Notice the "TEST ALL FEATURES" button in the top right?**  
> We have engineered an interactive, one-click evaluation center directly inside the live prototype! Evaluators, jury members, and defense officers can test every single surveillance capability live without needing to download datasets or configure command lines.

---

</div>

## 📑 Table of Contents

1. [Executive Summary & Problem Statement](#-executive-summary--problem-statement)
2. [Why IBVAP Beats Other Approaches](#-why-ibvap-beats-other-approaches)
3. [Live Cloud Prototype & Interactive Test Center](#-live-cloud-prototype--interactive-test-center)
4. [Core Architectural Modules](#-core-architectural-modules)
   - [Phase 1: Resilient Multi-Camera Ingestion](#phase-1-resilient-multi-camera-ingestion)
   - [Phase 2: Detection, ByteTrack Kinematics & Geofencing](#phase-2-detection-bytetrack-kinematics--geofencing)
   - [Phase 3: Specialized Border Analytics (ANPR, Face, Night CLAHE)](#phase-3-specialized-border-analytics)
   - [Phase 4: Behavioral Risk Engine & Cross-Camera Correlator](#phase-4-behavioral-risk-engine--cross-camera-correlator)
   - [Phase 5: Tactical Command Console & Military SOP Dossiers](#phase-5-tactical-command-console--military-sop-dossiers)
5. [Complete 14-Feature Interactive Test Matrix](#-complete-14-feature-interactive-test-matrix)
6. [End-to-End System Architecture](#-end-to-end-system-architecture)
7. [Repository Structure](#-repository-structure)
8. [Quickstart & Installation Guide](#-quickstart--installation-guide)
9. [Connecting Live RTSP / IP Cameras](#-connecting-live-rtsp--ip-cameras)
10. [Automated Verification & Test Suite (214/214 Green)](#-automated-verification--test-suite)
11. [Compliance, Security & Audit Readiness](#-compliance-security--audit-readiness)

---

## 📌 Executive Summary & Problem Statement

### The Critical National Defense Challenge (SIH Problem Statement #SIH26187)
India’s borders traverse thousands of kilometers of rugged terrain—from the riverine swamps of Bengal to the barbed zero-lines of Punjab and Jammu. While thousands of IP/CCTV cameras are deployed along these perimeters under the **Comprehensive Integrated Border Management System (CIBMS)**, conventional cameras remain **fundamentally passive recorders**:
* 🛑 **Severe Operator Fatigue**: Human jawans staring at 20+ split-screens for hours experience attention collapse within 20 minutes, leading to missed incursions.
* 🛑 **High False Alarm Rates**: Wildlife (cattle, dogs, birds) and wind-blown foliage overwhelm conventional motion sensors, causing alert fatigue.
* 🛑 **Hardware Cost Barrier**: Replacing thousands of legacy CCTV cameras with proprietary "smart edge cameras" would require thousands of crores of capital expenditure.
* 🛑 **Lack of Cross-Camera Intelligence**: Incidents tracked across multiple checkpoints are treated as disconnected events rather than a unified incursion timeline.

### The IBVAP Solution
**IBVAP (Integrated Border Video Analytics Platform)** transforms **existing, legacy CCTV and RTSP camera infrastructure** into an autonomous, proactive, AI-driven tactical defense grid **entirely in software**:
* ✅ **100% Hardware Agnostic**: Ingests standard RTSP, ONVIF, IP, or USB feeds without requiring specialized camera chipsets.
* ✅ **Explainable Threat Scoring**: Replaces opaque "black-box" confidence scores with deterministic, auditable 0–100 behavioral risk metrics.
* ✅ **Real Indian HSRP License Plate OCR**: Custom slot-aware Indian registration syntax parser with automatic RTO state mapping.
* ✅ **Zero-Blindspot Sentry Biometrics**: Whitelist face verification for authorized BSF personnel paired with instant intruder alerts.
* ✅ **Cross-Camera Incident Correlation**: Merges multi-point events across sectors within a 30-second temporal sliding window into unified tactical dossiers.

---

## ⚡ Why IBVAP Beats Other Approaches

| Surveillance Capability | Standard Hackathon Submissions | Generic Commercial NVRs | **IBVAP (SeemaDrishti)** |
| :--- | :---: | :---: | :---: |
| **Existing CCTV Compatibility** | Often requires high-end edge AI chips | Requires proprietary brand cameras | **100% Pure Software — Any RTSP/IP Stream** |
| **Verification & Testing** | 5–10 basic script tests | Manual QA checks | **214/214 Automated Tests (100% Green)** |
| **Indian Number Plate Recognition** | Generic Tesseract (fails on Indian fonts) | Expensive LPR add-ons | **Indian HSRP Slot-Aware Syntactic Disambiguator** |
| **Night & Low-Light Enhancement** | None or simple thresholding | IR hardware illumination required | **CLAHE Adaptive Histogram Equalization Engine** |
| **Behavioral Analytics** | Simple bounding box detection | Basic motion trigger | **Loitering, Run-to-Cover Surge & Squad Clustering** |
| **Camera Tampering Detection** | None | Simple signal loss check | **Laplacian Defocus, Occlusion & Lens Spray Detection** |
| **Multi-Camera Correlation** | Isolated per-camera alerts | Manual search | **30s Spatio-Temporal Cross-Camera Correlation** |
| **Military SOP Integration** | Plain text alerts | Generic email/SMS notifications | **BSF CIBMS SOP Checklist & One-Click Dossier Export** |
| **Interactive Test Center** | ❌ No test harness | ❌ No built-in test suite | **✅ 14-Feature Live Interactive Test Harness** |

---

## 🌐 Live Cloud Prototype & Interactive Test Center

### Instant Cloud Access
* **Live Server**: **[http://34.93.16.54:8000/](http://34.93.16.54:8000/)**
* **Preloaded Demo Cameras**:
  - `BOP-01` (Sector North): Aerial Perimeter Drone Patrol (VisDrone surveillance dataset).
  - `BOP-02` (Sector East): Night-Vision Thermal Reconnaissance (KAIST multispectral infrared).
  - `BOP-03` (Main Barrier): Border Checkpoint & Vehicle Inspection (Authentic Indian HSRP traffic).

### 🎯 The "TEST ALL FEATURES" Evaluation Center
Located at the top right of the dashboard, the **"TEST ALL FEATURES"** modal gives evaluators instant access to all 14 core operational capabilities. Each feature launches an isolated, dedicated test stream with real-time analytics, event telemetry, and diagnostic logs:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          IBVAP FEATURE TEST CENTER (14/14 READY)                        │
├──────────────────────────────┬──────────────────────────────┬───────────────────────────┤
│ 1. Intrusion / Zone Detect   │ 6. Speed & Run-to-Cover      │ 11. Edge Satellite Sync   │
│ 2. Virtual Fence Breach      │ 7. Loitering & Overstay      │ 12. Cross-Cam Correlation │
│ 3. Night Movement & Thermal  │ 8. Camera Tamper & Blinding  │ 13. Threat Score & Ledger │
│ 4. Indian HSRP ANPR Engine   │ 9. Incursion Clustering      │ 14. Military SOP Dossier  │
│ 5. Face Biometric Sentry     │ 10. Density Heatmap Analysis │                           │
└──────────────────────────────┴──────────────────────────────┴───────────────────────────┘
```

---

## 🔬 Core Architectural Modules

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 IBVAP PIPELINE OVERVIEW                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
 Existing RTSP/CCTV Streams (BOP-01 Aerial, BOP-02 Night Thermal, BOP-03 Checkpoint)
         │
         ▼
 1. Ingestion Layer (Auto-reconnect, dynamic FPS governor, circular frame decimation)
         │
         ▼
 2. Detection & Tracking (YOLO Nano + ByteTrack Kalman kinematic state estimation)
         │
         ▼
 3. Spatial & Behavioral Rules:
    ├── Polygon Geofencing (Point-in-polygon ray casting: RESTRICTED vs MONITORED)
    ├── Directional Virtual Tripwires (Two-point vector cross product)
    ├── Loitering Timer (Dwell time accumulation > 5s warning, > 15s critical)
    ├── Sudden Acceleration Surge ("Run-to-cover" velocity spike > 4.5 m/s)
    └── Spatial Incursion Clustering (Euclidean distance graph; squad detection >= 3)
         │
         ▼
 4. Specialized Analytic Engines:
    ├── Indian HSRP ANPR (Tesseract OCR + slot-aware syntactic state disambiguation)
    ├── Facial Biometrics (Cosine distance gallery matching: KNOWN vs UNKNOWN)
    ├── Night/Thermal CLAHE (Adaptive histogram equalization & luminance monitoring)
    └── Camera Tamper Guard (Laplacian blur variance & mean luminance shift)
         │
         ▼
 5. Spatio-Temporal Event Correlator (30s multi-camera sliding window, 0-100 risk score)
         │
         ▼
 6. Tactical Command Console & Edge Storage:
    ├── SQLite WAL Thread-Local Database (Lock-free zero-latency reads)
    ├── Server-Sent Events (SSE) Real-Time Push (< 50ms latency)
    ├── BSF CIBMS Tactical SOP Action Checklist (QRF, Thermal Drone, Lockdown)
    └── Exportable Forensic Dossiers (.MD / .JSON) & Tamper-Evident Audit Logs (.CSV)
```

---

### Phase 1: Resilient Multi-Camera Ingestion
* **Thread-Isolated Workers**: Each camera feed runs in an autonomous background thread with dedicated decimation buffers, guaranteeing that a network drop on one camera never freezes others.
* **Auto-Reconnection**: Exponential backoff reconnection logic (1s $\to$ 2s $\to$ 4s $\to$ max 15s) automatically recovers from network disconnects or power cycles.
* **Dynamic Frame Rate Adaptation**: Configurable target FPS throttle (default 15–20 FPS) prevents CPU saturation on constrained edge hardware.

### Phase 2: Detection, ByteTrack Kinematics & Geofencing
* **Ultralytics YOLO (Nano)**: Optimized for rapid CPU/GPU inference, detecting persons, vehicles, bags, and border equipment.
* **ByteTrack Multi-Object Tracking**: Maintains stable track identity across long occlusions, calculating instantaneous velocity ($\text{px/s}$), acceleration ($\text{px/s}^2$), and heading vectors.
* **Ray-Casting Polygon Geofences**: Supports multi-point convex/concave polygonal zones configured via `config/zones.json` (`RESTRICTED` Zero-Line, `MONITORED` buffer strip, `CHECKPOINT`).
* **Directional Virtual Tripwires**: Computes vector cross-products across perimeter lines (`config/lines.json`), distinguishing harmless inside movement from unauthorized border crossings.

### Phase 3: Specialized Border Analytics

#### 1. Indian HSRP Automatic Number Plate Recognition (ANPR)
* **Slot-Aware Syntactic Disambiguation**:
  - Positions 0–1: Strict 2-letter Indian State Code (`JK`, `PB`, `DL`, `HR`, `UP`, `MH`, `GJ`, `KA`, `WB`, etc.).
  - Positions 2–3: 2-digit RTO District Code (disambiguates letter `O`/`Q` $\to$ digit `0`).
  - Positions 4–5: 1–3 letter Series.
  - Positions 6–9: 1–4 digit Registration Number (disambiguates letter `B` $\to$ digit `8`, letter `I` $\to$ digit `1`).
* **Decal & Noise Filter**: Rejects vehicle branding stickers, bumper text (`WRITES`, `POLICE`, `ARMY`), and road noise.
* **Multi-Frame Consensus Aggregator**: Tracks plate reads across consecutive frames, committing the highest-confidence stabilized plate number.

#### 2. Facial Biometrics & Sentry Whitelisting
* Deep face embedding extractor matched against authorized BSF personnel in `gallery/` via cosine distance.
* **Authorized Sentry**: Confirmed match tags personnel as `KNOWN` (*e.g., Sub-Inspector Rajesh Kumar*).
* **Unidentified Individual**: Unmatched faces in perimeter sectors are tagged as `UNKNOWN`, instantly escalating the composite threat score.

#### 3. Night-Vision CLAHE Thermal Equalization
* **Real-Time Contrast Equalization**: Contrast Limited Adaptive Histogram Equalization (CLAHE) dynamically reveals hidden human silhouettes in zero-light thermal video (`BOP-02`).
* **Scene Luminance Monitor**: Calculates per-frame mean luminance ($L \in [0, 255]$); automatically triggers night-mode risk rules when luminance drops below threshold ($L < 80$).

#### 4. Camera Tampering & Lens Blinding Guard
* Computes Laplacian focus variance ($\sigma_{\text{Laplacian}}^2$) and mean luminance disparity to detect:
  - **Camera Blinding**: High-intensity searchlights or laser pointers directed at the lens.
  - **Occlusion & Spray**: Mud, paint, or physical obstruction over the camera housing.
  - **Signal Freeze**: Video frame freeze or sensor disconnect.

### Phase 4: Behavioral Risk Engine & Cross-Camera Correlator
* **Explainable Risk Formulation**: Every alert produces a transparent, deterministic score from 0 to 100 based on physical parameters:
  $$\text{Risk Score} = \sum (\text{Zone Severity} + \text{Dwell Score} + \text{Velocity Surge} + \text{Night Factor} + \text{Tamper Score})$$
* **Dwell & Loitering Accumulation**:
  - Dwell timer: $T_{\text{dwell}} = t_{\text{now}} - t_{\text{entry}}$
  - Loitering Warning: $> 5.0\text{s}$ ($\text{Score} +20$, `HIGH`)
  - Critical Loitering: $> 15.0\text{s}$ ($\text{Score} +35$, `CRITICAL`)
* **Run-to-Cover Surge**: Velocity surges ($> 4.5\text{ m/s}$) towards border fence ditches trigger instant `CRITICAL` incursion alarms.
* **Spatial Incursion Clustering**: Graph-based Euclidean clustering detects coordinated squad formations ($\ge 3$ intruders within $250\text{px}$ radius).
* **30-Second Cross-Camera Correlator**: Links tracks across camera fields of view by timeline, plate match, or biometric identity, merging fragmented triggers into a unified **Tactical Incident**.

### Phase 5: Tactical Command Console & Military SOP Dossiers
* **Sub-50ms Real-Time Push**: Server-Sent Events (`/api/stream`) replaces polling, streaming events to all clients instantly.
* **SQLite WAL & Thread-Local Storage**: Database reads execute lock-free via `threading.local()` connections, preventing lock starvation between camera threads and operator queries.
* **BSF CIBMS SOP Dispatch Checklist**: Operators work through standardized tactical action checklists:
  - `[ ]` Dispatch Quick Reaction Team (QRF) to Sector Coordinate
  - `[ ]` Slew-to-Cue Thermal PTZ Drone to Target
  - `[ ]` Initiate Perimeter Gate & Barrier Lockdown
* **Forensic Exporting**: Export complete incident dossiers to Markdown (`.md`) or JSON, and export compliance audit trails to CSV.

---

## 🎯 Complete 14-Feature Interactive Test Matrix

Every capability below can be triggered live from the **"TEST ALL FEATURES"** button in the prototype:

| ID | Capability | Scenario & Curated Dataset | Measurable Trigger | Expected Outcome |
| :---: | :--- | :--- | :--- | :--- |
| **01** | **Intrusion Detection** | `worker-zone-detection.mp4` | Person steps into restricted left polygon | Bounding box turns red; `ZONE_INTRUSION` alert fired |
| **02** | **Virtual Fence Breach** | `worker-zone-detection.mp4` | Crossing yellow directional tripwire | Direction logged (`INSIDE -> OUTSIDE`); alert raised |
| **03** | **Night Movement** | `night_detection_thermal.mp4` | Person moving in low-light thermal field | CLAHE enhancement applied; night incursion logged |
| **04** | **Indian HSRP ANPR** | `sih_video.mov` | Vehicle checkpoint approach | Plate `AP40EP4678` read, parsed & state tagged |
| **05** | **Face Biometrics** | `videos/output (1).mp4` & `output.mp4` | Sentry checkpoint verification | Whitelist matched (`KNOWN`) vs intruder (`UNKNOWN`) |
| **06** | **Run-to-Cover Surge** | `worker-zone-detection.mp4` | Sudden acceleration toward ditch | Velocity spike $>4.5\text{ m/s}$ detected; risk escalates |
| **07** | **Loitering / Overstay** | `bop1_visdrone_aerial.mp4` | Subject remaining stationary $>5\text{s}$ | Dwell timer accumulates; turns from `HIGH` to `CRITICAL` |
| **08** | **Camera Tampering** | Synthetic occlusion / spray test | Laplacian focus variance drops | `CAMERA_TAMPER_OCCLUSION` alarm triggered |
| **09** | **Squad Incursion** | VisDrone multi-person dataset | $\ge 3$ persons clustered within $250\text{px}$ | Graph clustering flags coordinated squad formation |
| **10** | **Density Heatmap** | High-traffic checkpoint feed | Frame-by-frame spatial accumulation | Real-time density grid overlay generated |
| **11** | **Edge Satellite Sync** | Network drop simulation | Offline queue buffering | Zero events dropped; seamless sync on link recovery |
| **12** | **Cross-Cam Correlate** | Vehicle entering `BOP-01` $\to$ `BOP-03` | 30s temporal sliding window | Multi-camera triggers fused into unified Incident |
| **13** | **Behavioral Risk Ledger** | Multi-factor trigger accumulation | Composite scoring formula | 0–100 score updated live in persistent SQLite WAL |
| **14** | **Military SOP Dossier** | Critical threat scenario | Incident selection in modal | BSF SOP checklist loaded; 1-click Markdown export |

---

## 🏗️ End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PHYSICAL LAYER                                       │
│    RTSP IP Cameras  ·  VisDrone Aerial UAV  ·  KAIST Thermal IR  ·  Border Barrier    │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ RTSP / H.264 / MJPEG
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             STREAM INGESTION & DECODING                                │
│       Thread-Isolated Camera Workers  ·  Auto-Reconnect  ·  Decimation Ring Buffer     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Decoded BGR Frame
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              INFERENCE & TRACKING PIPELINE                             │
│       Ultralytics YOLO (Nano)  ·  ByteTrack Kalman Kinematics  ·  Heading Vectors      │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ Tracks (x, y, w, h, vx, vy, speed)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          SPATIAL, TEMPORAL & BEHAVIORAL RULES                          │
│   Point-in-Polygon Geofence  ·  Directional Tripwire  ·  Loitering  ·  Squad Clustering│
└─────────────────────┬────────────────────────────────────────────────────┬─────────────┘
                      │                                                    │
                      ▼                                                    ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
│        SPECIALIZED ANALYTIC ENGINES      │  │        TAMPER & DEFENSE INTEGRITY        │
│  • Indian HSRP ANPR (Tesseract OCR)      │  │  • Laplacian Focus Variance Analyzer     │
│  • Facial Biometrics (Cosine Gallery)    │  │  • Mean Luminance Flash/Blinding Guard   │
│  • Night CLAHE Luminance Equalizer       │  │  • Video Signal Freeze Detector          │
└─────────────────────┬────────────────────┘  └────────────────────┬─────────────────────┘
                      │                                            │
                      └─────────────────────┬──────────────────────┘
                                            │ Raw Behavioral Triggers
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        SPATIO-TEMPORAL EVENT CORRELATOR (PHASE 4)                      │
│     30-Second Cross-Camera Sliding Window  ·  0-100 Composite Threat Score Formulation │
└───────────────────────┬──────────────────────────────────────────┬─────────────────────┘
                        │ Write Event                              │ Publish
                        ▼                                          ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────────────────┐
│          PERSISTENT EVENT STORE          │  │          THREAD-SAFE EVENT BUS           │
│   SQLite3 WAL Mode (Thread-Local Conns)  │  │   Server-Sent Events (SSE) Pub/Sub Engine│
└───────────────────────┬──────────────────┘  └────────────────────┬─────────────────────┘
                        │                                          │
                        └───────────────────┬──────────────────────┘
                                            │ Real-Time Push (< 50ms)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           TACTICAL COMMAND CONSOLE (PHASE 5)                           │
│  • Multi-Camera Video Wall (MJPEG)        • Real-Time Threat Ledger & Heatmaps         │
│  • BSF CIBMS Tactical SOP Checklist       • One-Click Incident Dossier (.MD / .JSON)   │
│  • Interactive 14-Feature Test Harness    • Tamper-Evident Compliance Audit Log (.CSV) │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
project/
├── app/
│   ├── anpr/                   # Indian HSRP License Plate Recognition
│   │   ├── aggregator.py       # Multi-frame consensus & plate stabilization
│   │   ├── ocr.py              # Tesseract OCR & slot-aware syntactic disambiguation
│   │   └── pipeline.py         # End-to-end vehicle crop & plate OCR pipeline
│   ├── face/                   # Facial Biometric Verification
│   │   ├── embedder.py         # Face embedding extraction
│   │   ├── gallery.py          # Authorized BSF sentry whitelist gallery
│   │   └── pipeline.py         # Cosine distance verification (KNOWN vs UNKNOWN)
│   ├── static/                 # CSS styling, audio alerts & client assets
│   ├── templates/
│   │   └── dashboard.html      # Tactical dark-mode command console
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
│   ├── test_center.py          # Interactive 14-Feature Test Harness engine
│   ├── tracker.py              # ByteTrack tracker adapter
│   └── zones.py                # Polygon geofencing & dwell time manager
├── assets/                     # Repository media & dashboard overview screenshot
├── config/
│   ├── lines.json              # Virtual tripwire coordinates
│   ├── risk.json               # Behavioral risk thresholds & scoring weights
│   └── zones.json              # Restricted polygon geofence coordinates
├── data/
│   ├── events.db               # SQLite database file (WAL mode)
│   └── indian_plates/          # Indian vehicle dataset annotations & images
├── evidence/                   # Saved JPEG forensic snapshots
├── gallery/                    # Enrolled BSF sentry reference faces
├── tests/                      # Automated test suite (214 test cases)
├── videos/                     # Demo video streams (VisDrone, KAIST, BOP-03)
├── conftest.py                 # Pytest configuration & sys.path environment hooks
├── pytest.ini                  # Pytest runner settings & scope isolation
├── healthcheck.sh              # Production edge probe script
├── main.py                     # Primary entry point
├── requirements.txt            # Python dependencies
├── run.py                      # Alternative server launcher
└── start_edge.sh               # Production daemon startup script
```

---

## ⚡ Quickstart & Installation Guide

### 1. Prerequisites
* **Python**: Version 3.9, 3.10, 3.11, or 3.12
* **Tesseract OCR**:
  - **macOS**: `brew install tesseract`
  - **Ubuntu / Debian**: `sudo apt-get update && sudo apt-get install -y tesseract-ocr`
  - **Windows**: Install via `UB-Mannheim/tesseract/wiki` and add to system `PATH`

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
# Start as background daemon with PID tracking and log redirection
./start_edge.sh

# Probe service health, active threads, and camera metrics
./healthcheck.sh
```

---

## ⚙️ Connecting Live RTSP / IP Cameras

Override any default camera with real IP/RTSP camera feeds using environment variables without modifying source code:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `IBVAP_HOST` | `127.0.0.1` | Dashboard binding host (`0.0.0.0` for network-wide access) |
| `IBVAP_PORT` | `8000` | Dashboard binding port |
| `IBVAP_CAMERA_BOP-01` | `videos/bop1_visdrone_aerial.mp4` | Video path or RTSP stream URL for `BOP-01` |
| `IBVAP_CAMERA_BOP-02` | `videos/bop2_kaist_night_thermal.mp4` | Video path or RTSP stream URL for `BOP-02` |
| `IBVAP_CAMERA_BOP-03` | `videos/bop3_indian_checkpoint_traffic.mp4` | Video path or RTSP stream URL for `BOP-03` |
| `IBVAP_CONFIDENCE` | `0.35` | Global YOLO detection confidence threshold |
| `IBVAP_ANPR_ENABLED` | `true` | Enable Indian HSRP ANPR engine |
| `IBVAP_FACE_ENABLED` | `true` | Enable Facial Biometrics sentry verification |

#### Example: Deploying on Real Border CCTV Feeds
```bash
export IBVAP_HOST="0.0.0.0"
export IBVAP_CAMERA_BOP-01="rtsp://admin:BSF_SecurePass@192.168.1.101:554/h264Preview_01_main"
export IBVAP_CAMERA_BOP-02="rtsp://admin:BSF_SecurePass@192.168.1.102:554/h264Preview_01_main"
export IBVAP_CAMERA_BOP-03="rtsp://admin:BSF_SecurePass@192.168.1.103:554/h264Preview_01_main"
python run.py
```

---

## 🧪 Automated Verification & Test Suite

The project includes an exhaustive, battle-tested automated test suite consisting of **214 test cases** covering every algorithmic unit, multi-threaded worker, and end-to-end audit scenario.

```bash
# Run the complete test suite
python3 -m pytest -q
```

### Verified Test Execution Result:
```
........................................................................ [ 33%]
........................................................................ [ 67%]
......................................................................   [100%]
============================== 214 passed in 17.72s ==============================
```

### Test Coverage Highlights:
* **`tests/test_anpr.py`**: Indian HSRP syntax disambiguation, regex state parsing, decal rejection, multi-frame stabilization.
* **`tests/test_face_recognition.py`**: Cosine similarity matching, BSF sentry gallery enrollment, unknown intruder escalation.
* **`tests/test_night_detection.py`**: CLAHE histogram equalization, mean luminance thresholding, low-light silhouette recovery.
* **`tests/test_tamper_detector.py`**: Laplacian blur variance detection, camera blinding, physical occlusion, signal freeze.
* **`tests/test_risk_engine.py`**: Explainable 0–100 threat formulation, dwell time accumulation, run-to-cover acceleration spikes.
* **`tests/test_zones_and_lines.py`**: Ray-casting point-in-polygon calculations, directional crossing vector cross-products.
* **`tests/test_correlator.py`**: 30-second cross-camera rolling window incident fusion and threat deduplication.
* **`tests/test_dashboard_api.py`**: Server-Sent Events (SSE) streaming, SQLite WAL thread-local concurrency, control endpoints.
* **`tests/test_audit_feature_scenarios.py`**: 12 complete military audit and compliance lifecycle scenarios.

---

## 🛡️ Compliance, Security & Audit Readiness

* **Tamper-Evident Audit Logging**: Every system event (`SYSTEM_BOOT`, `CAMERA_ONLINE`, `STREAM_DROPPED`, `TAMPER_DETECTED`, `CRITICAL_THREAT`) is cryptographically structured and stored with millisecond timestamps.
* **Forensic Evidence Dossiers**: Critical risk events automatically preserve annotated full-frame snapshots and high-res crops in `evidence/`.
* **Zero Cloud Dependency (Air-Gapped Ready)**: Operates completely offline without sending video feeds or biometrics to external cloud APIs, strictly adhering to Indian Defense Air-Gap standards.
* **One-Click Export**: Incident dossiers can be exported immediately to Markdown (`.md`) or JSON, and the compliance log exported to CSV for war room debriefs.

---

<div align="center">

### 🏆 Team SeemaDrishti · Smart India Hackathon 2026
*Protecting the Nation's Borders through Sovereign, Scalable Artificial Intelligence.*

**[🌐 Experience the Live Prototype at http://34.93.16.54:8000/](http://34.93.16.54:8000/)**

</div>
