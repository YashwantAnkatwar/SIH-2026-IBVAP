#!/usr/bin/env bash
# ==============================================================================
# IBVAP — Integrated Border Video Analytics Platform (SIH 2026)
# Edge Node Appliance Bootstrap & Production Launch Script
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "    IBVAP EDGE SURVEILLANCE NODE — BOOTSTRAP INITIALIZATION      "
echo "    Problem Statement: AI-Based Video Analytics for Border CCTV  "
echo "=================================================================="

# 1. Check Python Virtual Environment
if [ -d "venv" ]; then
    echo "[✓] Virtual environment detected: ./venv"
    PYTHON_EXEC="./venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "[✓] Active virtual environment: $VIRTUAL_ENV"
    PYTHON_EXEC="python"
else
    echo "[!] No active venv found. Checking system python3..."
    PYTHON_EXEC="python3"
fi

# 2. Check Core Model Weights
MODEL_PATH="models/yolov8n.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[!] Model weight not found at $MODEL_PATH"
    echo "[*] Downloading YOLOv8n base weights..."
    mkdir -p models
    curl -L -o "$MODEL_PATH" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    echo "[✓] YOLOv8n weights downloaded successfully."
else
    echo "[✓] YOLOv8 neural network weights verified ($MODEL_PATH)."
fi

# 3. Check OCR Engine (Tesseract) for ANPR
if command -v tesseract >/dev/null 2>&1; then
    echo "[✓] Tesseract OCR engine verified: $(tesseract --version 2>&1 | head -n 1)"
else
    echo "[!] WARNING: Tesseract OCR binary not found in PATH. ANPR will use fallback mode."
fi

# 4. Verify Directory Structure
mkdir -p data/evidence data/audit config
echo "[✓] Persistent database and evidence directories ready."

# 5. Launch IBVAP Server
echo ""
echo "[*] Starting IBVAP Multi-Camera Surveillance Appliance..."
echo "[*] Web Console: http://127.0.0.1:8000"
echo "=================================================================="
exec "$PYTHON_EXEC" run.py
