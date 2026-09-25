#!/usr/bin/env bash
# ==============================================================================
# IBVAP — Edge Node Health Probe & Liveness Check
# Usage: ./healthcheck.sh [PORT]
# Returns 0 if healthy, 1 if critical/offline
# ==============================================================================

PORT="${1:-8000}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}/api/system"

echo "=== IBVAP Edge Health Probe ==="
echo "Target: ${URL}"

HTTP_CODE=$(curl -s -o /tmp/ibvap_health.json -w "%{http_code}" --max-time 3 "${URL}" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" != "200" ]; then
    echo "[-] FAILED: IBVAP Edge Appliance is UNREACHABLE (HTTP ${HTTP_CODE})"
    exit 1
fi

HEALTH=$(grep -o '"health": *"[^"]*"' /tmp/ibvap_health.json | head -n 1 | cut -d'"' -f4)
ONLINE=$(grep -o '"cameras_online": *[0-9]*' /tmp/ibvap_health.json | head -n 1 | grep -o '[0-9]*')
TOTAL=$(grep -o '"cameras_total": *[0-9]*' /tmp/ibvap_health.json | head -n 1 | grep -o '[0-9]*')
EVENTS=$(grep -o '"total_events_recorded": *[0-9]*' /tmp/ibvap_health.json | head -n 1 | grep -o '[0-9]*')

echo "[✓] Edge Status: ${HEALTH}"
echo "[✓] Cameras: ${ONLINE}/${TOTAL} Online"
echo "[✓] Recorded Events: ${EVENTS}"

if [ "$HEALTH" == "CRITICAL" ] || [ "$ONLINE" == "0" ]; then
    echo "[-] ALERT: Edge node health is CRITICAL or NO CAMERAS ONLINE"
    exit 1
fi

echo "[+] Probe OK: System is operational."
exit 0
