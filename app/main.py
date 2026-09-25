"""
main.py

Entry point for IBVAP. Starts all configured camera workers (detection +
tracking + zones + risk engine, each in its own thread) and then serves
the web dashboard, which is the primary way to observe the running
system (live feeds, status, events, alerts).

Run from the project root with:  python run.py
or directly with:                python app/main.py
"""

import sys
import time

from camera_manager import CameraManager
from dashboard import create_app

from config import (
    CAMERAS,
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    EVENTS_DB_PATH,
    EVIDENCE_DIR,
)


def main():
    print()
    print("=" * 60)
    print("IBVAP - INTEGRATED BORDER VIDEO ANALYTICS PLATFORM")
    print("=" * 60)
    print()

    if not MODEL_PATH.exists():
        print(f"FATAL: model file not found at {MODEL_PATH}")
        sys.exit(1)

    manager = CameraManager(
        camera_config=CAMERAS,
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD,
    )

    manager.start_all()

    import shared
    shared.event_store.record_audit(
        user="SystemKernel",
        role="ADMIN",
        action="SYSTEM_BOOT",
        target="EDGE_NODE_01",
        details=f"IBVAP edge appliance online. Initialized {len(CAMERAS)} camera streams.",
    )

    print(f"{len(CAMERAS)} camera worker(s) starting...")
    print(f"Dashboard: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"Event history DB: {EVENTS_DB_PATH}")
    print(f"Evidence snapshots: {EVIDENCE_DIR}")
    print("Tabs: Dashboard | Alerts | Analytics | Event History | Audit | Settings")
    print("Press CTRL+C to stop.")
    print()

    # Give workers a moment to attempt their first connection before the
    # dashboard becomes reachable, purely so early page loads don't show
    # every camera as OFFLINE for no reason.
    time.sleep(1)

    app = create_app(manager)

    try:
        app.run(
            host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            threaded=True,
            debug=False,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping IBVAP...")
        try:
            manager.stop_all()
        except Exception:
            pass
        try:
            shared.event_store.record_audit(
                user="SystemKernel",
                role="ADMIN",
                action="SYSTEM_SHUTDOWN",
                target="EDGE_NODE_01",
                details="IBVAP edge appliance graceful shutdown completed.",
            )
        except Exception:
            pass
        print("IBVAP stopped.")


if __name__ == "__main__":
    main()
