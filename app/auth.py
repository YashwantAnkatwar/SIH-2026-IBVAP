"""
auth.py

Role-Based Access Control (RBAC) for the IBVAP Surveillance Console.
Provides role definitions, authorization checks, and session helpers for
Smart India Hackathon 2026 (CIBMS compliance).

Roles:
- OFFICER: Full administrative authority (Zone calibration, threshold changes,
           alert resolution, feature toggles, satellite sync controls).
- JAWAN: Tactical sentry authority (Live monitoring, acknowledging threats,
         flagging false alarms, reading dossiers).
"""

from typing import Dict, Set

ROLE_OFFICER = "OFFICER"
ROLE_JAWAN = "JAWAN"

# Permission definitions
PERM_VIEW_SURVEILLANCE = "VIEW_SURVEILLANCE"
PERM_ACKNOWLEDGE_ALERT = "ALERT_ACKNOWLEDGE"
PERM_FLAG_FALSE_ALARM = "ALERT_FALSE_ALARM"
PERM_RESOLVE_ALERT = "ALERT_RESOLVE"
PERM_CALIBRATE_ZONES = "CALIBRATE_ZONES"
PERM_FEATURE_TOGGLE = "FEATURE_TOGGLE"
PERM_SYNC_OVERRIDE = "SYNC_OVERRIDE"
PERM_CAMERA_CONTROL = "CAMERA_CONTROL"
PERM_CONFIDENCE_ADJUST = "CONFIDENCE_ADJUST"
PERM_VIEW_AUDIT = "VIEW_AUDIT"

ROLE_PERMISSIONS: Dict[str, Set[str]] = {
    ROLE_OFFICER: {
        PERM_VIEW_SURVEILLANCE,
        PERM_ACKNOWLEDGE_ALERT,
        PERM_FLAG_FALSE_ALARM,
        PERM_RESOLVE_ALERT,
        PERM_CALIBRATE_ZONES,
        PERM_FEATURE_TOGGLE,
        PERM_SYNC_OVERRIDE,
        PERM_CAMERA_CONTROL,
        PERM_CONFIDENCE_ADJUST,
        PERM_VIEW_AUDIT,
    },
    ROLE_JAWAN: {
        PERM_VIEW_SURVEILLANCE,
        PERM_ACKNOWLEDGE_ALERT,
        PERM_FLAG_FALSE_ALARM,
        PERM_VIEW_AUDIT,
    },
}

USERS = {
    "officer": {
        "username": "officer",
        "display_name": "Capt. Yashwant",
        "role": ROLE_OFFICER,
        "designation": "Commanding Officer · Sector HQ",
    },
    "jawan": {
        "username": "jawan",
        "display_name": "Sepoy Rajesh",
        "role": ROLE_JAWAN,
        "designation": "Sentry Operator · BOP Post 1",
    },
}

# In-memory active user for the local prototype console (defaults to OFFICER for demonstration)
_active_user = USERS["officer"].copy()


def get_current_user():
    return dict(_active_user)


def set_current_user(username_or_role: str):
    global _active_user
    key = username_or_role.strip().lower()
    if key in USERS:
        _active_user = USERS[key].copy()
        return _active_user
    for u in USERS.values():
        if u["role"].lower() == key:
            _active_user = u.copy()
            return _active_user
    return None


def has_permission(permission: str, user: dict = None) -> bool:
    u = user or _active_user
    role = u.get("role", ROLE_JAWAN)
    allowed = ROLE_PERMISSIONS.get(role, set())
    return permission in allowed
