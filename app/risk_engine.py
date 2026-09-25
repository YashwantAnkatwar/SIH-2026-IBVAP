"""
risk_engine.py

Transparent, rule-based risk scoring. Every decision here can be traced
back to a simple, explicit condition (dwell time, zone type, object
class, or number of simultaneous objects) rather than an opaque score -
this is intentional: for a surveillance/security prototype, an analyst
needs to be able to say *why* an alert fired.

Levels: LOW < MEDIUM < HIGH < CRITICAL
"""

from collections import defaultdict


class RiskLevel:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    _ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3}

    @classmethod
    def max(cls, a, b):
        return a if cls._ORDER[a] >= cls._ORDER[b] else b


# Vehicle classes vs. pedestrian - used for zone-appropriateness rules.
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class RiskEngine:
    """
    Evaluates risk for a single tracked object's current zone situation.

    Rules (in order of increasing severity), all configurable via the
    constructor:

    1. Being in a "restricted" zone at all         -> MEDIUM
    2. Dwelling in a restricted zone past a         -> HIGH
       short threshold (default 5s)
    3. Dwelling in a restricted zone past a         -> CRITICAL
       long threshold (default 15s)
    4. A vehicle class present in a zone whose      -> HIGH (or CRITICAL if
       allowed_classes excludes it (e.g. a             also restricted)
       "pedestrian_only" zone)
    5. Group incursion: >= group_threshold distinct -> escalates whatever
       tracks simultaneously inside the SAME           level was computed
       restricted zone                                 by one step
    """

    def __init__(
        self,
        dwell_high_seconds=5.0,
        dwell_critical_seconds=15.0,
        group_threshold=3,
    ):
        self.dwell_high_seconds = dwell_high_seconds
        self.dwell_critical_seconds = dwell_critical_seconds
        self.group_threshold = group_threshold

        # zone_name -> set of track_ids currently inside, refreshed every
        # time evaluate_frame() is called (see camera_worker.py).
        self._zone_occupancy = defaultdict(set)

    def begin_frame(self):
        """Call once per frame before evaluating each track's zone events."""
        self._zone_occupancy.clear()

    def evaluate_track(self, track_id, class_name, zone_events):
        """
        Given the ZONE_ENTER/ZONE_DWELL/ZONE_EXIT events for one track this
        frame (from ZoneManager.update), return (risk_level, reasons: list[str]).

        Only ZONE_ENTER and ZONE_DWELL events are risk-relevant; ZONE_EXIT
        simply means the object left, which is not itself a risk (though it
        is still a loggable event elsewhere).
        """
        level = RiskLevel.LOW
        reasons = []

        for ev in zone_events:
            if ev["type"] not in ("ZONE_ENTER", "ZONE_DWELL"):
                continue

            zone_name = ev["zone"]
            zone_type = ev["zone_type"]
            dwell = ev["dwell_seconds"]

            if zone_type == "restricted":
                self._zone_occupancy[zone_name].add(track_id)

                zone_level = RiskLevel.MEDIUM
                reason = f"Present in restricted zone '{zone_name}'"

                if dwell >= self.dwell_critical_seconds:
                    zone_level = RiskLevel.CRITICAL
                    reason = (
                        f"Loitering in restricted zone '{zone_name}' "
                        f"for {dwell:.0f}s (>= {self.dwell_critical_seconds:.0f}s)"
                    )
                elif dwell >= self.dwell_high_seconds:
                    zone_level = RiskLevel.HIGH
                    reason = (
                        f"Extended dwell in restricted zone '{zone_name}' "
                        f"for {dwell:.0f}s (>= {self.dwell_high_seconds:.0f}s)"
                    )

                level = RiskLevel.max(level, zone_level)
                reasons.append(reason)

            if ev["violation"]:
                # e.g. a vehicle inside a pedestrian-only zone
                zone_level = (
                    RiskLevel.CRITICAL if zone_type == "restricted" else RiskLevel.HIGH
                )
                level = RiskLevel.max(level, zone_level)
                reasons.append(
                    f"'{class_name}' not permitted in zone '{zone_name}' ({zone_type})"
                )

        return level, reasons

    def apply_group_escalation(self, level, reasons):
        """
        After all tracks in a frame have been evaluated, call this per-track
        to escalate risk if the zone(s) it's in also contain a group of
        other tracks at the same time (e.g. multiple people crossing a
        restricted boundary together).
        """
        for zone_name, occupants in self._zone_occupancy.items():
            if len(occupants) >= self.group_threshold:
                level = self._escalate(level)
                reasons.append(
                    f"Group incursion: {len(occupants)} objects simultaneously "
                    f"in restricted zone '{zone_name}' (>= {self.group_threshold})"
                )
        return level, reasons

    @staticmethod
    def _escalate(level):
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        idx = order.index(level)
        return order[min(idx + 1, len(order) - 1)]

    # ------------------------------------------------------------------
    # Explainable 0-100 score (Phase 6)
    # ------------------------------------------------------------------

    # Base score for the middle of each level's band (see config.RISK_SCORE_BANDS).
    _LEVEL_BASE_SCORE = {
        RiskLevel.LOW: 10,
        RiskLevel.MEDIUM: 32,
        RiskLevel.HIGH: 58,
        RiskLevel.CRITICAL: 85,
    }
    # Each additional contributing reason nudges the score up within the
    # level's band (capped so it never crosses into the next band), which
    # is what lets an operator sort/compare two alerts at the same level.
    _POINTS_PER_EXTRA_REASON = 4
    _MAX_EXTRA_REASONS_COUNTED = 3

    @classmethod
    def score_for(cls, level, reasons):
        """
        Deterministic 0-100 score derived from the rule-based level plus
        how many distinct reasons contributed. This is NOT a separate
        machine-learned estimate; it is a transparent re-expression of
        the same rules RiskEngine already applied, purely so alerts are
        sortable/comparable on the dashboard.

        Returns (score:int, band:str) where band matches
        config.RISK_SCORE_BANDS.
        """
        base = cls._LEVEL_BASE_SCORE.get(level, 0)
        extra_reasons = max(0, len(reasons) - 1)
        extra_reasons = min(extra_reasons, cls._MAX_EXTRA_REASONS_COUNTED)
        score = base + extra_reasons * cls._POINTS_PER_EXTRA_REASON

        # Clamp to the level's own band so a HIGH with many reasons never
        # numerically outranks a plain CRITICAL.
        band_bounds = {
            RiskLevel.LOW: (0, 24),
            RiskLevel.MEDIUM: (25, 49),
            RiskLevel.HIGH: (50, 74),
            RiskLevel.CRITICAL: (75, 100),
        }
        lo, hi = band_bounds.get(level, (0, 100))
        score = max(lo, min(hi, score))
        return score, level

    @staticmethod
    def severity_for_score(score: int) -> str:
        """
        Maps a 0-100 score to the transparent rule-based category:
        LOW: 0-24, MEDIUM: 25-49, HIGH: 50-74, CRITICAL: 75-100.
        """
        if score <= 24:
            return RiskLevel.LOW
        elif score <= 49:
            return RiskLevel.MEDIUM
        elif score <= 74:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    @classmethod
    def calculate_composite_risk(
        cls,
        factors: dict = None,
        camera_id: str = "",
        track_id: str = "",
        timestamp: float = None,
    ) -> dict:
        """
        Transparent, explainable 0-100 rule-based score derived from
        measurable factors for border surveillance (Phase 4).

        Categories:
        - LOW: 0-24
        - MEDIUM: 25-49
        - HIGH: 50-74
        - CRITICAL: 75-100
        """
        import time
        FACTOR_WEIGHTS = {
            "restricted_zone_entry": 30,
            "dwell_warning": 20,
            "dwell_critical": 35,
            "night_movement": 25,
            "unknown_face": 15,
            "sudden_movement": 20,
            "running": 20,
            "line_crossing": 20,
            "group_incursion": 25,
            "camera_tamper": 50,
        }
        ALIASES = {
            "zone_entry_restricted": "restricted_zone_entry",
            "loitering_warning": "dwell_warning",
            "loitering_critical": "dwell_critical",
            "sudden_acceleration": "sudden_movement",
            "group_incursion": "group_incursion",
            "camera_tamper": "camera_tamper",
            "night_movement": "night_movement",
        }

        # Normalize factors if passed as list of strings or dicts
        norm_factors = {}
        if isinstance(factors, list):
            for item in factors:
                if isinstance(item, dict):
                    fname = item.get("factor") or item.get("name") or ""
                    norm_factors[fname] = True
                elif isinstance(item, str):
                    norm_factors[item] = True
        elif isinstance(factors, dict):
            norm_factors = dict(factors)

        score = 0
        contributing_factors = []
        factor_values = {}
        factor_breakdown = []

        for raw_name, active in norm_factors.items():
            if not active:
                continue
            key = str(raw_name).strip().lower()
            key = ALIASES.get(key, key)
            if key in FACTOR_WEIGHTS:
                weight = FACTOR_WEIGHTS[key]
                score += weight
                contributing_factors.append(key)
                factor_values[key] = weight
                factor_breakdown.append({
                    "factor": key,
                    "points": weight,
                    "raw_factor": raw_name,
                })

        if "explicit_score" in norm_factors:
            score = int(norm_factors["explicit_score"])
            contributing_factors = ["explicit_score"]
            factor_values = {"explicit_score": score}
            factor_breakdown = [{"factor": "explicit_score", "points": score}]

        score = max(0, min(100, score))
        severity = cls.severity_for_score(score)

        return {
            "total_score": score,
            "severity": severity,
            "contributing_factors": contributing_factors,
            "factor_values": factor_values,
            "factor_breakdown": factor_breakdown,
            "timestamp": timestamp or time.time(),
            "camera_id": camera_id,
            "track_id": track_id,
        }

    @classmethod
    def find_spatial_clusters(cls, occupants_with_centers: list, cluster_radius=250.0):
        """
        Identifies spatial clusters of tracks inside a zone where each track
        is within cluster_radius of at least one other member in the cluster.
        Returns list of clusters: [ [track_id_1, track_id_2, ...], ... ]
        """
        if not occupants_with_centers:
            return []

        n = len(occupants_with_centers)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                curr_tid, (cx1, cy1) = occupants_with_centers[curr]
                cluster.append(curr_tid)

                for j in range(n):
                    if not visited[j]:
                        other_tid, (cx2, cy2) = occupants_with_centers[j]
                        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                        if dist <= cluster_radius:
                            visited[j] = True
                            queue.append(j)

            clusters.append(cluster)

        return clusters
