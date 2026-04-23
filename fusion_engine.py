"""
fusion_engine.py
----------------
Combines vision AI output (from Person 1) with vehicle sensor data
into a unified risk score and driver state classification.

The fusion uses a weighted scoring approach:
  - Vision features (EAR, MAR, head pose, CNN state) → vision_score
  - Sensor features (speed, hard braking, lane departure)  → sensor_score
  - Fusion = weighted combination → final risk score ∈ [0, 1]

The output of this module feeds directly into the RiskClassifier.

Usage:
    engine = FusionEngine()
    result = engine.fuse(vision_data, sensor_reading)
    print(result.risk_score, result.driver_state)
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from sensor_module import SensorReading

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data contracts                                                       #
# ------------------------------------------------------------------ #

@dataclass
class VisionFeatures:
    """
    Expected output from Person 1's vision pipeline.
    All fields have safe defaults so the fusion engine works
    even before Person 1's code is integrated.

    EAR  (Eye Aspect Ratio)  — lower = more closed  (threshold ~0.25)
    MAR  (Mouth Aspect Ratio) — higher = yawning     (threshold ~0.65)
    """
    ear:             float = 0.30   # Eye Aspect Ratio        (0 = fully closed, ~0.3 = open)
    mar:             float = 0.20   # Mouth Aspect Ratio      (0 = closed, >0.6 = yawn)
    head_pitch:      float = 0.0    # degrees, + = looking down
    head_yaw:        float = 0.0    # degrees, + = looking right
    head_roll:       float = 0.0    # degrees
    cnn_state:       str   = "alert"     # "alert" | "drowsy" | "distracted"
    cnn_confidence:  float = 1.0    # model confidence [0, 1]
    blink_rate_pm:   float = 15.0   # blinks per minute (normal ~15)
    microsleep:      bool  = False  # eyes closed > 1.5 s
    yawn_detected:   bool  = False
    face_visible:    bool  = True
    timestamp:       float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionFeatures":
        """Safely build from a dictionary (e.g. from a shared queue/pipe)."""
        safe = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**safe)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class FusionResult:
    """Unified output of the fusion engine."""
    timestamp:        float = 0.0

    # Intermediate scores
    vision_score:     float = 0.0    # [0, 1]
    sensor_score:     float = 0.0    # [0, 1]
    risk_score:       float = 0.0    # [0, 1]  final fused score

    # Discrete driver state
    driver_state:     str   = "alert"   # "alert" | "drowsy" | "distracted" | "unknown"

    # Individual component contributions (for explainability)
    contributions:    Dict[str, float] = field(default_factory=dict)

    # Passthrough of inputs
    vision:           Optional[VisionFeatures] = None
    sensor:           Optional[SensorReading]  = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k not in ("vision", "sensor")}
        if self.vision:
            d["vision"] = self.vision.to_dict()
        if self.sensor:
            d["sensor"] = self.sensor.to_dict()
        return d


# ------------------------------------------------------------------ #
#  Fusion engine                                                        #
# ------------------------------------------------------------------ #

class FusionEngine:
    """
    Rule-based + weighted fusion of vision and sensor streams.

    Weights (tunable):
      Vision stream contributes 70% of the final score
      (face/eye data is the primary indicator of driver state).
      Sensor stream contributes 30%
      (vehicle behaviour is a secondary corroborating signal).

    Scoring logic per feature:
      Each feature is mapped to a sub-score ∈ [0, 1] using a
      piecewise linear function, then multiplied by its weight.
      Sub-scores are summed and normalised.
    """

    # Feature weights  — must sum to 1.0 within each stream
    VISION_WEIGHTS = {
        "ear_score":        0.30,   # eye openness
        "mar_score":        0.15,   # yawning
        "head_pose_score":  0.20,   # looking away
        "cnn_score":        0.25,   # CNN classifier
        "blink_score":      0.10,   # blink rate
    }

    SENSOR_WEIGHTS = {
        "lane_score":    0.40,   # lane departure
        "brake_score":   0.30,   # hard braking
        "speed_score":   0.15,   # erratic speed
        "swerve_score":  0.15,   # sudden swerve
    }

    # Stream blending
    VISION_WEIGHT = 0.70
    SENSOR_WEIGHT = 0.30

    # EAR thresholds
    EAR_OPEN    = 0.30   # fully alert
    EAR_DROWSY  = 0.20   # drowsy boundary
    EAR_CLOSED  = 0.15   # eyes closed / microsleep

    # MAR thresholds
    MAR_CLOSED  = 0.30
    MAR_YAWN    = 0.65

    # Head pose thresholds (degrees)
    HEAD_PITCH_SAFE     = 10
    HEAD_PITCH_DANGER   = 30
    HEAD_YAW_SAFE       = 15
    HEAD_YAW_DANGER     = 40

    def __init__(
        self,
        vision_weight: float = 0.70,
        sensor_weight: float = 0.30,
    ):
        assert abs(vision_weight + sensor_weight - 1.0) < 1e-6, \
            "vision_weight + sensor_weight must equal 1.0"
        self.VISION_WEIGHT = vision_weight
        self.SENSOR_WEIGHT = sensor_weight

        # Temporal smoothing (exponential moving average)
        self._ema_alpha   = 0.4
        self._ema_risk    = 0.0

        logger.info(
            f"FusionEngine ready  "
            f"vision={vision_weight:.0%}  sensor={sensor_weight:.0%}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        vision: VisionFeatures,
        sensor: SensorReading,
    ) -> FusionResult:
        """
        Compute the fused risk score and driver state.

        Parameters
        ----------
        vision : VisionFeatures  —  output from Person 1's vision pipeline
        sensor : SensorReading   —  output from SensorModule.get_latest()

        Returns
        -------
        FusionResult with risk_score ∈ [0, 1] and a driver_state string.
        """

        # --- Vision sub-scores ---
        ear_s  = self._score_ear(vision.ear, vision.microsleep)
        mar_s  = self._score_mar(vision.mar, vision.yawn_detected)
        pose_s = self._score_head_pose(vision.head_pitch, vision.head_yaw)
        cnn_s  = self._score_cnn(vision.cnn_state, vision.cnn_confidence)
        blink_s = self._score_blink_rate(vision.blink_rate_pm)

        vision_score = (
            self.VISION_WEIGHTS["ear_score"]       * ear_s +
            self.VISION_WEIGHTS["mar_score"]       * mar_s +
            self.VISION_WEIGHTS["head_pose_score"] * pose_s +
            self.VISION_WEIGHTS["cnn_score"]       * cnn_s +
            self.VISION_WEIGHTS["blink_score"]     * blink_s
        )

        # If face not visible — treat as high risk
        if not vision.face_visible:
            vision_score = max(vision_score, 0.8)

        # --- Sensor sub-scores ---
        lane_s   = 1.0 if sensor.lane_departure else 0.0
        brake_s  = 1.0 if sensor.hard_brake     else 0.0
        swerve_s = 1.0 if sensor.sudden_swerve  else 0.0
        speed_s  = self._score_speed_variance(sensor.speed_kmh, sensor.acceleration)

        sensor_score = (
            self.SENSOR_WEIGHTS["lane_score"]   * lane_s   +
            self.SENSOR_WEIGHTS["brake_score"]  * brake_s  +
            self.SENSOR_WEIGHTS["speed_score"]  * speed_s  +
            self.SENSOR_WEIGHTS["swerve_score"] * swerve_s
        )

        # --- Fuse ---
        raw_risk = (
            self.VISION_WEIGHT * vision_score +
            self.SENSOR_WEIGHT * sensor_score
        )
        raw_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # Exponential moving average smoothing
        self._ema_risk = (
            self._ema_alpha * raw_risk +
            (1 - self._ema_alpha) * self._ema_risk
        )
        smoothed_risk = float(np.clip(self._ema_risk, 0.0, 1.0))

        # --- Driver state ---
        driver_state = self._classify_state(vision, sensor, smoothed_risk)

        result = FusionResult(
            timestamp     = time.time(),
            vision_score  = round(vision_score,  3),
            sensor_score  = round(sensor_score,  3),
            risk_score    = round(smoothed_risk, 3),
            driver_state  = driver_state,
            vision        = vision,
            sensor        = sensor,
            contributions = {
                "ear":    round(ear_s,   3),
                "mar":    round(mar_s,   3),
                "pose":   round(pose_s,  3),
                "cnn":    round(cnn_s,   3),
                "blink":  round(blink_s, 3),
                "lane":   round(lane_s,  3),
                "brake":  round(brake_s, 3),
                "swerve": round(swerve_s,3),
                "speed":  round(speed_s, 3),
            },
        )

        logger.debug(
            f"Fused | vision={vision_score:.3f}  sensor={sensor_score:.3f}  "
            f"risk={smoothed_risk:.3f}  state={driver_state}"
        )
        return result

    # ------------------------------------------------------------------ #
    #  Individual scoring functions                                        #
    # ------------------------------------------------------------------ #

    def _score_ear(self, ear: float, microsleep: bool) -> float:
        """Lower EAR = more drowsy."""
        if microsleep:
            return 1.0
        if ear >= self.EAR_OPEN:
            return 0.0
        if ear <= self.EAR_CLOSED:
            return 1.0
        # Linear interpolation
        return 1.0 - (ear - self.EAR_CLOSED) / (self.EAR_OPEN - self.EAR_CLOSED)

    def _score_mar(self, mar: float, yawn: bool) -> float:
        """Higher MAR = yawning."""
        if yawn:
            return 0.8
        if mar <= self.MAR_CLOSED:
            return 0.0
        if mar >= self.MAR_YAWN:
            return 1.0
        return (mar - self.MAR_CLOSED) / (self.MAR_YAWN - self.MAR_CLOSED)

    def _score_head_pose(self, pitch: float, yaw: float) -> float:
        """Large head angle = looking away from road."""
        pitch_abs = abs(pitch)
        yaw_abs   = abs(yaw)

        pitch_score = 0.0
        if pitch_abs > self.HEAD_PITCH_DANGER:
            pitch_score = 1.0
        elif pitch_abs > self.HEAD_PITCH_SAFE:
            pitch_score = (pitch_abs - self.HEAD_PITCH_SAFE) / (
                self.HEAD_PITCH_DANGER - self.HEAD_PITCH_SAFE
            )

        yaw_score = 0.0
        if yaw_abs > self.HEAD_YAW_DANGER:
            yaw_score = 1.0
        elif yaw_abs > self.HEAD_YAW_SAFE:
            yaw_score = (yaw_abs - self.HEAD_YAW_SAFE) / (
                self.HEAD_YAW_DANGER - self.HEAD_YAW_SAFE
            )

        return max(pitch_score, yaw_score)

    def _score_cnn(self, state: str, confidence: float) -> float:
        """CNN classifier output → risk sub-score."""
        base = {
            "alert"      : 0.0,
            "drowsy"     : 0.8,
            "distracted" : 0.6,
            "unknown"    : 0.4,
        }.get(state.lower(), 0.4)
        return base * confidence

    def _score_blink_rate(self, rate_pm: float) -> float:
        """
        Normal blink rate: 10-20/min.
        Very low rate (eyes barely closing) → possible stare or microsleep.
        Very high rate → eye fatigue.
        """
        if 10 <= rate_pm <= 20:
            return 0.0
        if rate_pm < 5 or rate_pm > 30:
            return 1.0
        if rate_pm < 10:
            return (10 - rate_pm) / 10
        return (rate_pm - 20) / 20

    def _score_speed_variance(self, speed: float, accel: float) -> float:
        """Erratic acceleration / deceleration indicates poor control."""
        return float(np.clip(abs(accel) / 5.0, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  State classification                                                #
    # ------------------------------------------------------------------ #

    def _classify_state(
        self,
        vision: VisionFeatures,
        sensor: SensorReading,
        risk: float,
    ) -> str:
        """
        Determine a human-readable driver state from features + risk score.
        Priority: microsleep > CNN-classified drowsy > CNN-classified distracted
        > head pose > risk threshold.
        """
        if vision.microsleep:
            return "drowsy"
        if vision.cnn_state == "drowsy" and vision.cnn_confidence > 0.6:
            return "drowsy"
        if vision.cnn_state == "distracted" and vision.cnn_confidence > 0.6:
            return "distracted"
        if abs(vision.head_yaw) > self.HEAD_YAW_DANGER:
            return "distracted"
        if not vision.face_visible:
            return "distracted"
        if risk > 0.65:
            return "drowsy"
        if risk > 0.40:
            return "distracted"
        return "alert"


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    engine = FusionEngine()
    sensor = SensorReading(
        speed_kmh=80, acceleration=-0.5, lane_deviation=5,
        brake_pressure=0, hard_brake=False, sudden_swerve=False,
    )

    scenarios = [
        ("Alert driver",      VisionFeatures(ear=0.32, mar=0.2,  cnn_state="alert",      cnn_confidence=0.95)),
        ("Mild drowsiness",   VisionFeatures(ear=0.22, mar=0.4,  cnn_state="drowsy",     cnn_confidence=0.70, blink_rate_pm=6)),
        ("Severe drowsiness", VisionFeatures(ear=0.15, mar=0.7,  cnn_state="drowsy",     cnn_confidence=0.92, microsleep=True)),
        ("Distracted",        VisionFeatures(ear=0.29, mar=0.2,  cnn_state="distracted", cnn_confidence=0.85, head_yaw=45)),
        ("Yawning",           VisionFeatures(ear=0.28, mar=0.72, yawn_detected=True,      cnn_state="drowsy")),
    ]

    print(f"\n{'Scenario':<25} {'Vision':>8} {'Sensor':>8} {'Risk':>8} {'State':<15}")
    print("-" * 70)
    for name, vis in scenarios:
        r = engine.fuse(vis, sensor)
        print(f"{name:<25} {r.vision_score:>8.3f} {r.sensor_score:>8.3f} {r.risk_score:>8.3f}  {r.driver_state}")
