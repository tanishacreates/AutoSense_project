"""
risk_classifier.py
------------------
Scikit-learn ML pipeline that classifies driver risk into three levels:
  0 = SAFE      (risk_score < 0.35)
  1 = WARNING   (risk_score 0.35–0.65)
  2 = ALERT     (risk_score > 0.65)

The classifier is trained on a synthetic dataset that mirrors realistic
feature distributions from published drowsiness detection research
(NTHU-DDD / YawDD / CEW datasets as reference).

When real session data is available, retrain with:
    clf = RiskClassifier()
    clf.train(features_array, labels_array)
    clf.save("models/risk_classifier.joblib")
"""

import time
import logging
import numpy as np
import pandas as pd
import joblib
import sklearn
import os
from pathlib import Path
from typing import Optional, Tuple, List

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

from alert_system import AlertLevel
from fusion_engine import FusionResult

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "models" / "risk_classifier.joblib"

# FIX: removed risk_score, vision_score, sensor_score from feature vector.
# Those three columns are derived directly from the labels — including them
# causes the model to ignore all the meaningful EAR/MAR/head-pose signals
# and just memorise the derived scores, giving fake 1.0 accuracy.
FEATURE_NAMES = [
    "ear", "mar", "head_pitch", "head_yaw", "head_roll",
    "cnn_drowsy", "cnn_distracted", "cnn_confidence",
    "blink_rate_pm", "microsleep", "yawn_detected",
    "speed_kmh", "acceleration", "lateral_accel",
    "brake_pressure", "lane_deviation", "hard_brake",
    "sudden_swerve", "throttle_pct",
]


def extract_features(result: FusionResult) -> np.ndarray:
    """
    Convert a FusionResult into a flat numpy feature vector.
    Single source of truth for both training and inference.
    """
    v = result.vision
    s = result.sensor

    features = [
        v.ear              if v else 0.30,
        v.mar              if v else 0.20,
        v.head_pitch       if v else 0.0,
        v.head_yaw         if v else 0.0,
        v.head_roll        if v else 0.0,
        int(v.cnn_state == "drowsy")     if v else 0,
        int(v.cnn_state == "distracted") if v else 0,
        v.cnn_confidence   if v else 1.0,
        v.blink_rate_pm    if v else 15.0,
        int(v.microsleep)  if v else 0,
        int(v.yawn_detected) if v else 0,

        s.speed_kmh        if s else 60.0,
        s.acceleration     if s else 0.0,
        s.lateral_accel    if s else 0.0,
        s.brake_pressure   if s else 0.0,
        s.lane_deviation   if s else 0.0,
        int(s.hard_brake)  if s else 0,
        int(s.sudden_swerve) if s else 0,
        s.throttle_pct     if s else 30.0,
    ]
    return np.array(features, dtype=np.float32)


# ------------------------------------------------------------------ #
#  Synthetic dataset generator                                          #
# ------------------------------------------------------------------ #

def generate_synthetic_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a labelled synthetic dataset that mirrors the feature
    distributions described in NTHU-DDD and YawDD drowsiness research.

    FIX: Added class boundary overlap and label noise so the model
    learns real decision boundaries instead of memorising perfectly
    separated synthetic clusters (which caused fake 1.0 accuracy).
    """
    rng = np.random.default_rng(seed)

    rows   = []
    labels = []

    samples_per_class = n_samples // 3

    # --- Class 0: SAFE ---
    for _ in range(samples_per_class):
        ear   = rng.normal(0.30, 0.03)
        mar   = rng.normal(0.25, 0.05)
        yaw   = rng.normal(0, 8)
        pitch = rng.normal(0, 6)
        rows.append({
            "ear": ear, "mar": mar,
            "head_pitch": pitch, "head_yaw": yaw, "head_roll": rng.normal(0, 3),
            "cnn_drowsy": 0, "cnn_distracted": 0,
            "cnn_confidence": rng.uniform(0.8, 1.0),
            "blink_rate_pm": rng.normal(15, 3),
            "microsleep": 0, "yawn_detected": 0,
            "speed_kmh": rng.normal(65, 15), "acceleration": rng.normal(0, 0.5),
            "lateral_accel": rng.normal(0, 0.3),
            "brake_pressure": rng.uniform(0, 10), "lane_deviation": rng.normal(0, 5),
            "hard_brake": 0, "sudden_swerve": 0,
            "throttle_pct": rng.uniform(20, 50),
        })
        labels.append(0)

    # --- Class 1: WARNING ---
    for _ in range(samples_per_class):
        ear   = rng.normal(0.23, 0.03)
        mar   = rng.normal(0.45, 0.10)
        yaw   = rng.normal(0, 20)
        pitch = rng.normal(-5, 8)
        cnn_d = int(rng.random() > 0.4)
        rows.append({
            "ear": ear, "mar": mar,
            "head_pitch": pitch, "head_yaw": yaw, "head_roll": rng.normal(0, 6),
            "cnn_drowsy": cnn_d, "cnn_distracted": int(not cnn_d),
            "cnn_confidence": rng.uniform(0.55, 0.80),
            "blink_rate_pm": rng.normal(9, 4),
            "microsleep": 0, "yawn_detected": int(mar > 0.55),
            "speed_kmh": rng.normal(70, 20), "acceleration": rng.normal(0, 1.2),
            "lateral_accel": rng.uniform(0, 1.5),
            "brake_pressure": rng.uniform(0, 30), "lane_deviation": rng.normal(0, 15),
            "hard_brake": 0, "sudden_swerve": int(rng.random() > 0.85),
            "throttle_pct": rng.uniform(20, 60),
        })
        labels.append(1)

    # --- Class 2: ALERT ---
    for _ in range(n_samples - 2 * samples_per_class):
        ear   = rng.normal(0.17, 0.03)
        mar   = rng.normal(0.65, 0.10)
        yaw   = rng.normal(0, 35)
        pitch = rng.normal(-10, 10)
        rows.append({
            "ear": max(0.05, ear), "mar": min(1.0, mar),
            "head_pitch": pitch, "head_yaw": yaw, "head_roll": rng.normal(0, 10),
            "cnn_drowsy": int(rng.random() > 0.25),
            "cnn_distracted": int(rng.random() > 0.50),
            "cnn_confidence": rng.uniform(0.70, 1.0),
            "blink_rate_pm": rng.normal(5, 3),
            "microsleep": int(rng.random() > 0.40),
            "yawn_detected": int(mar > 0.55),
            "speed_kmh": rng.normal(75, 25), "acceleration": rng.normal(0, 2.5),
            "lateral_accel": rng.uniform(1.5, 4.0),
            "brake_pressure": rng.uniform(40, 100),
            "lane_deviation": rng.normal(0, 35),
            "hard_brake": int(rng.random() > 0.50),
            "sudden_swerve": int(rng.random() > 0.40),
            "throttle_pct": rng.uniform(0, 80),
        })
        labels.append(2)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    X  = df.values.astype(np.float32)
    y  = np.array(labels, dtype=np.int32)

    # FIX: add 5% label noise to simulate real-world ambiguity between classes.
    # Without this the classes are perfectly separable and the model gets fake 1.0 accuracy.
    noise_mask = rng.random(len(y)) < 0.05
    y[noise_mask] = (y[noise_mask] + rng.integers(1, 3, size=noise_mask.sum())) % 3

    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ------------------------------------------------------------------ #
#  Classifier                                                           #
# ------------------------------------------------------------------ #

class RiskClassifier:
    """
    Gradient Boosting classifier wrapped in a scikit-learn Pipeline
    (StandardScaler → GradientBoostingClassifier).
    """

    LABEL_MAP = {
        0: AlertLevel.SAFE,
        1: AlertLevel.WARNING,
        2: AlertLevel.ALERT,
    }

    LABEL_NAMES = {0: "SAFE", 1: "WARNING", 2: "ALERT"}

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._trained   = False
        self._metrics: dict = {}
        self._saved_versions: dict = {}

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def train_synthetic(
        self,
        n_samples: int = 5000,
        test_size: float = 0.20,
        verbose: bool = True,
    ) -> dict:
        logger.info(f"Generating synthetic dataset  n={n_samples}")
        X, y = generate_synthetic_dataset(n_samples)
        return self.train(X, y, test_size=test_size, verbose=verbose)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.20,
        verbose: bool = True,
    ) -> dict:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators     = 200,
                learning_rate    = 0.08,
                max_depth        = 4,
                min_samples_leaf = 10,
                subsample        = 0.8,
                random_state     = 42,
            )),
        ])

        logger.info("Training GradientBoostingClassifier ...")
        t0 = time.time()
        self._pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        self._trained = True

        y_pred = self._pipeline.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=["SAFE", "WARNING", "ALERT"]
        )

        cv_scores = cross_val_score(
            self._pipeline, X, y, cv=5, scoring="f1_weighted"
        )

        self._metrics = {
            "accuracy":    round(acc, 4),
            "f1_weighted": round(f1, 4),
            "cv_mean":     round(cv_scores.mean(), 4),
            "cv_std":      round(cv_scores.std(), 4),
            "confusion_matrix": cm.tolist(),
            "train_time_s": round(elapsed, 2),
            "n_train":     len(X_train),
            "n_test":      len(X_test),
        }

        if verbose:
            print("\n" + "="*50)
            print("  RISK CLASSIFIER — TRAINING RESULTS")
            print("="*50)
            print(f"  Accuracy     : {acc:.4f}")
            print(f"  F1 (weighted): {f1:.4f}")
            print(f"  CV F1        : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Training time: {elapsed:.2f}s")
            print(f"\n{report}")
            print("  Confusion matrix (rows=true, cols=pred):")
            print("  " + str(np.array(cm)))
            print("="*50 + "\n")

        return self._metrics

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, result: FusionResult) -> AlertLevel:
        if self._trained and self._pipeline:
            X     = extract_features(result).reshape(1, -1)
            label = int(self._pipeline.predict(X)[0])
            return self.LABEL_MAP[label]

        r = result.risk_score
        if r > 0.65:
            return AlertLevel.ALERT
        if r > 0.35:
            return AlertLevel.WARNING
        return AlertLevel.SAFE

    def predict_proba(self, result: FusionResult) -> dict:
        if self._trained and self._pipeline:
            X     = extract_features(result).reshape(1, -1)
            probs = self._pipeline.predict_proba(X)[0]
            return {
                "SAFE":    round(float(probs[0]), 4),
                "WARNING": round(float(probs[1]), 4),
                "ALERT":   round(float(probs[2]), 4),
            }

        r = result.risk_score
        return {
            "SAFE":    round(max(0, 1 - r * 2), 4),
            "WARNING": round(max(0, 1 - abs(r - 0.5) * 4), 4),
            "ALERT":   round(max(0, r * 2 - 1), 4),
        }

    def feature_importance(self, top_n: int = 10) -> List[Tuple[str, float]]:
        if not self._trained:
            return []
        clf_step    = self._pipeline.named_steps["clf"]
        importances = clf_step.feature_importances_
        pairs = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1], reverse=True
        )
        return [(name, round(imp, 4)) for name, imp in pairs[:top_n]]

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: Optional[str] = None):
        """Serialize the trained pipeline + version metadata to disk."""
        p = Path(path) if path else MODEL_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline":       self._pipeline,
            "metrics":        self._metrics,
            # FIX: store versions so is_compatible() can check cheaply
            # without needing to run a dummy predict() that might crash
            "sklearn_version": sklearn.__version__,
            "numpy_version":   np.__version__,
            "n_features":      len(FEATURE_NAMES),
        }, p)
        logger.info(f"Model saved to {p}")

    def load(self, path: Optional[str] = None) -> bool:
        """Load a trained pipeline from disk. Returns True on success."""
        p = Path(path) if path else MODEL_PATH
        if not p.exists():
            logger.warning(f"Model file not found: {p}")
            return False
        try:
            obj = joblib.load(p)
            self._pipeline       = obj["pipeline"]
            self._metrics        = obj.get("metrics", {})
            self._saved_versions = {
                "sklearn_version": obj.get("sklearn_version"),
                "numpy_version":   obj.get("numpy_version"),
                "n_features":      obj.get("n_features"),
            }
            self._trained = True
            logger.info(f"Model loaded from {p}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            return False

    def is_compatible(self) -> bool:
        """
        FIX: cheap version-based compatibility check.
        Replaces the old dummy-predict approach which crashed on any API
        change and triggered a full 7.5s retrain on the main thread.
        """
        if not self._trained or not self._saved_versions:
            return False

        current = {
            "sklearn_version": sklearn.__version__,
            "numpy_version":   np.__version__,
            "n_features":      len(FEATURE_NAMES),
        }

        for key, val in current.items():
            saved = self._saved_versions.get(key)
            if saved is None:
                # Old model saved before version tracking — treat as incompatible
                logger.warning(f"Saved model missing {key} — treating as incompatible")
                return False
            if key == "n_features":
                if saved != val:
                    logger.warning(f"Feature count changed: saved={saved}, current={val}")
                    return False
            else:
                # Compare major.minor only — patch updates are fine
                saved_mm   = ".".join(str(saved).split(".")[:2])
                current_mm = ".".join(str(val).split(".")[:2])
                if saved_mm != current_mm:
                    logger.warning(f"{key} mismatch: saved={saved}, current={val}")
                    return False

        return True

    def get_metrics(self) -> dict:
        return self._metrics.copy()


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    clf     = RiskClassifier()
    metrics = clf.train_synthetic(n_samples=6000)
    clf.save()

    print("\nTop feature importances:")
    for name, imp in clf.feature_importance():
        bar = "█" * int(imp * 60)
        print(f"  {name:<20} {imp:.4f}  {bar}")

    from fusion_engine import FusionResult, VisionFeatures
    from sensor_module import SensorReading

    test_cases = [
        ("Alert driver",   FusionResult(risk_score=0.10, vision_score=0.05, sensor_score=0.02,
                                        driver_state="alert",
                                        vision=VisionFeatures(ear=0.32, cnn_state="alert"),
                                        sensor=SensorReading(speed_kmh=70))),
        ("Warning (mild)", FusionResult(risk_score=0.45, vision_score=0.40, sensor_score=0.30,
                                        driver_state="drowsy",
                                        vision=VisionFeatures(ear=0.22, cnn_state="drowsy", cnn_confidence=0.65),
                                        sensor=SensorReading(speed_kmh=75))),
        ("Alert (severe)", FusionResult(risk_score=0.80, vision_score=0.85, sensor_score=0.60,
                                        driver_state="drowsy",
                                        vision=VisionFeatures(ear=0.14, microsleep=True, cnn_state="drowsy"),
                                        sensor=SensorReading(speed_kmh=80, hard_brake=True))),
    ]

    print("\nInference tests:")
    for name, result in test_cases:
        level = clf.predict(result)
        proba = clf.predict_proba(result)
        print(f"  {name:<20} → {level.name:<10}  proba={proba}")
