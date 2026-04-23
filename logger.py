"""
logger.py
---------
Event logger and performance evaluator for the driver assistance system.

Functions:
  - EventLogger   : writes timestamped CSV rows every prediction cycle
  - SessionEvaluator : reads a session CSV and computes metrics + plots

Usage:
    logger   = EventLogger("logs/session_001.csv")
    logger.log(fusion_result, alert_level)

    # After session:
    evaluator = SessionEvaluator("logs/session_001.csv")
    evaluator.summary()
    evaluator.plot_timeline()
    evaluator.plot_confusion()
"""

import csv
import time
import logging
import os
import threading
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  CSV columns                                                         #
# ------------------------------------------------------------------ #

CSV_COLUMNS = [
    "timestamp", "datetime",
    # Driver state
    "driver_state", "alert_level", "risk_score",
    "vision_score", "sensor_score",
    # Vision features
    "ear", "mar", "head_yaw", "head_pitch", "blink_rate_pm",
    "microsleep", "yawn_detected", "cnn_state", "cnn_confidence",
    # Sensor features
    "speed_kmh", "acceleration", "lateral_accel",
    "brake_pressure", "lane_deviation", "hard_brake", "sudden_swerve",
    # Contributions
    "contrib_ear", "contrib_mar", "contrib_pose", "contrib_cnn",
    "contrib_lane", "contrib_brake",
]


# ------------------------------------------------------------------ #
#  Event Logger                                                        #
# ------------------------------------------------------------------ #

class EventLogger:
    """
    Thread-safe CSV logger.  Opens the file once, writes one row per
    call to log().  Buffered writes are flushed every FLUSH_INTERVAL_S
    seconds to avoid hitting disk every prediction cycle.

    Parameters
    ----------
    path        : CSV file path (auto-named if None)
    flush_s     : how often to flush buffer to disk
    """

    FLUSH_INTERVAL_S = 5.0

    def __init__(self, path: Optional[str] = None, flush_s: float = 5.0):
        if path is None:
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = LOG_DIR / f"session_{ts}.csv"

        self.path      = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_s  = flush_s
        self._lock     = threading.Lock()
        self._file     = None
        self._writer   = None
        self._row_count = 0
        self._last_flush = time.time()

        self._open()
        logger.info(f"EventLogger opened: {self.path}")

    def _open(self):
        self._file   = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def log(self, fusion_result, alert_level) -> int:
        """
        Write one row.  Returns current row count.

        Parameters
        ----------
        fusion_result : FusionResult from fusion_engine.py
        alert_level   : AlertLevel enum from alert_system.py
        """
        from fusion_engine import FusionResult
        from alert_system  import AlertLevel

        v = fusion_result.vision
        s = fusion_result.sensor
        c = fusion_result.contributions

        row = {
            "timestamp":        fusion_result.timestamp,
            "datetime":         time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(fusion_result.timestamp)
            ),
            "driver_state":     fusion_result.driver_state,
            "alert_level":      alert_level.name if hasattr(alert_level, "name") else str(alert_level),
            "risk_score":       fusion_result.risk_score,
            "vision_score":     fusion_result.vision_score,
            "sensor_score":     fusion_result.sensor_score,

            "ear":              v.ear           if v else "",
            "mar":              v.mar           if v else "",
            "head_yaw":         v.head_yaw      if v else "",
            "head_pitch":       v.head_pitch    if v else "",
            "blink_rate_pm":    v.blink_rate_pm if v else "",
            "microsleep":       int(v.microsleep)     if v else "",
            "yawn_detected":    int(v.yawn_detected)  if v else "",
            "cnn_state":        v.cnn_state     if v else "",
            "cnn_confidence":   v.cnn_confidence if v else "",

            "speed_kmh":        s.speed_kmh      if s else "",
            "acceleration":     s.acceleration   if s else "",
            "lateral_accel":    s.lateral_accel  if s else "",
            "brake_pressure":   s.brake_pressure if s else "",
            "lane_deviation":   s.lane_deviation if s else "",
            "hard_brake":       int(s.hard_brake)     if s else "",
            "sudden_swerve":    int(s.sudden_swerve)  if s else "",

            "contrib_ear":      c.get("ear",   ""),
            "contrib_mar":      c.get("mar",   ""),
            "contrib_pose":     c.get("pose",  ""),
            "contrib_cnn":      c.get("cnn",   ""),
            "contrib_lane":     c.get("lane",  ""),
            "contrib_brake":    c.get("brake", ""),
        }

        with self._lock:
            self._writer.writerow(row)
            self._row_count += 1

            now = time.time()
            if now - self._last_flush > self._flush_s:
                self._file.flush()
                self._last_flush = now

        return self._row_count

    def close(self):
        """Flush and close the CSV file."""
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None
        logger.info(f"EventLogger closed: {self._row_count} rows written to {self.path}")

    def __del__(self):
        self.close()


# ------------------------------------------------------------------ #
#  Session Evaluator                                                   #
# ------------------------------------------------------------------ #

class SessionEvaluator:
    """
    Reads a session CSV file and computes summary statistics and plots.

    Usage:
        ev = SessionEvaluator("logs/session_001.csv")
        ev.summary()
        ev.plot_timeline()
        ev.save_report("logs/session_001_report.txt")
    """

    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Session file not found: {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} rows from {csv_path}")

    # ------------------------------------------------------------------ #
    #  Summary stats                                                       #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        df = self.df

        if df.empty:
            print("Empty session — no data to evaluate.")
            return {}

        duration_s = float(df["timestamp"].max() - df["timestamp"].min())
        fps_equiv  = len(df) / max(duration_s, 1)

        alert_counts = df["alert_level"].value_counts().to_dict()
        state_counts = df["driver_state"].value_counts().to_dict()

        risk_stats = {
            "mean":   round(df["risk_score"].mean(),   4),
            "max":    round(df["risk_score"].max(),    4),
            "min":    round(df["risk_score"].min(),    4),
            "std":    round(df["risk_score"].std(),    4),
            "p95":    round(df["risk_score"].quantile(0.95), 4),
        }

        # Time spent in each alert zone
        total   = len(df)
        pct_safe    = round(100 * (df["alert_level"] == "SAFE").sum()    / total, 1)
        pct_warning = round(100 * (df["alert_level"] == "WARNING").sum() / total, 1)
        pct_alert   = round(100 * (df["alert_level"].isin(["ALERT","CRITICAL"])).sum() / total, 1)

        results = {
            "duration_s":    round(duration_s, 1),
            "total_frames":  total,
            "fps_equivalent": round(fps_equiv, 1),
            "alert_counts":  alert_counts,
            "state_counts":  state_counts,
            "risk_stats":    risk_stats,
            "pct_safe":      pct_safe,
            "pct_warning":   pct_warning,
            "pct_alert":     pct_alert,
        }

        self._print_summary(results)
        return results

    def _print_summary(self, r: dict):
        print("\n" + "="*55)
        print("  DRIVER SESSION EVALUATION REPORT")
        print("="*55)
        print(f"  Duration       : {r['duration_s']:.1f}s  ({r['total_frames']} frames)")
        print(f"  Effective FPS  : {r['fps_equivalent']}")
        print()
        print("  Time distribution:")
        print(f"    SAFE     : {r['pct_safe']:5.1f}%")
        print(f"    WARNING  : {r['pct_warning']:5.1f}%")
        print(f"    ALERT    : {r['pct_alert']:5.1f}%")
        print()
        print("  Risk score statistics:")
        rs = r["risk_stats"]
        print(f"    Mean : {rs['mean']:.4f}")
        print(f"    Max  : {rs['max']:.4f}")
        print(f"    P95  : {rs['p95']:.4f}")
        print(f"    Std  : {rs['std']:.4f}")
        print()
        print("  Driver state breakdown:")
        for state, cnt in r["state_counts"].items():
            pct = 100 * cnt / max(r["total_frames"], 1)
            print(f"    {state:<15}: {cnt:4d}  ({pct:.1f}%)")
        print("="*55 + "\n")

    # ------------------------------------------------------------------ #
    #  Plots                                                               #
    # ------------------------------------------------------------------ #

    def plot_timeline(self, save_path: Optional[str] = None):
        """Plot risk score over time with alert level zones."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not installed — skipping plot")
            return

        df = self.df.copy()
        t  = df["timestamp"] - df["timestamp"].min()

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="#1a1a2e")

        # ---- Subplot 1: Risk score ----
        ax1 = axes[0]
        ax1.set_facecolor("#16213e")
        ax1.axhspan(0,    0.35, alpha=0.15, color="#2ecc71")
        ax1.axhspan(0.35, 0.65, alpha=0.15, color="#f39c12")
        ax1.axhspan(0.65, 1.0,  alpha=0.15, color="#e74c3c")
        ax1.plot(t, df["risk_score"], color="#e94560", linewidth=1.5, label="Risk score")
        ax1.axhline(0.35, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.axhline(0.65, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.7)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Risk Score", color="#eaeaea")
        ax1.set_title("Driver Risk Timeline", color="#eaeaea", pad=10)
        ax1.legend(facecolor="#16213e", labelcolor="#eaeaea")
        ax1.tick_params(colors="#8892a4")
        for spine in ax1.spines.values():
            spine.set_color("#0f3460")

        # ---- Subplot 2: EAR + MAR ----
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        if "ear" in df.columns and df["ear"].notna().any():
            ax2.plot(t, df["ear"], color="#3498db", linewidth=1.2, label="EAR")
            ax2.axhline(0.25, color="#3498db", linewidth=0.8, linestyle="--", alpha=0.6)
        if "mar" in df.columns and df["mar"].notna().any():
            ax2.plot(t, df["mar"], color="#9b59b6", linewidth=1.2, label="MAR")
            ax2.axhline(0.65, color="#9b59b6", linewidth=0.8, linestyle="--", alpha=0.6)
        ax2.set_ylabel("EAR / MAR", color="#eaeaea")
        ax2.legend(facecolor="#16213e", labelcolor="#eaeaea")
        ax2.tick_params(colors="#8892a4")
        for spine in ax2.spines.values():
            spine.set_color("#0f3460")

        # ---- Subplot 3: Speed ----
        ax3 = axes[2]
        ax3.set_facecolor("#16213e")
        if "speed_kmh" in df.columns and df["speed_kmh"].notna().any():
            ax3.plot(t, df["speed_kmh"], color="#2ecc71", linewidth=1.2, label="Speed (km/h)")
        ax3.set_xlabel("Time (s)", color="#eaeaea")
        ax3.set_ylabel("Speed km/h", color="#eaeaea")
        ax3.legend(facecolor="#16213e", labelcolor="#eaeaea")
        ax3.tick_params(colors="#8892a4")
        for spine in ax3.spines.values():
            spine.set_color("#0f3460")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
            logger.info(f"Timeline plot saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_feature_distribution(self, save_path: Optional[str] = None):
        """Histogram grid of key features by alert level."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        features = ["risk_score", "ear", "mar", "head_yaw", "speed_kmh", "lane_deviation"]
        colours  = {"SAFE": "#2ecc71", "WARNING": "#f39c12",
                    "ALERT": "#e74c3c", "CRITICAL": "#c0392b"}

        df = self.df.dropna(subset=["alert_level"])
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="#1a1a2e")
        axes_flat = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes_flat[i]
            ax.set_facecolor("#16213e")
            if feat not in df.columns:
                ax.set_visible(False)
                continue
            col = pd.to_numeric(df[feat], errors="coerce")
            for level, grp in df.groupby("alert_level"):
                data = pd.to_numeric(grp[feat], errors="coerce").dropna()
                if len(data) > 1:
                    ax.hist(data, bins=30, alpha=0.5,
                            label=level, color=colours.get(level, "#888"),
                            density=True)
            ax.set_title(feat, color="#eaeaea", fontsize=10)
            ax.tick_params(colors="#8892a4", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#0f3460")
            ax.legend(facecolor="#16213e", labelcolor="#eaeaea", fontsize=7)

        plt.suptitle("Feature Distributions by Alert Level",
                     color="#eaeaea", fontsize=13)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
        else:
            plt.show()
        plt.close(fig)

    def plot_confusion(
        self,
        y_true: List[str],
        y_pred: List[str],
        save_path: Optional[str] = None,
    ):
        """
        Plot a confusion matrix for predicted vs true alert levels.
        y_true / y_pred should be lists of level strings.
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        except ImportError:
            return

        labels = ["SAFE", "WARNING", "ALERT"]
        cm     = confusion_matrix(y_true, y_pred, labels=labels)
        disp   = ConfusionMatrixDisplay(cm, display_labels=labels)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", color="#eaeaea")
        ax.tick_params(colors="#eaeaea")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
        else:
            plt.show()
        plt.close(fig)

    def save_report(self, output_path: Optional[str] = None) -> str:
        """Write a text evaluation report and return its path."""
        if output_path is None:
            base        = self.path.stem
            output_path = self.path.parent / f"{base}_report.txt"

        stats = self.summary()

        lines = [
            "DRIVER ASSISTANCE SYSTEM — SESSION EVALUATION REPORT",
            "=" * 55,
            f"Session file    : {self.path}",
            f"Duration        : {stats.get('duration_s', 0):.1f}s",
            f"Total frames    : {stats.get('total_frames', 0)}",
            f"Effective FPS   : {stats.get('fps_equivalent', 0)}",
            "",
            "Time Distribution:",
            f"  SAFE     : {stats.get('pct_safe', 0):5.1f}%",
            f"  WARNING  : {stats.get('pct_warning', 0):5.1f}%",
            f"  ALERT    : {stats.get('pct_alert', 0):5.1f}%",
            "",
            "Risk Score Stats:",
        ]
        for k, v in stats.get("risk_stats", {}).items():
            lines.append(f"  {k:<6}: {v:.4f}")

        lines += [
            "",
            "Driver State Breakdown:",
        ]
        for state, cnt in stats.get("state_counts", {}).items():
            pct = 100 * cnt / max(stats.get("total_frames", 1), 1)
            lines.append(f"  {state:<15}: {cnt:4d}  ({pct:.1f}%)")

        report_text = "\n".join(lines)
        Path(output_path).write_text(report_text, encoding="utf-8")
        logger.info(f"Report saved: {output_path}")
        return str(output_path)


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import random
    from fusion_engine  import FusionResult, VisionFeatures
    from sensor_module  import SensorReading
    from alert_system   import AlertLevel
    from fusion_engine  import FusionEngine

    logging.basicConfig(level=logging.INFO)

    engine  = FusionEngine()
    csv_out = LOG_DIR / "test_session.csv"
    ev_log  = EventLogger(str(csv_out))

    print(f"Writing test session to {csv_out} ...")

    for i in range(200):
        t  = i * 0.1
        ear = max(0.10, 0.30 - 0.01 * (i // 20))
        vis = VisionFeatures(
            ear=ear, mar=0.25 + 0.3 * (i % 30 == 0),
            head_yaw=15 * np.sin(t),
            cnn_state="drowsy" if ear < 0.22 else "alert",
            cnn_confidence=0.85,
        )
        sen = SensorReading(
            speed_kmh=70 + 5 * np.sin(t),
            acceleration=0.3 * np.cos(t),
            lane_deviation=8 * np.sin(t / 3),
            timestamp=time.time(),
        )
        result = engine.fuse(vis, sen)
        level  = AlertLevel.ALERT if result.risk_score > 0.65 else (
                 AlertLevel.WARNING if result.risk_score > 0.35 else AlertLevel.SAFE)
        ev_log.log(result, level)

    ev_log.close()
    print("Done. Evaluating...")

    evaluator = SessionEvaluator(str(csv_out))
    evaluator.summary()
    evaluator.plot_timeline(str(csv_out.with_suffix(".png")))
    print(f"Plot saved to {csv_out.with_suffix('.png')}")
