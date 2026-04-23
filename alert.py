"""
AutoSense- Alert System
Handles multi-level alerts: audio beeps, log file, and console warnings.
"""

import time
import csv
import threading
import os
from dataclasses import dataclass
from typing import List, Dict
from detector import DriverState

try:
    # import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class Alert:
    level:    int    # 1=info, 2=warning, 3=critical
    message:  str
    timestamp: float = 0.0
    active:   bool  = True

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


ALERT_COLORS = {
    1: (0, 200, 100),    # green-ish
    2: (0, 165, 255),    # orange
    3: (0, 0, 220),      # red
}

ALERT_LABELS = {1: "INFO", 2: "WARNING", 3: "CRITICAL"}


class AlertSystem:
    """
    Manages alert state, cooldowns, audio beeps, and optional CSV logging.
    """

    # Cooldown seconds per alert type to avoid spam
    COOLDOWNS = {
        "drowsy_mild":     10.0,
        "drowsy_moderate": 5.0,
        "drowsy_severe":   2.0,
        "distracted":      8.0,
        "eyes_closed":     3.0,
        "yawning":         15.0,
        "no_face":         5.0,
    }

    def __init__(self, save_log: bool = False):
        self.save_log = save_log
        self._active_alerts: Dict[str, Alert] = {}
        self._last_triggered: Dict[str, float] = {}
        self._lock = threading.Lock()

        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
                self._audio_ok = True
            except Exception:
                self._audio_ok = False
        else:
            self._audio_ok = False

        if save_log:
            self._init_log()

    # ── Log ───────────────────────────────────────────────────────────────────

    def _init_log(self):
        os.makedirs("logs", exist_ok=True)
        fname = f"logs/vigidrive_{int(time.time())}.csv"
        self._log_file = open(fname, "w", newline="")
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(["timestamp", "alert_type", "level", "message",
                             "ear", "mar", "perclos", "blink_rate",
                             "pitch", "yaw", "drowsiness_level"])
        print(f"[INFO] Logging to {fname}")

    def _write_log(self, alert_type: str, alert: Alert, state: DriverState):
        if not self.save_log:
            return
        self._csv.writerow([
            round(alert.timestamp, 3),
            alert_type,
            alert.level,
            alert.message,
            round(state.ear, 4),
            round(state.mar, 4),
            round(state.perclos, 4),
            round(state.blink_rate, 2),
            round(state.pitch, 2),
            round(state.yaw, 2),
            state.drowsiness_level,
        ])
        self._log_file.flush()

    # ── Processing ────────────────────────────────────────────────────────────

    def process(self, state: DriverState):
        """Evaluate state and raise/clear alerts."""
        with self._lock:
            now = time.time()

            if not state.face_detected:
                self._trigger("no_face", Alert(1, "Face not detected — adjust camera"), state, now)
            else:
                self._clear("no_face")

            # Drowsiness levels
            if state.drowsiness_level == 1:
                self._trigger("drowsy_mild",
                              Alert(1, "Mild drowsiness detected — stay alert"), state, now)
                self._clear("drowsy_moderate")
                self._clear("drowsy_severe")

            elif state.drowsiness_level == 2:
                self._trigger("drowsy_moderate",
                              Alert(2, "Moderate drowsiness — consider a break"), state, now)
                self._clear("drowsy_mild")
                self._clear("drowsy_severe")

            elif state.drowsiness_level >= 3:
                self._trigger("drowsy_severe",
                              Alert(3, "⚠ SEVERE DROWSINESS — PULL OVER NOW!"), state, now)
                self._clear("drowsy_mild")
                self._clear("drowsy_moderate")
            else:
                self._clear("drowsy_mild")
                self._clear("drowsy_moderate")
                self._clear("drowsy_severe")

            # Distraction
            if state.distracted:
                self._trigger("distracted",
                              Alert(2, "Driver distracted — eyes on the road!"), state, now)
            else:
                self._clear("distracted")

            # Eyes closed (sustained)
            if state.eyes_closed:
                self._trigger("eyes_closed",
                              Alert(2, "Eyes closed detected"), state, now)
            else:
                self._clear("eyes_closed")

            # Yawning
            if state.yawning:
                self._trigger("yawning",
                              Alert(1, "Yawning detected — fatigue sign"), state, now)

    def _trigger(self, key: str, alert: Alert, state: DriverState, now: float):
        cooldown = self.COOLDOWNS.get(key, 5.0)
        last = self._last_triggered.get(key, 0.0)
        if now - last < cooldown:
            return
        self._last_triggered[key] = now
        self._active_alerts[key]   = alert

        label = ALERT_LABELS[alert.level]
        print(f"[{label}] {alert.message}")

        self._write_log(key, alert, state)

        if alert.level >= 2 and self._audio_ok:
            threading.Thread(target=self._beep, args=(alert.level,), daemon=True).start()

    def _clear(self, key: str):
        self._active_alerts.pop(key, None)

    def get_active_alerts(self) -> List[Alert]:
        with self._lock:
            return sorted(self._active_alerts.values(),
                          key=lambda a: -a.level)

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _beep(self, level: int):
        """Generate a simple sine-wave beep."""
        if not NUMPY_AVAILABLE or not self._audio_ok:
            return
        try:
            sr   = 44100
            freq = 880 if level >= 3 else 660
            dur  = 0.4  if level >= 3 else 0.25
            t    = np.linspace(0, dur, int(sr * dur), endpoint=False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            snd  = pygame.sndarray.make_sound(wave)
            snd.play()
            time.sleep(dur + 0.05)
        except Exception:
            pass

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        if self.save_log and hasattr(self, "_log_file"):
            self._log_file.close()
            print("[INFO] Log file saved.")
