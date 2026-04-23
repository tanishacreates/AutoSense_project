"""
alert_system.py
---------------
Multi-modal alert system for the driver assistance pipeline.

Alert levels (in escalating severity):
  SAFE     → no alert
  WARNING  → gentle chime + yellow overlay
  ALERT    → loud beep + red overlay + TTS message
  CRITICAL → repeated rapid beep + red flashing + TTS

Audio is generated procedurally with numpy/pygame — no external audio files needed.
Text-to-speech uses pyttsx3 (works offline, no API key).

Usage:
    alerts = AlertSystem()
    alerts.trigger(AlertLevel.ALERT, "Drowsiness detected! Please take a break.")
    alerts.trigger(AlertLevel.WARNING, reason="Distraction")
    alerts.stop()
"""

import time
import threading
import logging
import queue
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    SAFE     = 0
    WARNING  = 1
    ALERT    = 2
    CRITICAL = 3


@dataclass
class AlertEvent:
    level:     AlertLevel
    message:   str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# ------------------------------------------------------------------ #
#  Tone generator (numpy → pygame)                                     #
# ------------------------------------------------------------------ #

class ToneGenerator:
    """
    Generate audio tones as numpy arrays and play them with pygame.
    Handles pygame init gracefully if a display/audio device is absent
    (e.g. headless CI) by falling back to a silent no-op.
    """

    SAMPLE_RATE = 44_100

    def __init__(self):
        self._available = False
        try:
            import pygame
            import pygame.mixer
            pygame.mixer.pre_init(self.SAMPLE_RATE, -16, 1, 512)
            pygame.mixer.init()
            self._pygame = pygame
            self._available = True
            logger.info("Pygame audio initialised")
        except Exception as exc:
            logger.warning(f"Pygame audio unavailable ({exc}) — audio alerts silent")

    def _make_wave(
        self,
        freq_hz: float,
        duration_s: float,
        volume: float = 0.8,
        shape: str = "sine",
        fade_ms: int = 30,
    ) -> "pygame.mixer.Sound":
        """Build a Sound object from a generated waveform."""
        n_samples = int(self.SAMPLE_RATE * duration_s)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)

        if shape == "sine":
            wave = np.sin(2 * np.pi * freq_hz * t)
        elif shape == "square":
            wave = np.sign(np.sin(2 * np.pi * freq_hz * t))
        elif shape == "sawtooth":
            wave = 2 * (t * freq_hz - np.floor(t * freq_hz + 0.5))
        else:
            wave = np.sin(2 * np.pi * freq_hz * t)

        # Fade in/out to remove clicks
        fade_samples = int(self.SAMPLE_RATE * fade_ms / 1000)
        if fade_samples > 0 and fade_samples * 2 < n_samples:
            fade_in  = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples]  *= fade_in
            wave[-fade_samples:] *= fade_out

        wave = (wave * volume * 32767).astype(np.int16)
        sound = self._pygame.sndarray.make_sound(wave)
        return sound

    def play(
        self,
        freq_hz: float,
        duration_s: float,
        volume: float = 0.8,
        shape: str = "sine",
        repeats: int = 1,
        gap_s: float = 0.1,
    ):
        """Play a tone (blocks for total duration)."""
        if not self._available:
            return
        try:
            sound = self._make_wave(freq_hz, duration_s, volume, shape)
            for i in range(repeats):
                sound.play()
                time.sleep(duration_s)
                if i < repeats - 1:
                    time.sleep(gap_s)
        except Exception as exc:
            logger.debug(f"Tone play error: {exc}")

    def stop_all(self):
        if self._available:
            try:
                self._pygame.mixer.stop()
            except Exception:
                pass


# ------------------------------------------------------------------ #
#  TTS engine                                                          #
# ------------------------------------------------------------------ #

class TTSEngine:
    """Offline text-to-speech via pyttsx3."""

    def __init__(self, rate: int = 165, volume: float = 0.9):
        self._engine   = None
        self._rate     = rate
        self._volume   = volume
        self._lock     = threading.Lock()

        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate",   rate)
            eng.setProperty("volume", volume)
            self._engine = eng
            logger.info("TTS engine initialised")
        except Exception as exc:
            logger.warning(f"TTS unavailable ({exc}) — voice alerts silent")

    def speak(self, text: str):
        if self._engine is None:
            logger.info(f"[TTS WOULD SAY]: {text}")
            return
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                logger.debug(f"TTS error: {exc}")


# ------------------------------------------------------------------ #
#  Visual overlay helpers (OpenCV)                                     #
# ------------------------------------------------------------------ #

class VisualOverlay:
    """
    Draws coloured alert banners onto OpenCV frames.
    Works with any BGR numpy array (the frame from the video pipeline).
    """

    # BGR colours for each alert level
    COLOURS = {
        AlertLevel.SAFE    : (50,  200, 50),    # green
        AlertLevel.WARNING : (0,   200, 255),   # yellow
        AlertLevel.ALERT   : (0,   80,  220),   # orange-red
        AlertLevel.CRITICAL: (0,   0,   255),   # full red
    }

    LABELS = {
        AlertLevel.SAFE    : "SAFE",
        AlertLevel.WARNING : "WARNING",
        AlertLevel.ALERT   : "ALERT",
        AlertLevel.CRITICAL: "CRITICAL",
    }

    def __init__(self):
        self._flash_state  = False
        self._last_flash   = 0.0
        self._flash_hz     = 4.0   # flashes per second for CRITICAL

    def draw(self, frame: np.ndarray, level: AlertLevel, message: str = "") -> np.ndarray:
        """
        Return a copy of `frame` with an alert overlay drawn on it.
        """
        import cv2
        out = frame.copy()
        h, w = out.shape[:2]

        if level == AlertLevel.SAFE:
            # Minimal green corner indicator only
            cv2.rectangle(out, (w - 120, 10), (w - 10, 50),
                          self.COLOURS[level], -1)
            cv2.putText(out, "SAFE", (w - 105, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return out

        colour = self.COLOURS[level]

        # CRITICAL: flashing effect
        if level == AlertLevel.CRITICAL:
            now = time.time()
            if now - self._last_flash > 1.0 / self._flash_hz:
                self._flash_state  = not self._flash_state
                self._last_flash   = now
            if not self._flash_state:
                # flash off — just draw a dim border
                cv2.rectangle(out, (0, 0), (w, h), colour, 4)
                return out

        # Semi-transparent banner at the top
        banner_h = max(70, h // 8)
        overlay  = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), colour, -1)
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # Border around the whole frame
        border_thickness = 4 if level == AlertLevel.WARNING else 6
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), colour, border_thickness)

        # Level label
        label     = self.LABELS[level]
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.2 if level == AlertLevel.CRITICAL else 1.0
        cv2.putText(out, label, (20, 48), font, font_size,
                    (255, 255, 255), 3, cv2.LINE_AA)

        # Message
        if message:
            short_msg = message[:60]
            cv2.putText(out, short_msg, (20, banner_h - 12), font, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)

        return out

    @staticmethod
    def draw_sensor_hud(
        frame: np.ndarray,
        speed_kmh: float,
        risk_score: float,
        driver_state: str,
    ) -> np.ndarray:
        """
        Draw a small HUD in the bottom-left showing key metrics.
        """
        import cv2
        out   = frame.copy()
        h, w  = out.shape[:2]
        x0, y0 = 10, h - 110

        # Semi-transparent background
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + 210, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        white = (220, 220, 220)
        size  = 0.5
        th    = 1

        lines = [
            f"Speed    : {speed_kmh:.0f} km/h",
            f"Risk     : {risk_score:.2f}",
            f"State    : {driver_state}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(out, line, (x0 + 8, y0 + 24 + i * 28),
                        font, size, white, th, cv2.LINE_AA)

        return out


# ------------------------------------------------------------------ #
#  Main AlertSystem class                                              #
# ------------------------------------------------------------------ #

class AlertSystem:
    """
    Orchestrates audio + visual alerts.

    Thread model:
      - `trigger()` is non-blocking — puts events onto an internal queue.
      - A background worker thread processes the queue in order.
      - This prevents the main video loop from stalling on TTS / audio.

    Parameters
    ----------
    enable_audio   : play beep tones
    enable_tts     : speak alert messages aloud
    enable_visual  : return annotated frames from draw_frame()
    cooldown_s     : minimum seconds between repeated alerts of the same level
    on_alert_cb    : optional callback(AlertEvent) called on every triggered alert
    """

    # Beep profiles:  (freq_hz, duration_s, volume, shape, repeats, gap_s)
    BEEP_PROFILES = {
        AlertLevel.WARNING : (880,  0.15, 0.5, "sine",    1, 0.0),
        AlertLevel.ALERT   : (1200, 0.25, 0.8, "square",  2, 0.1),
        AlertLevel.CRITICAL: (1500, 0.15, 1.0, "square",  4, 0.05),
    }

    TTS_MESSAGES = {
        AlertLevel.WARNING : "Warning. Please focus on the road.",
        AlertLevel.ALERT   : "Alert! Drowsiness or distraction detected. Please stay alert.",
        AlertLevel.CRITICAL: "Critical alert! Pull over safely and rest immediately.",
    }

    def __init__(
        self,
        enable_audio:  bool = True,
        enable_tts:    bool = True,
        enable_visual: bool = True,
        cooldown_s:    float = 5.0,
        on_alert_cb:   Optional[Callable[[AlertEvent], None]] = None,
    ):
        self.enable_audio  = enable_audio
        self.enable_tts    = enable_tts
        self.enable_visual = enable_visual
        self.cooldown_s    = cooldown_s
        self._callback     = on_alert_cb

        self._tone_gen  = ToneGenerator()   if enable_audio  else None
        self._tts       = TTSEngine()       if enable_tts    else None
        self._visual    = VisualOverlay()   if enable_visual else None

        self._queue: queue.Queue[AlertEvent] = queue.Queue(maxsize=20)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Cooldown tracking per level
        self._last_alert_time: dict[AlertLevel, float] = {
            lv: 0.0 for lv in AlertLevel
        }

        # Current display state (set by worker, read by draw_frame)
        self._current_level   = AlertLevel.SAFE
        self._current_message = ""
        self._level_lock      = threading.Lock()

        self.start()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        """Start the background alert worker."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="AlertWorker"
        )
        self._worker_thread.start()

    def stop(self):
        """Flush queue and stop worker."""
        self._running = False
        self._queue.put(None)   # sentinel
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
        if self._tone_gen:
            self._tone_gen.stop_all()

    def trigger(self, level: AlertLevel, message: str = ""):
        """
        Queue an alert.  Non-blocking — returns immediately.
        Uses the default TTS message if `message` is empty.
        """
        if level == AlertLevel.SAFE:
            with self._level_lock:
                self._current_level   = AlertLevel.SAFE
                self._current_message = ""
            return

        if not message:
            message = self.TTS_MESSAGES.get(level, str(level.name))

        # Cooldown check
        now = time.time()
        last = self._last_alert_time.get(level, 0.0)
        if now - last < self.cooldown_s:
            return   # still in cooldown

        self._last_alert_time[level] = now

        event = AlertEvent(level=level, message=message)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.debug("Alert queue full — dropping event")

    def draw_frame(
        self,
        frame: np.ndarray,
        speed_kmh: float = 0.0,
        risk_score: float = 0.0,
        driver_state: str = "UNKNOWN",
    ) -> np.ndarray:
        """
        Apply the current alert overlay + HUD to a video frame.
        Returns the annotated frame (does not modify the input).
        """
        if self._visual is None or frame is None:
            return frame

        with self._level_lock:
            level   = self._current_level
            message = self._current_message

        out = self._visual.draw(frame, level, message)
        out = self._visual.draw_sensor_hud(out, speed_kmh, risk_score, driver_state)
        return out

    def get_current_level(self) -> AlertLevel:
        with self._level_lock:
            return self._current_level

    # ------------------------------------------------------------------ #
    #  Worker                                                              #
    # ------------------------------------------------------------------ #

    def _worker(self):
        """Background thread: consume alert queue and fire audio/TTS."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None:   # sentinel
                break

            logger.info(f"Alert triggered: {event.level.name} — {event.message}")

            # Update display state immediately
            with self._level_lock:
                self._current_level   = event.level
                self._current_message = event.message

            # Fire callback
            if self._callback:
                try:
                    self._callback(event)
                except Exception as exc:
                    logger.debug(f"Alert callback error: {exc}")

            # Audio beep
            if self._tone_gen:
                prof = self.BEEP_PROFILES.get(event.level)
                if prof:
                    threading.Thread(
                        target=self._tone_gen.play,
                        args=prof,
                        daemon=True
                    ).start()

            # TTS (runs in separate thread so it doesn't block beeps)
            if self._tts and event.level.value >= AlertLevel.ALERT.value:
                threading.Thread(
                    target=self._tts.speak,
                    args=(event.message,),
                    daemon=True
                ).start()

            self._queue.task_done()


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Alert System Test ===")

    fired = []
    alert = AlertSystem(
        enable_audio=True,
        enable_tts=True,
        enable_visual=True,
        cooldown_s=1.0,
        on_alert_cb=lambda e: fired.append(e),
    )

    for level, msg in [
        (AlertLevel.WARNING,  "Slight distraction"),
        (AlertLevel.ALERT,    "Drowsiness detected"),
        (AlertLevel.CRITICAL, "Eyes closed 3 seconds"),
    ]:
        print(f"\nTriggering {level.name}: {msg}")
        alert.trigger(level, msg)
        time.sleep(2.0)

    alert.stop()
    print(f"\n{len(fired)} alert events fired.")
    for e in fired:
        print(f"  [{e.level.name}] {e.message}  @ {e.timestamp:.2f}")
