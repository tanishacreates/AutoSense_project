"""
AutoSense - Dashboard / Display Overlay
Renders HUD on top of the camera feed.
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from detector import DriverState
from alert import Alert

# ── Color palette (BGR) ───────────────────────────────────────────────────────
C_GREEN   = (50,  205,  50)
C_YELLOW  = (0,   210, 210)
C_ORANGE  = (0,   140, 255)
C_RED     = (30,   30, 220)
C_WHITE   = (240, 240, 240)
C_BLACK   = (10,   10,  10)
C_DARK    = (20,   20,  20)
C_TEAL    = (180, 200,   0)

LEVEL_COLORS = {
    0: C_GREEN,
    1: C_YELLOW,
    2: C_ORANGE,
    3: C_RED,
}

DROWSY_LABELS = {
    0: "ALERT",
    1: "MILD FATIGUE",
    2: "DROWSY",
    3: "DANGER",
}


class Dashboard:
    def __init__(self):
        self.fps  = 0.0
        self._start = time.time()

    def render(self, frame: np.ndarray, state: DriverState,
               alerts: List[Alert]) -> np.ndarray:
        out = frame.copy()

        self._draw_landmarks(out, state)
        self._draw_metrics_panel(out, state)
        self._draw_status_bar(out, state)
        self._draw_alerts(out, alerts)
        self._draw_fps(out)

        return out

    # ── Landmarks ─────────────────────────────────────────────────────────────

    def _draw_landmarks(self, frame: np.ndarray, state: DriverState):
        if not state.face_detected or state.landmarks is None:
            return

        try:
            import mediapipe as mp
            mp_draw = mp.solutions.drawing_utils
            mp_style = mp.solutions.drawing_styles
            mp_face  = mp.solutions.face_mesh

            # Draw mesh
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=state.landmarks,
                connections=mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=(60, 60, 60), thickness=1, circle_radius=0),
            )
            # Eye contours highlighted
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=state.landmarks,
                connections=mp_face.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=LEVEL_COLORS[state.drowsiness_level], thickness=2),
            )
            mp_draw.draw_landmarks(
                image=frame,
                landmark_list=state.landmarks,
                connections=mp_face.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(
                    color=LEVEL_COLORS[state.drowsiness_level], thickness=2),
            )
        except Exception:
            pass

    # ── Metrics Panel (top-left) ───────────────────────────────────────────────

    def _draw_metrics_panel(self, frame: np.ndarray, state: DriverState):
        h, w = frame.shape[:2]
        panel_w, panel_h = 260, 230
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h),
                      C_DARK, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Title
        cv2.putText(frame, "AutoSense", (20, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, C_TEAL, 2)

        metrics = [
            ("EAR",        f"{state.ear:.3f}",          state.eyes_closed),
            ("MAR",        f"{state.mar:.3f}",          state.yawning),
            ("PERCLOS",    f"{state.perclos*100:.1f}%",  state.perclos > 0.20),
            ("BLINK/min",  f"{state.blink_rate:.1f}",   False),
            ("PITCH",      f"{state.pitch:.1f}°",        state.head_down),
            ("YAW",        f"{state.yaw:.1f}°",          state.looking_away),
        ]

        for i, (label, val, warn) in enumerate(metrics):
            y = 60 + i * 28
            color = C_ORANGE if warn else C_WHITE
            cv2.putText(frame, f"{label:<10} {val}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    # ── Status Bar (bottom) ───────────────────────────────────────────────────

    def _draw_status_bar(self, frame: np.ndarray, state: DriverState):
        h, w = frame.shape[:2]
        bar_h = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), C_DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Driver state label
        if not state.face_detected:
            label = "NO FACE DETECTED"
            color = C_YELLOW
        elif state.drowsy and state.distracted:
            label = "DROWSY + DISTRACTED"
            color = C_RED
        elif state.drowsy:
            label = DROWSY_LABELS[state.drowsiness_level]
            color = LEVEL_COLORS[state.drowsiness_level]
        elif state.distracted:
            label = "DISTRACTED"
            color = C_ORANGE
        else:
            label = "ALERT"
            color = C_GREEN

        # Pulsing effect for danger
        if state.drowsiness_level >= 3:
            pulse = int(abs(np.sin(time.time() * 4)) * 100)
            color = (max(0, color[0] - pulse),
                     max(0, color[1] - pulse),
                     min(255, color[2] + pulse))

        cv2.putText(frame, label, (w // 2 - 120, h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        # Side indicators
        # Yawn
        yawn_c = C_ORANGE if state.yawning else C_WHITE
        cv2.putText(frame, "YAWN", (20, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, yawn_c, 1)
        # Eyes
        eye_c = C_RED if state.eyes_closed else C_GREEN
        cv2.putText(frame, "EYES", (80, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, eye_c, 1)

        # Session time
        elapsed = int(time.time() - self._start)
        m, s = divmod(elapsed, 60)
        cv2.putText(frame, f"{m:02d}:{s:02d}", (w - 80, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1)

    # ── Alert Banners ─────────────────────────────────────────────────────────

    def _draw_alerts(self, frame: np.ndarray, alerts: List[Alert]):
        if not alerts:
            return

        h, w = frame.shape[:2]
        top_alert = alerts[0]

        if top_alert.level >= 2:
            overlay = frame.copy()
            border = 6
            cv2.rectangle(overlay, (0, 0), (w, border),
                          LEVEL_COLORS.get(top_alert.level, C_RED), -1)
            cv2.rectangle(overlay, (0, h - border), (w, h),
                          LEVEL_COLORS.get(top_alert.level, C_RED), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Show top 2 alerts
        for i, alert in enumerate(alerts[:2]):
            y = 280 + i * 35
            color = LEVEL_COLORS.get(alert.level, C_WHITE)
            bg_overlay = frame.copy()
            tw = cv2.getTextSize(alert.message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
            cv2.rectangle(bg_overlay, (10, y - 22), (20 + tw, y + 6), C_DARK, -1)
            cv2.addWeighted(bg_overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, alert.message, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # ── FPS ───────────────────────────────────────────────────────────────────

    def _draw_fps(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_WHITE, 1)
