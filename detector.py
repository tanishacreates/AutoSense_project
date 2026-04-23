"""
AutoSense - Core Detector Module
Handles face detection, EAR, MAR, head pose estimation,
and driver state classification.
"""

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.spatial import distance as dist

try:
    import mediapipe as mp
    _test = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    _test.close()
    del _test
    MEDIAPIPE_AVAILABLE = True
    print("[INFO] MediaPipe FaceMesh available.")
except Exception as _e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[WARN] MediaPipe not functional ({_e}). Falling back to dlib.")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

# ─── MediaPipe Landmark Indices ───────────────────────────────────────────────
LEFT_EYE     = [362, 385, 387, 263, 373, 380]
RIGHT_EYE    = [33,  160, 158, 133, 153, 144]
MOUTH        = [61,  39,  0,   269, 291, 405, 17,  181]
NOSE_TIP     = 1
LEFT_EAR_PT  = 234
RIGHT_EAR_PT = 454
CHIN         = 152
FOREHEAD     = 10


# ─── Sensitivity Presets ──────────────────────────────────────────────────────
SENSITIVITY = {
    "low": {
        "EAR_THRESHOLD":        0.20,
        "EAR_CONSEC_FRAMES":    48,
        "MAR_THRESHOLD":        0.75,
        "YAWN_CONSEC_FRAMES":   20,
        "PITCH_THRESHOLD":      25,
        "YAW_THRESHOLD":        30,
        "DISTRACTION_FRAMES":   60,
        "PERCLOS_THRESHOLD":    0.30,
    },
    "medium": {
        "EAR_THRESHOLD":        0.22,
        "EAR_CONSEC_FRAMES":    36,
        "MAR_THRESHOLD":        0.70,
        "YAWN_CONSEC_FRAMES":   15,
        "PITCH_THRESHOLD":      20,
        "YAW_THRESHOLD":        25,
        "DISTRACTION_FRAMES":   45,
        "PERCLOS_THRESHOLD":    0.25,
    },
    "high": {
        "EAR_THRESHOLD":        0.25,
        "EAR_CONSEC_FRAMES":    24,
        "MAR_THRESHOLD":        0.65,
        "YAWN_CONSEC_FRAMES":   10,
        "PITCH_THRESHOLD":      15,
        "YAW_THRESHOLD":        20,
        "DISTRACTION_FRAMES":   30,
        "PERCLOS_THRESHOLD":    0.20,
    },
}


@dataclass
class DriverState:
    """Holds the complete analysis result for one frame."""
    ear:              float = 1.0
    mar:              float = 0.0
    perclos:          float = 0.0
    blink_rate:       float = 0.0
    pitch:            float = 0.0
    yaw:              float = 0.0
    roll:             float = 0.0

    eyes_closed:      bool  = False
    yawning:          bool  = False
    head_down:        bool  = False
    looking_away:     bool  = False
    face_detected:    bool  = False

    drowsiness_level: int   = 0
    distracted:       bool  = False
    drowsy:           bool  = False

    landmarks:        Optional[object] = field(default=None, repr=False)
    frame_shape:      Tuple[int,int]   = (720, 1280)
    timestamp:        float = field(default_factory=time.time)


class DriverMonitor:
    """
    Main driver monitoring class.
    Uses MediaPipe FaceMesh (preferred) or dlib as fallback.
    """

    def __init__(self, sensitivity: str = "medium"):
        self.cfg = SENSITIVITY[sensitivity]
        self._init_detector()
        self._init_state()
        print(f"[INFO] DriverMonitor initialized | sensitivity={sensitivity}")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_detector(self):
        if MEDIAPIPE_AVAILABLE:
            self.backend   = "mediapipe"
            self.mp_face   = mp.solutions.face_mesh
            self.face_mesh = self.mp_face.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            print("[INFO] Using MediaPipe FaceMesh backend")
        elif DLIB_AVAILABLE:
            self.backend  = "dlib"
            self.detector = dlib.get_frontal_face_detector()
            try:
                self.predictor = dlib.shape_predictor(
                    "models/shape_predictor_68_face_landmarks.dat"
                )
                print("[INFO] Using dlib backend")
            except Exception:
                print("[ERROR] dlib predictor model not found.")
                print("        Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                self.backend = "none"
        else:
            self.backend = "none"

        # FIX: fail loudly instead of silently running with no detector
        # Previously the system would start, detect nothing, and log 0 blinks/yawns
        if self.backend == "none":
            raise RuntimeError(
                "\n\n[FATAL] No face detection backend is available.\n"
                "Fix MediaPipe (see README) OR download the dlib shape predictor:\n"
                "  curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "  bunzip2 shape_predictor_68_face_landmarks.dat.bz2\n"
                "  mv shape_predictor_68_face_landmarks.dat models/\n"
            )

    def _init_state(self):
        self.ear_counter   = 0
        self.yawn_counter  = 0
        self.dist_counter  = 0

        # FIX: use deque for O(1) append/evict instead of list.pop(0) which is O(n)
        self.perclos_max    = 900   # ~30s at 30fps
        self.perclos_window = deque(maxlen=self.perclos_max)

        self.blink_window = 60.0   # seconds
        self.blink_times  = deque()  # timestamps; trim from left as they expire

        self.total_blinks  = 0
        self.total_yawns   = 0
        self.session_start = time.time()

        self._prev_ear_closed = False

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> DriverState:
        """Analyze a single BGR frame and return a DriverState."""
        state = DriverState(frame_shape=frame.shape[:2])

        if self.backend == "mediapipe":
            self._analyze_mediapipe(frame, state)
        elif self.backend == "dlib":
            self._analyze_dlib(frame, state)
        else:
            return state

        if state.face_detected:
            self._update_counters(state)
            self._classify(state)

        return state

    # ── MediaPipe Analysis ────────────────────────────────────────────────────

    def _analyze_mediapipe(self, frame: np.ndarray, state: DriverState):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return

        state.face_detected = True
        lms = results.multi_face_landmarks[0]
        state.landmarks = lms

        h, w = frame.shape[:2]
        pts  = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark])

        left_ear  = self._ear(pts, LEFT_EYE)
        right_ear = self._ear(pts, RIGHT_EYE)
        state.ear = (left_ear + right_ear) / 2.0
        state.mar = self._mar(pts, MOUTH)
        state.pitch, state.yaw, state.roll = self._head_pose_mediapipe(pts, frame.shape)

    def _ear(self, pts: np.ndarray, eye_idx: List[int]) -> float:
        """Eye Aspect Ratio (Soukupová & Čech, 2016)."""
        p = pts[eye_idx]
        A = dist.euclidean(p[1], p[5])
        B = dist.euclidean(p[2], p[4])
        C = dist.euclidean(p[0], p[3])
        return (A + B) / (2.0 * C + 1e-6)

    def _mar(self, pts: np.ndarray, mouth_idx: List[int]) -> float:
        """Mouth Aspect Ratio for yawn detection."""
        p = pts[mouth_idx]
        A = dist.euclidean(p[1], p[7])
        B = dist.euclidean(p[2], p[6])
        C = dist.euclidean(p[3], p[5])
        D = dist.euclidean(p[0], p[4])
        return (A + B + C) / (2.0 * D + 1e-6)

    def _head_pose_mediapipe(
        self, pts: np.ndarray, shape: Tuple
    ) -> Tuple[float, float, float]:
        """Estimate head pose using solvePnP with 6 facial landmarks."""
        h, w = shape[:2]

        model_3d = np.array([
            [0.0,    0.0,    0.0   ],
            [0.0,   -330.0, -65.0 ],
            [-225.0, 170.0, -135.0],
            [225.0,  170.0, -135.0],
            [-150.0,-150.0, -125.0],
            [150.0, -150.0, -125.0],
        ], dtype=np.float64)

        image_2d = np.array([
            pts[NOSE_TIP],
            pts[CHIN],
            pts[LEFT_EAR_PT],
            pts[RIGHT_EAR_PT],
            pts[61],
            pts[291],
        ], dtype=np.float64)

        focal   = w
        cam_mat = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1   ]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model_3d, image_2d, cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy       = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
            yaw   = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
            roll  = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
            yaw   = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            roll  = 0.0

        return float(pitch), float(yaw), float(roll)

    # ── dlib Analysis (fallback) ──────────────────────────────────────────────

    def _analyze_dlib(self, frame: np.ndarray, state: DriverState):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        if not faces:
            return

        state.face_detected = True
        shape = self.predictor(gray, faces[0])
        pts   = np.array([[shape.part(i).x, shape.part(i).y]
                          for i in range(68)], dtype=np.float64)

        LEFT_DLIB  = list(range(36, 42))
        RIGHT_DLIB = list(range(42, 48))
        MOUTH_DLIB = [48, 50, 52, 54, 56, 58, 60, 64]

        left_ear  = self._ear(pts, LEFT_DLIB)
        right_ear = self._ear(pts, RIGHT_DLIB)
        state.ear = (left_ear + right_ear) / 2.0
        state.mar = self._mar(pts, MOUTH_DLIB)

    # ── Counter Updates ───────────────────────────────────────────────────────

    def _update_counters(self, state: DriverState):
        cfg = self.cfg

        # ── Eye closure ──
        eyes_closed_now  = state.ear < cfg["EAR_THRESHOLD"]
        state.eyes_closed = eyes_closed_now

        if eyes_closed_now:
            self.ear_counter += 1
        else:
            if self._prev_ear_closed:
                self.total_blinks += 1
                self.blink_times.append(time.time())
            self.ear_counter = 0
        self._prev_ear_closed = eyes_closed_now

        # FIX: deque with maxlen — append is O(1), eviction is automatic
        self.perclos_window.append(1.0 if eyes_closed_now else 0.0)
        state.perclos = float(np.mean(self.perclos_window)) if self.perclos_window else 0.0

        # FIX: trim expired blink timestamps from the left — O(1) per frame
        now = time.time()
        while self.blink_times and now - self.blink_times[0] > self.blink_window:
            self.blink_times.popleft()
        state.blink_rate = len(self.blink_times) * (60.0 / self.blink_window)

        # ── Yawning ──
        yawning_now = state.mar > cfg["MAR_THRESHOLD"]
        if yawning_now:
            self.yawn_counter += 1
            if self.yawn_counter == cfg["YAWN_CONSEC_FRAMES"]:
                self.total_yawns += 1
        else:
            self.yawn_counter = 0
        state.yawning = yawning_now and (self.yawn_counter >= cfg["YAWN_CONSEC_FRAMES"])

        # ── Head pose ──
        state.head_down    = state.pitch < -cfg["PITCH_THRESHOLD"]
        state.looking_away = abs(state.yaw) > cfg["YAW_THRESHOLD"]

        if state.looking_away or state.head_down:
            self.dist_counter += 1
        else:
            self.dist_counter = 0

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self, state: DriverState):
        cfg = self.cfg

        state.distracted = self.dist_counter >= cfg["DISTRACTION_FRAMES"]

        score = 0

        if state.perclos > cfg["PERCLOS_THRESHOLD"] + 0.15:
            score += 3
        elif state.perclos > cfg["PERCLOS_THRESHOLD"]:
            score += 2
        elif state.perclos > cfg["PERCLOS_THRESHOLD"] - 0.05:
            score += 1

        if self.ear_counter >= cfg["EAR_CONSEC_FRAMES"]:
            score += 2
        elif self.ear_counter >= cfg["EAR_CONSEC_FRAMES"] // 2:
            score += 1

        if state.yawning:
            score += 1

        if state.head_down:
            score += 1

        if state.blink_rate < 8 or state.blink_rate > 30:
            score += 1

        state.drowsiness_level = min(score, 3)
        state.drowsy = state.drowsiness_level >= 2

    # ── Session Stats ─────────────────────────────────────────────────────────

    def get_session_stats(self) -> dict:
        elapsed = time.time() - self.session_start
        return {
            "duration_sec":   round(elapsed, 1),
            "total_blinks":   self.total_blinks,
            "total_yawns":    self.total_yawns,
            "avg_blink_rate": round(self.total_blinks / (elapsed / 60.0 + 1e-6), 1),
        }
