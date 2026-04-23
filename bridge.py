"""
bridge.py
---------
Converts Person 1's DriverState → Person 2's VisionFeatures.
This is the single point of integration between both codebases.

Person 1 produces:  DriverState  (from core/detector.py  DriverMonitor.analyze())
Person 2 consumes:  VisionFeatures (from fusion_engine.py  FusionEngine.fuse())

Nothing in Person 1's code or Person 2's code was changed.
This file is the only glue needed.
"""

from detector import DriverState


def driver_state_to_vision_features(state: DriverState):
    """
    Map every field from DriverState to the matching VisionFeatures field.

    Field mapping table:
    ─────────────────────────────────────────────────────────────────
    DriverState field        VisionFeatures field    Notes
    ─────────────────────────────────────────────────────────────────
    state.ear                vf.ear                  direct
    state.mar                vf.mar                  direct
    state.pitch              vf.head_pitch           direct (degrees)
    state.yaw                vf.head_yaw             direct (degrees)
    state.roll               vf.roll                 direct (degrees)
    state.blink_rate         vf.blink_rate_pm        direct (per min)
    state.eyes_closed        vf.microsleep           proxy (sustained)
    state.yawning            vf.yawn_detected        direct
    state.face_detected      vf.face_visible         direct
    state.drowsy             → cnn_state="drowsy"    mapped
    state.distracted         → cnn_state="distracted"mapped
    state.drowsiness_level   → cnn_confidence        0→0.50 … 3→0.95
    state.timestamp          vf.timestamp            direct
    ─────────────────────────────────────────────────────────────────
    """

    # Lazy import so this module can be imported even without all of
    # Person 2's dependencies installed (e.g. during unit tests).
    from fusion_engine import VisionFeatures

    # CNN state + confidence from drowsiness_level
    # Level 0 = alert, 1 = mild, 2 = moderate, 3 = severe
    CONF_MAP = {0: 0.90, 1: 0.65, 2: 0.80, 3: 0.95}
    conf = CONF_MAP.get(state.drowsiness_level, 0.70)

    if not state.face_detected:
        cnn_state = "distracted"   # face not in view = distracted
        conf      = 0.70
    elif state.drowsy:
        cnn_state = "drowsy"
    elif state.distracted:
        cnn_state = "distracted"
    else:
        cnn_state = "alert"

    return VisionFeatures(
        ear             = state.ear,
        mar             = state.mar,
        head_pitch      = state.pitch,
        head_yaw        = state.yaw,
        head_roll       = state.roll,
        cnn_state       = cnn_state,
        cnn_confidence  = conf,
        blink_rate_pm   = state.blink_rate,
        microsleep      = state.eyes_closed,      # sustained eye closure
        yawn_detected   = state.yawning,
        face_visible    = state.face_detected,
        timestamp       = state.timestamp,
    )
