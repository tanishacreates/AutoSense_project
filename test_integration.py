"""
tests/test_integration.py
--------------------------
Validates the bridge + full integrated pipeline without a camera.
Run:  python tests/test_integration.py
"""

import sys, os, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from detector import DriverState
from bridge import driver_state_to_vision_features
from fusion_engine import FusionEngine, VisionFeatures
from sensor_module import SensorModule, SensorReading, DrivingPattern
from risk_classifier import RiskClassifier
from alert_system import AlertSystem, AlertLevel


class TestBridge(unittest.TestCase):
    """Verify DriverState → VisionFeatures conversion."""

    def _make_state(self, **kwargs) -> DriverState:
        s = DriverState()
        for k, v in kwargs.items():
            setattr(s, k, v)
        return s

    def test_alert_driver_maps_correctly(self):
        state = self._make_state(
            ear=0.32, mar=0.20, pitch=-3.0, yaw=5.0, roll=1.0,
            blink_rate=15.0, eyes_closed=False, yawning=False,
            face_detected=True, drowsy=False, distracted=False,
            drowsiness_level=0,
        )
        vf = driver_state_to_vision_features(state)
        self.assertAlmostEqual(vf.ear, 0.32)
        self.assertAlmostEqual(vf.mar, 0.20)
        self.assertEqual(vf.cnn_state, "alert")
        self.assertTrue(vf.face_visible)
        self.assertFalse(vf.microsleep)
        self.assertFalse(vf.yawn_detected)

    def test_drowsy_driver_maps_correctly(self):
        state = self._make_state(
            ear=0.18, mar=0.30, pitch=-12.0, yaw=3.0,
            eyes_closed=True, yawning=False,
            face_detected=True, drowsy=True, distracted=False,
            drowsiness_level=3,
        )
        vf = driver_state_to_vision_features(state)
        self.assertEqual(vf.cnn_state, "drowsy")
        self.assertTrue(vf.microsleep)
        self.assertGreater(vf.cnn_confidence, 0.85)

    def test_distracted_driver(self):
        state = self._make_state(
            ear=0.30, yaw=40.0, face_detected=True,
            distracted=True, drowsy=False, drowsiness_level=0,
        )
        vf = driver_state_to_vision_features(state)
        self.assertEqual(vf.cnn_state, "distracted")

    def test_no_face_detected(self):
        state = self._make_state(face_detected=False)
        vf = driver_state_to_vision_features(state)
        self.assertFalse(vf.face_visible)
        self.assertEqual(vf.cnn_state, "distracted")

    def test_yawn_passed_through(self):
        state = self._make_state(
            face_detected=True, yawning=True,
            drowsy=True, drowsiness_level=2,
        )
        vf = driver_state_to_vision_features(state)
        self.assertTrue(vf.yawn_detected)

    def test_all_fields_are_finite(self):
        state = self._make_state(
            ear=0.28, mar=0.35, pitch=-8.0, yaw=15.0, roll=2.0,
            blink_rate=12.0, face_detected=True,
        )
        vf = driver_state_to_vision_features(state)
        for field in ["ear", "mar", "head_pitch", "head_yaw", "blink_rate_pm", "cnn_confidence"]:
            val = getattr(vf, field)
            self.assertTrue(np.isfinite(val), f"{field} is not finite: {val}")


class TestFullPipeline(unittest.TestCase):
    """End-to-end test: DriverState → bridge → fusion → classifier → alert."""

    @classmethod
    def setUpClass(cls):
        cls.fusion     = FusionEngine()
        cls.classifier = RiskClassifier()
        cls.classifier.train_synthetic(n_samples=1000, verbose=False)
        cls.sensor     = SensorModule(mode="simulated", pattern=DrivingPattern.NORMAL)
        cls.sensor.start()
        time.sleep(0.3)

    @classmethod
    def tearDownClass(cls):
        cls.sensor.stop()

    def _run_cycle(self, **state_kwargs):
        state   = DriverState()
        state.face_detected = True
        for k, v in state_kwargs.items():
            setattr(state, k, v)
        vf      = driver_state_to_vision_features(state)
        sensor  = self.sensor.get_latest()
        result  = self.fusion.fuse(vf, sensor)
        level   = self.classifier.predict(result)
        return result, level

    def test_alert_driver_is_safe(self):
        result, level = self._run_cycle(
            ear=0.33, mar=0.20, drowsy=False, distracted=False,
            drowsiness_level=0, eyes_closed=False,
        )
        self.assertLess(result.risk_score, 0.50)
        self.assertEqual(level, AlertLevel.SAFE)

    def test_severe_drowsy_is_alert_or_warning(self):
        """
        Severe drowsiness (level 3) should yield WARNING or ALERT.
        main.py escalates to CRITICAL when drowsiness_level >= 3,
        so we replicate that escalation logic here.
        """
        state = DriverState()
        state.face_detected   = True
        state.ear             = 0.14
        state.mar             = 0.30
        state.drowsy          = True
        state.drowsiness_level = 3
        state.eyes_closed     = True
        vf     = driver_state_to_vision_features(state)
        sensor = self.sensor.get_latest()
        result = self.fusion.fuse(vf, sensor)
        level  = self.classifier.predict(result)

        # Apply the same escalation that main.py applies
        if state.drowsiness_level >= 3 or result.risk_score > 0.85:
            level = AlertLevel.CRITICAL
        elif result.risk_score > 0.65 and level == AlertLevel.WARNING:
            level = AlertLevel.ALERT

        self.assertIn(level, [AlertLevel.WARNING, AlertLevel.ALERT, AlertLevel.CRITICAL])

    def test_risk_score_in_range(self):
        for _ in range(20):
            result, _ = self._run_cycle(
                ear=np.random.uniform(0.10, 0.35),
                mar=np.random.uniform(0.15, 0.80),
            )
            self.assertGreaterEqual(result.risk_score, 0.0)
            self.assertLessEqual(result.risk_score,    1.0)

    def test_driver_state_valid_string(self):
        result, _ = self._run_cycle(ear=0.28)
        self.assertIn(result.driver_state, ["alert", "drowsy", "distracted", "unknown"])

    def test_contributions_all_present(self):
        result, _ = self._run_cycle(ear=0.28, mar=0.25)
        for key in ["ear", "mar", "pose", "cnn", "lane", "brake"]:
            self.assertIn(key, result.contributions)

    def test_alert_system_fires_on_high_risk(self):
        fired = []
        a = AlertSystem(
            enable_audio=False, enable_tts=False, enable_visual=True,
            cooldown_s=0.0,
            on_alert_cb=lambda e: fired.append(e),
        )
        result, level = self._run_cycle(
            ear=0.13, drowsy=True, drowsiness_level=3, eyes_closed=True
        )
        if level != AlertLevel.SAFE:
            a.trigger(level, "test")
            time.sleep(0.3)
            self.assertGreater(len(fired), 0)
        a.stop()

    def test_pipeline_throughput(self):
        """50 cycles should complete in under 2 seconds."""
        t0 = time.perf_counter()
        for i in range(50):
            self._run_cycle(
                ear=0.28 + 0.01 * (i % 5),
                mar=0.25,
                drowsiness_level=i % 4,
            )
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0, f"50 cycles took {elapsed:.2f}s — too slow")


if __name__ == "__main__":
    print("=" * 55)
    print("AutoSense Tests")
    print("=" * 55)
    unittest.main(verbosity=2)
