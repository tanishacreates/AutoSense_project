"""
tests/test_pipeline.py
-----------------------
Unit tests for Person 2's pipeline components.

Run with:  python -m pytest tests/test_pipeline.py -v
Or:        python tests/test_pipeline.py
"""

import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensor_module   import SensorModule, SensorReading, DrivingPattern
from fusion_engine   import FusionEngine, VisionFeatures, FusionResult
from risk_classifier import RiskClassifier, extract_features, FEATURE_NAMES
from alert_system    import AlertSystem, AlertLevel, VisualOverlay


# ------------------------------------------------------------------ #
#  SensorModule tests                                                   #
# ------------------------------------------------------------------ #

class TestSensorModule(unittest.TestCase):

    def test_start_stop(self):
        s = SensorModule(mode="simulated", pattern=DrivingPattern.NORMAL)
        s.start()
        time.sleep(0.2)
        r = s.get_latest()
        s.stop()
        self.assertIsInstance(r, SensorReading)

    def test_all_patterns_produce_readings(self):
        for pattern in DrivingPattern:
            s = SensorModule(mode="simulated", pattern=pattern)
            s.start()
            time.sleep(0.3)
            r = s.get_latest()
            s.stop()
            self.assertGreaterEqual(r.speed_kmh, 0, f"Pattern {pattern.value} gave negative speed")

    def test_derived_flags_hard_brake(self):
        s = SensorModule(mode="simulated")
        s.start()
        time.sleep(0.1)
        s.inject_event("hard_brake")
        time.sleep(0.15)
        r = s.get_latest()
        s.stop()
        self.assertEqual(r.hard_brake, True)
        self.assertLess(r.acceleration, -3.0)

    def test_derived_flags_swerve(self):
        s = SensorModule(mode="simulated")
        s.start()
        time.sleep(0.1)
        s.inject_event("swerve")
        time.sleep(0.15)
        r = s.get_latest()
        s.stop()
        self.assertEqual(r.sudden_swerve, True)

    def test_lane_departure_flag_without_signal(self):
        s = SensorModule(mode="simulated")
        s.start()
        time.sleep(0.1)
        s.inject_event("lane_depart")
        time.sleep(0.15)
        r = s.get_latest()
        s.stop()
        self.assertGreater(abs(r.lane_deviation), 30)

    def test_thread_safety(self):
        """Multiple threads reading simultaneously should not crash."""
        import threading
        s = SensorModule(mode="simulated")
        s.start()
        errors = []

        def reader():
            for _ in range(20):
                try:
                    s.get_latest()
                    time.sleep(0.01)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s.stop()
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")


# ------------------------------------------------------------------ #
#  FusionEngine tests                                                   #
# ------------------------------------------------------------------ #

class TestFusionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = FusionEngine()
        self.sensor = SensorReading(
            speed_kmh=70, acceleration=0, lane_deviation=3,
            hard_brake=False, sudden_swerve=False, timestamp=time.time()
        )

    def test_alert_driver_low_risk(self):
        vis = VisionFeatures(ear=0.32, mar=0.20, cnn_state="alert", cnn_confidence=0.95)
        r   = self.engine.fuse(vis, self.sensor)
        self.assertLess(r.risk_score, 0.40)
        self.assertEqual(r.driver_state, "alert")

    def test_microsleep_high_risk(self):
        vis = VisionFeatures(ear=0.14, microsleep=True, cnn_state="drowsy", cnn_confidence=0.90)
        r   = self.engine.fuse(vis, self.sensor)
        self.assertGreater(r.risk_score, 0.50)
        self.assertEqual(r.driver_state, "drowsy")

    def test_distracted_driver(self):
        vis = VisionFeatures(ear=0.29, head_yaw=50, cnn_state="distracted", cnn_confidence=0.85)
        r   = self.engine.fuse(vis, self.sensor)
        self.assertGreater(r.risk_score, 0.30)

    def test_hard_brake_raises_risk(self):
        vis   = VisionFeatures(ear=0.30, cnn_state="alert", cnn_confidence=0.95)
        safe_sensor  = SensorReading(speed_kmh=70, hard_brake=False, timestamp=time.time())
        brake_sensor = SensorReading(speed_kmh=70, hard_brake=True,  acceleration=-5.0, timestamp=time.time())
        r_safe  = self.engine.fuse(vis, safe_sensor)
        r_brake = self.engine.fuse(vis, brake_sensor)
        self.assertGreater(r_brake.risk_score, r_safe.risk_score)

    def test_face_not_visible_high_risk(self):
        vis = VisionFeatures(face_visible=False)
        r   = self.engine.fuse(vis, self.sensor)
        self.assertGreater(r.risk_score, 0.50)

    def test_risk_score_range(self):
        for _ in range(50):
            vis = VisionFeatures(
                ear  = np.random.uniform(0.05, 0.40),
                mar  = np.random.uniform(0.10, 0.90),
                head_yaw = np.random.uniform(-60, 60),
            )
            r = self.engine.fuse(vis, self.sensor)
            self.assertGreaterEqual(r.risk_score, 0.0)
            self.assertLessEqual(r.risk_score, 1.0)

    def test_contributions_present(self):
        vis = VisionFeatures()
        r   = self.engine.fuse(vis, self.sensor)
        for key in ["ear", "mar", "pose", "cnn", "blink", "lane", "brake", "swerve", "speed"]:
            self.assertIn(key, r.contributions)


# ------------------------------------------------------------------ #
#  RiskClassifier tests                                                 #
# ------------------------------------------------------------------ #

class TestRiskClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Train once for the whole test class."""
        cls.clf = RiskClassifier()
        cls.clf.train_synthetic(n_samples=1000, verbose=False)
        cls.engine = FusionEngine()
        cls.sensor = SensorReading(speed_kmh=70, timestamp=time.time())

    def _make_result(self, risk: float, state: str, ear: float = 0.28) -> FusionResult:
        vis = VisionFeatures(
            ear=ear, cnn_state=state, cnn_confidence=0.85
        )
        r = self.engine.fuse(vis, self.sensor)
        # Override smoothed risk with a direct value for testing
        r.risk_score = risk
        return r

    def test_safe_prediction(self):
        r     = self._make_result(0.10, "alert", ear=0.32)
        level = self.clf.predict(r)
        self.assertEqual(level, AlertLevel.SAFE)

    def test_alert_prediction(self):
        r     = self._make_result(0.82, "drowsy", ear=0.13)
        level = self.clf.predict(r)
        self.assertIn(level, [AlertLevel.WARNING, AlertLevel.ALERT])

    def test_predict_proba_sums_to_one(self):
        r     = self._make_result(0.50, "drowsy")
        proba = self.clf.predict_proba(r)
        total = sum(proba.values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_predict_proba_keys(self):
        r     = self._make_result(0.30, "alert")
        proba = self.clf.predict_proba(r)
        self.assertEqual(set(proba.keys()), {"SAFE", "WARNING", "ALERT"})

    def test_feature_extraction_length(self):
        r      = self._make_result(0.40, "alert")
        feats  = extract_features(r)
        self.assertEqual(len(feats), len(FEATURE_NAMES))

    def test_feature_extraction_no_nan(self):
        r     = self._make_result(0.40, "alert")
        feats = extract_features(r)
        self.assertFalse(np.any(np.isnan(feats)))

    def test_metrics_after_training(self):
        metrics = self.clf.get_metrics()
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.70)

    def test_save_load(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            self.clf.save(path)
            clf2 = RiskClassifier()
            loaded = clf2.load(path)
            self.assertTrue(loaded)
            r     = self._make_result(0.10, "alert", ear=0.32)
            level = clf2.predict(r)
            self.assertIsInstance(level, AlertLevel)
        finally:
            os.unlink(path)


# ------------------------------------------------------------------ #
#  AlertSystem tests                                                    #
# ------------------------------------------------------------------ #

class TestAlertSystem(unittest.TestCase):

    def setUp(self):
        self.fired = []
        self.alert = AlertSystem(
            enable_audio  = False,
            enable_tts    = False,
            enable_visual = True,
            cooldown_s    = 0.1,
            on_alert_cb   = lambda e: self.fired.append(e),
        )

    def tearDown(self):
        self.alert.stop()

    def test_trigger_warning(self):
        self.alert.trigger(AlertLevel.WARNING, "Test warning")
        time.sleep(0.3)
        levels = [e.level for e in self.fired]
        self.assertIn(AlertLevel.WARNING, levels)

    def test_trigger_alert(self):
        self.alert.trigger(AlertLevel.ALERT, "Test alert")
        time.sleep(0.3)
        levels = [e.level for e in self.fired]
        self.assertIn(AlertLevel.ALERT, levels)

    def test_safe_clears_level(self):
        self.alert.trigger(AlertLevel.WARNING, "X")
        time.sleep(0.2)
        self.alert.trigger(AlertLevel.SAFE)
        time.sleep(0.2)
        self.assertEqual(self.alert.get_current_level(), AlertLevel.SAFE)

    def test_cooldown_prevents_spam(self):
        """Two rapid triggers of same level should fire at most once (within cooldown)."""
        # Use a long cooldown
        self.alert.cooldown_s = 5.0
        self.alert.trigger(AlertLevel.WARNING, "First")
        self.alert.trigger(AlertLevel.WARNING, "Second (should be blocked)")
        time.sleep(0.5)
        warning_count = sum(1 for e in self.fired if e.level == AlertLevel.WARNING)
        self.assertLessEqual(warning_count, 1)

    def test_draw_frame_returns_ndarray(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.alert.trigger(AlertLevel.WARNING, "overlay test")
        time.sleep(0.2)
        out = self.alert.draw_frame(frame, speed_kmh=70, risk_score=0.4, driver_state="drowsy")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, frame.shape)


# ------------------------------------------------------------------ #
#  VisualOverlay tests                                                  #
# ------------------------------------------------------------------ #

class TestVisualOverlay(unittest.TestCase):

    def setUp(self):
        self.overlay = VisualOverlay()
        self.frame   = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_draw_returns_same_shape(self):
        for level in AlertLevel:
            out = self.overlay.draw(self.frame, level, "test")
            self.assertEqual(out.shape, self.frame.shape)

    def test_draw_does_not_modify_input(self):
        original = self.frame.copy()
        self.overlay.draw(self.frame, AlertLevel.ALERT, "test")
        np.testing.assert_array_equal(self.frame, original)

    def test_hud_overlay(self):
        out = self.overlay.draw_sensor_hud(self.frame, 80.0, 0.35, "drowsy")
        self.assertEqual(out.shape, self.frame.shape)


# ------------------------------------------------------------------ #
#  Integration smoke test                                               #
# ------------------------------------------------------------------ #

class TestIntegration(unittest.TestCase):
    """End-to-end smoke test of the full pipeline (no camera, no UI)."""

    def test_full_pipeline_one_cycle(self):
        sensor     = SensorModule(mode="simulated", pattern=DrivingPattern.NORMAL)
        engine     = FusionEngine()
        classifier = RiskClassifier()
        classifier.train_synthetic(n_samples=500, verbose=False)
        alerts     = AlertSystem(
            enable_audio=False, enable_tts=False, enable_visual=True
        )

        sensor.start()
        time.sleep(0.3)

        for _ in range(5):
            vis    = VisionFeatures(ear=0.28, cnn_state="alert", cnn_confidence=0.9)
            sen    = sensor.get_latest()
            result = engine.fuse(vis, sen)
            level  = classifier.predict(result)

            self.assertIsInstance(result.risk_score, float)
            self.assertIsInstance(level, AlertLevel)
            self.assertIn(result.driver_state, ["alert", "drowsy", "distracted", "unknown"])

            alerts.trigger(level, "smoke test")
            time.sleep(0.1)

        sensor.stop()
        alerts.stop()


# ------------------------------------------------------------------ #
#  Runner                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("="*55)
    print("  DRIVER ASSISTANCE SYSTEM — UNIT TESTS")
    print("="*55)
    unittest.main(verbosity=2)
