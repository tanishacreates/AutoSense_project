"""
main.py  —  AutoSense  System
========================================
macOS threading rules:
  - Tkinter (dashboard) MUST run on the main thread
  - cv2.imshow()        MUST run on the main thread
  Solution: camera_loop runs in a background thread but never calls imshow().
            The rendered frame is placed in a shared queue.
            The main thread drains that queue in a Tkinter after() callback
            and calls cv2.imshow() there.

Run:
    python main.py                         # webcam + full UI
    python main.py --no-display            # headless terminal only
    python main.py --no-dashboard          # no Tkinter window
    python main.py --no-audio              # silence alerts
    python main.py --sensitivity high      # more aggressive detection
    python main.py --sensor-pattern drowsy # simulate drowsy sensor data
    python main.py --evaluate logs/x.csv  # post-session evaluation and exit
"""
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# FIX: import cv2 once, then immediately configure threading
import cv2
# setNumThreads is absent in opencv-python-headless on some builds — guard it
if hasattr(cv2, "setNumThreads"):
    cv2.setNumThreads(1)
if hasattr(cv2, "ocl") and hasattr(cv2.ocl, "setUseOpenCL"):
    cv2.ocl.setUseOpenCL(False)

import time
import argparse
import logging
import sys
import platform
import threading
import queue
import signal
from pathlib import Path

# ── Person 1 ──────────────────────────────────────────────────────────────────
from detector import DriverMonitor

# ── Bridge ────────────────────────────────────────────────────────────────────
from bridge import driver_state_to_vision_features

# ── Person 2 ──────────────────────────────────────────────────────────────────
from sensor_module   import SensorModule, DrivingPattern
from fusion_engine   import FusionEngine
from risk_classifier import RiskClassifier
from alert_system    import AlertSystem, AlertLevel
from dashboard       import DashboardApp, DashboardData
from logger          import EventLogger, SessionEvaluator
from display         import Dashboard
from alert           import AlertSystem as P1AlertSystem

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("main")

LOG_DIR   = Path("logs")
MODEL_DIR = Path("models")
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ── Shared state between threads ──────────────────────────────────────────────
_stop_event  = threading.Event()
_frame_queue = queue.Queue(maxsize=2)
_data_queue  = queue.Queue(maxsize=10)

# Global reference so imshow_loop can reschedule itself via Tkinter
_imshow_root = None

LEVEL_COLOURS = {
    AlertLevel.SAFE    : (50,  200,  50),
    AlertLevel.WARNING : (0,   200, 255),
    AlertLevel.ALERT   : (0,    80, 220),
    AlertLevel.CRITICAL: (0,     0, 255),
}


# ─────────────────────────────────────────────────────────────────────────────
def dashboard_loop(dashboard):
    """
    Background thread — drains _data_queue and pushes DashboardData
    into the dashboard via safe_update(). Never touches Tkinter directly.
    """
    while not _stop_event.is_set():
        try:
            data = _data_queue.get(timeout=0.05)
            dashboard.safe_update(data)
        except queue.Empty:
            pass


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="AutoSense — AI Driver Assistance System")
    p.add_argument("--camera",         type=int, default=0)
    p.add_argument("--no-display",     action="store_true", help="No OpenCV window")
    p.add_argument("--no-dashboard",   action="store_true", help="No Tkinter window")
    p.add_argument("--no-audio",       action="store_true", help="No audio / TTS")
    p.add_argument("--save-log",       action="store_true", help="Write Person 1 CSV")
    p.add_argument("--sensitivity",    choices=["low", "medium", "high"], default="medium")
    p.add_argument("--sensor-pattern",
                   choices=["normal", "highway", "city", "drowsy", "distracted"],
                   default="normal")
    p.add_argument("--evaluate", metavar="CSV",
                   help="Evaluate a saved session CSV and exit")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def draw_p2_hud(frame, result, alert_level, sensor):
    """Person 2 risk + sensor overlay (bottom-right corner)."""
    h, w   = frame.shape[:2]
    x0, y0 = w - 240, h - 130
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 8, h - 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    colour = LEVEL_COLOURS.get(alert_level, (200, 200, 200))
    font   = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"P2: {alert_level.name}", (x0 + 8, y0 + 22),
                font, 0.6, colour, 2, cv2.LINE_AA)
    for i, line in enumerate([
        f"Risk  : {result.risk_score:.3f}",
        f"Speed : {sensor.speed_kmh:.0f} km/h",
        f"Lane  : {sensor.lane_deviation:+.0f} cm",
    ]):
        cv2.putText(frame, line, (x0 + 8, y0 + 46 + i * 22),
                    font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    bx, by, bw, bh = x0 + 8, y0 + 118, 212, 6
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by),
                  (bx + int(bw * result.risk_score), by + bh), colour, -1)


# ─────────────────────────────────────────────────────────────────────────────
def camera_loop(cap, monitor, p1_display, p1_alerts,
                fusion, classifier, sensor, p2_alerts,
                event_log, dashboard, args):
    """
    Runs in a background thread.
    NEVER calls cv2.imshow() or any Tkinter method directly.
    Puts rendered frames into _frame_queue and data into _data_queue.
    """
    while not _stop_event.is_set():

        ret, frame = cap.read()
        if not ret:
            continue

        # ── Person 1 ──
        driver_state = monitor.analyze(frame)

        # FIX: gate all downstream processing on face detection
        # Without this, zero-EAR frames score as WARNING/ALERT (false positive loop)
        if not driver_state.face_detected:
            print("[INFO] Face not detected — adjust camera")
            # Still push a plain frame so the video window stays live
            try:
                _frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    _frame_queue.get_nowait()
                    _frame_queue.put_nowait(frame)
                except Exception:
                    pass
            continue

        p1_alerts.process(driver_state)
        p1_active = p1_alerts.get_active_alerts()

        # ── Bridge ──
        vision_features = driver_state_to_vision_features(driver_state)

        # ── Person 2 ──
        sensor_reading = sensor.get_latest()
        fusion_result  = fusion.fuse(vision_features, sensor_reading)
        alert_level    = classifier.predict(fusion_result)

        # ── Safety override ──
        if driver_state.drowsiness_level >= 3 or fusion_result.risk_score > 0.85:
            alert_level = AlertLevel.CRITICAL

        # ── Alerts ──
        msg_map = {
            AlertLevel.WARNING : "Stay alert — early fatigue signs",
            AlertLevel.ALERT   : "Take a break",
            AlertLevel.CRITICAL: "Pull over safely",
        }
        if alert_level != AlertLevel.SAFE:
            p2_alerts.trigger(alert_level, msg_map[alert_level])
            if dashboard:
                dashboard.log_alert(alert_level.name, msg_map[alert_level])
        else:
            p2_alerts.trigger(AlertLevel.SAFE)

        event_log.log(fusion_result, alert_level)

        # ── Build display frame ──
        display_frame = p1_display.render(frame, driver_state, p1_active)
        display_frame = p2_alerts.draw_frame(
            display_frame,
            speed_kmh    = sensor_reading.speed_kmh,
            risk_score   = fusion_result.risk_score,
            driver_state = fusion_result.driver_state,
        )
        draw_p2_hud(display_frame, fusion_result, alert_level, sensor_reading)

        # ── Push frame to display queue (drop oldest if full) ──
        try:
            _frame_queue.put_nowait(display_frame)
        except queue.Full:
            try:
                _frame_queue.get_nowait()
                _frame_queue.put_nowait(display_frame)
            except Exception:
                pass

        # ── Push data to dashboard queue ──
        try:
            _data_queue.put_nowait({
                "driver_state" : fusion_result.driver_state,
                "risk_score"   : fusion_result.risk_score,
                "alert_level"  : alert_level.name,
                "speed"        : sensor_reading.speed_kmh,
            })
        except queue.Full:
            pass

    _stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
def imshow_loop(win_name: str, interval_ms: int = 30):
    global _imshow_root

    try:
        frame = _frame_queue.get_nowait()
        cv2.imshow(win_name, frame)
    except queue.Empty:
        pass

    if not _stop_event.is_set():
        try:
            _imshow_root.after(interval_ms, lambda: imshow_loop(win_name, interval_ms))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
def main():
    global _imshow_root

    signal.signal(signal.SIGINT, lambda s, f: _stop_event.set())

    args = parse_args()

    # ── Evaluate mode ─────────────────────────────────────────────────────
    if args.evaluate:
        ev = SessionEvaluator(args.evaluate)
        ev.summary()
        ev.plot_timeline()
        return

    print("=" * 58)
    print("   AutoSense —  Integrated Driver Assistance System")
    print("=" * 58)
    print(f"  Camera        : {args.camera}")
    print(f"  Sensitivity   : {args.sensitivity}")
    print(f"  Sensor pattern: {args.sensor_pattern}")
    print(f"  Audio alerts  : {'off' if args.no_audio else 'on'}")
    print(f"  Dashboard     : {'off' if args.no_dashboard else 'on'}")
    print("=" * 58)

    # ── Build subsystems ──────────────────────────────────────────────────
    monitor    = DriverMonitor(sensitivity=args.sensitivity)
    p1_display = Dashboard()
    p1_alerts  = P1AlertSystem(save_log=args.save_log)

    pattern = DrivingPattern[args.sensor_pattern.upper()]
    sensor  = SensorModule(mode="simulated", pattern=pattern, update_hz=20)

    fusion     = FusionEngine(vision_weight=0.70, sensor_weight=0.30)
    classifier = RiskClassifier()
    model_path = str(MODEL_DIR / "risk_classifier.joblib")

    if not classifier.load(model_path):
        log.info("No model found — training classifier ...")
        classifier.train_synthetic(n_samples=4000, verbose=True)
        classifier.save(model_path)
    else:
        # FIX: version-aware compatibility check instead of dummy predict()
        if not classifier.is_compatible():
            log.warning("Saved model incompatible — retraining ...")
            classifier.train_synthetic(n_samples=4000, verbose=True)
            classifier.save(model_path)

    p2_alerts = AlertSystem(
        enable_audio  = not args.no_audio,
        enable_tts    = not args.no_audio,
        enable_visual = True,
        cooldown_s    = 8.0,
    )

    dashboard = None
    if not args.no_dashboard:
        dashboard = DashboardApp()

    ts_str    = time.strftime("%Y%m%d_%H%M%S")
    event_log = EventLogger(str(LOG_DIR / f"session_{ts_str}.csv"))

    sensor.start()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error(f"Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n[INFO] System ready.  Press Q in the video window to quit.\n")
    print(f"{'Frame':<7} {'State':<13} {'Risk':>6} {'Level':<10} {'Speed':>7}")
    print("-" * 48)

    cam_args = (cap, monitor, p1_display, p1_alerts,
                fusion, classifier, sensor, p2_alerts,
                event_log, dashboard, args)

    # ── Launch ────────────────────────────────────────────────────────────
    if platform.system() == "Darwin":
        cam_thread = threading.Thread(
            target=camera_loop, args=cam_args,
            daemon=True, name="CameraLoop"
        )
        cam_thread.start()

        if dashboard:
            data_thread = threading.Thread(
                target=dashboard_loop, args=(dashboard,),
                daemon=True, name="DashboardLoop"
            )
            data_thread.start()

        if dashboard:
            dashboard.start()

            win_name = "AutoSense"

            def _on_key(event):
                if event.keysym in ('q', 'Q', 'Escape'):
                    print("\n[INFO] Quit key pressed.")
                    _stop_event.set()
                    if dashboard._tk_root:
                        dashboard._tk_root.quit()

            # FIX: single post_build_hook that does both key binding AND imshow setup
            # Previously two separate assignments — the second always silently overwrote the first
            def _post_build():
                global _imshow_root
                _imshow_root = dashboard._tk_root
                if _imshow_root:
                    _imshow_root.bind('<Key>', _on_key)
                    if not args.no_display:
                        _imshow_root.after(200, lambda: imshow_loop(win_name))

            dashboard._post_build_hook = _post_build
            dashboard.run_blocking()

        else:
            if not args.no_display:
                win_name = "AutoSense"
                while not _stop_event.is_set():
                    try:
                        frame = _frame_queue.get(timeout=0.05)
                        cv2.imshow(win_name, frame)
                    except queue.Empty:
                        pass
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        print("\n[INFO] Quit key pressed.")
                        _stop_event.set()
                        break
            else:
                try:
                    cam_thread.join()
                except KeyboardInterrupt:
                    _stop_event.set()

        _stop_event.set()
        cam_thread.join(timeout=3.0)

    else:
        # Linux / Windows
        if dashboard:
            dashboard.start()
            data_thread = threading.Thread(
                target=dashboard_loop, args=(dashboard,),
                daemon=True, name="DashboardLoop"
            )
            data_thread.start()
            time.sleep(0.3)
        try:
            camera_loop(*cam_args)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt.")
            _stop_event.set()

    # ── Shutdown ──────────────────────────────────────────────────────────
    print("\n[INFO] Shutting down ...")
    _stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    sensor.stop()
    p2_alerts.stop()
    p1_alerts.close()
    event_log.close()
    if dashboard:
        dashboard.stop()

    print("\n" + "=" * 58)
    print("  SESSION COMPLETE")
    print("=" * 58)
    stats = monitor.get_session_stats()
    print(f"  Duration    : {stats['duration_sec']:.0f}s")
    print(f"  Blinks      : {stats['total_blinks']}")
    print(f"  Yawns       : {stats['total_yawns']}")
    print(f"  Avg blink/m : {stats['avg_blink_rate']}")

    try:
        csvs = sorted(LOG_DIR.glob("session_*.csv"))
        if csvs:
            ev = SessionEvaluator(str(csvs[-1]))
            ev.summary()
            plot_path = str(csvs[-1]).replace(".csv", "_timeline.png")
            ev.plot_timeline(plot_path)
            print(f"  Timeline plot: {plot_path}")
    except Exception as exc:
        log.warning(f"Post-session evaluation failed: {exc}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
