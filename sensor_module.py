"""
sensor_module.py
----------------
Handles vehicle sensor data: speed, braking, acceleration, lane deviation.

Supports two modes:
  - SIMULATED: generates realistic driving patterns for development/testing
  - REAL (OBD-II): reads from an ELM327 OBD-II adapter via pyserial/obd library

Usage:
    sensor = SensorModule(mode="simulated")
    sensor.start()
    data = sensor.get_latest()
    sensor.stop()
"""

import time
import math
import random
import threading
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DrivingPattern(Enum):
    """Simulated driving scenario patterns."""
    NORMAL      = "normal"
    HIGHWAY     = "highway"
    CITY        = "city"
    DROWSY      = "drowsy"       # irregular speed, late braking
    DISTRACTED  = "distracted"   # sudden lane changes, speed variance


@dataclass
class SensorReading:
    """Snapshot of vehicle sensor data at a point in time."""
    timestamp:          float = 0.0
    speed_kmh:          float = 0.0      # vehicle speed  (km/h)
    acceleration:       float = 0.0      # longitudinal   (m/s²), positive = accel, negative = brake
    lateral_accel:      float = 0.0      # lateral g-force (m/s²)
    brake_pressure:     float = 0.0      # 0-100 %
    steering_angle:     float = 0.0      # degrees, 0 = straight
    lane_deviation:     float = 0.0      # cm from lane centre
    turn_signal_on:     bool  = False
    rpm:                float = 0.0      # engine RPM
    throttle_pct:       float = 0.0      # 0-100 %

    # Derived risk flags (computed by fusion engine, stored here for convenience)
    hard_brake:         bool  = False    # |acceleration| > 3.5 m/s²  during braking
    sudden_swerve:      bool  = False    # |lateral_accel| > 2.0 m/s²
    lane_departure:     bool  = False    # |lane_deviation| > 30 cm  without signal

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SensorModule:
    """
    Central sensor manager.  Thread-safe; call get_latest() from any thread.

    Parameters
    ----------
    mode        : "simulated" or "obd"
    pattern     : DrivingPattern used in simulation mode
    update_hz   : How often (Hz) the background thread refreshes data
    """

    # Thresholds for derived risk flags
    HARD_BRAKE_THRESHOLD    = 3.5   # m/s²
    SUDDEN_SWERVE_THRESHOLD = 2.0   # m/s²
    LANE_DEPARTURE_CM       = 30    # cm

    def __init__(
        self,
        mode: str = "simulated",
        pattern: DrivingPattern = DrivingPattern.NORMAL,
        update_hz: float = 10.0,
    ):
        self.mode        = mode
        self.pattern     = pattern
        self.update_hz   = update_hz
        self._interval   = 1.0 / update_hz

        self._reading    = SensorReading()
        self._lock       = threading.Lock()
        self._running    = False
        self._thread: Optional[threading.Thread] = None

        # Simulation state
        self._sim_speed  = 50.0   # km/h starting speed
        self._sim_time   = 0.0

        # OBD connection (lazy import so the lib is optional)
        self._obd_conn   = None

        logger.info(f"SensorModule initialised  mode={mode}  pattern={pattern.value}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self):
        """Start background sensor polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("SensorModule started")

    def stop(self):
        """Stop background thread and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._obd_conn:
            self._obd_conn.close()
        logger.info("SensorModule stopped")

    def get_latest(self) -> SensorReading:
        """Return the most recent sensor snapshot (thread-safe copy)."""
        with self._lock:
            r = SensorReading(**self._reading.__dict__)
        return r

    def inject_event(self, event: str):
        """
        Manually inject a driving event for testing.
        event: "hard_brake" | "swerve" | "lane_depart" | "speed_spike"
        """
        with self._lock:
            if event == "hard_brake":
                self._reading.acceleration   = -5.0
                self._reading.brake_pressure = 90.0
            elif event == "swerve":
                self._reading.lateral_accel  = 3.5
                self._reading.steering_angle = 35.0
            elif event == "lane_depart":
                self._reading.lane_deviation = 45.0
            elif event == "speed_spike":
                self._reading.speed_kmh      = 140.0
        logger.debug(f"Injected event: {event}")

    # ------------------------------------------------------------------ #
    #  Internal loop                                                       #
    # ------------------------------------------------------------------ #

    def _run_loop(self):
        while self._running:
            t0 = time.perf_counter()
            try:
                if self.mode == "simulated":
                    reading = self._simulate()
                elif self.mode == "obd":
                    reading = self._read_obd()
                else:
                    reading = SensorReading(timestamp=time.time())

                self._compute_flags(reading)

                with self._lock:
                    self._reading = reading

            except Exception as exc:
                logger.warning(f"Sensor read error: {exc}")

            elapsed = time.perf_counter() - t0
            sleep_for = max(0, self._interval - elapsed)
            time.sleep(sleep_for)

    # ------------------------------------------------------------------ #
    #  Simulation engine                                                   #
    # ------------------------------------------------------------------ #

    def _simulate(self) -> SensorReading:
        """
        Generate a realistic sensor reading based on the chosen DrivingPattern.
        Uses sine waves + noise to mimic real driving variability.
        """
        self._sim_time += self._interval
        t = self._sim_time

        pattern = self.pattern

        # --- base speed profile ---
        if pattern == DrivingPattern.HIGHWAY:
            target_speed = 110 + 10 * math.sin(t / 30) + random.gauss(0, 2)
        elif pattern == DrivingPattern.CITY:
            # stop-start cycle ~every 45 s
            cycle = (t % 45) / 45
            target_speed = max(0, 40 * math.sin(math.pi * cycle) + random.gauss(0, 3))
        elif pattern == DrivingPattern.DROWSY:
            # gradual speed drift + micro-corrections
            target_speed = 70 + 15 * math.sin(t / 40) + random.gauss(0, 5)
        elif pattern == DrivingPattern.DISTRACTED:
            # erratic speed with occasional spikes
            target_speed = 80 + random.gauss(0, 12)
        else:  # NORMAL
            target_speed = 60 + 8 * math.sin(t / 20) + random.gauss(0, 2)

        target_speed = max(0.0, min(160.0, target_speed))
        accel = (target_speed - self._sim_speed) * 0.15   # simple P-controller
        self._sim_speed += accel * self._interval * 3.6    # integrate

        # --- steering & lane deviation ---
        if pattern == DrivingPattern.DROWSY:
            lane_dev = 10 * math.sin(t / 8) + random.gauss(0, 4)
            steering = 3 * math.sin(t / 8) + random.gauss(0, 1)
        elif pattern == DrivingPattern.DISTRACTED:
            lane_dev = random.gauss(0, 15)
            steering = random.gauss(0, 8)
        else:
            lane_dev = 5 * math.sin(t / 15) + random.gauss(0, 2)
            steering = 2 * math.sin(t / 15) + random.gauss(0, 0.5)

        # --- brake & throttle ---
        brake = max(0.0, -accel * 8) + random.uniform(0, 2)
        throttle = max(0.0, accel * 8) + random.uniform(0, 3)
        brake    = min(100.0, brake)
        throttle = min(100.0, throttle)

        # --- lateral accel (from cornering) ---
        lat_accel = abs(steering) * 0.04 + random.gauss(0, 0.1)
        if pattern == DrivingPattern.DISTRACTED and random.random() < 0.02:
            lat_accel += random.uniform(2, 4)   # sudden swerve event

        # --- RPM estimate ---
        rpm = 800 + self._sim_speed * 25 + random.gauss(0, 50)

        return SensorReading(
            timestamp      = time.time(),
            speed_kmh      = round(self._sim_speed, 1),
            acceleration   = round(accel * 3.6 / self._interval * 0.05, 2),   # m/s²
            lateral_accel  = round(lat_accel, 2),
            brake_pressure = round(brake, 1),
            steering_angle = round(steering, 1),
            lane_deviation = round(lane_dev, 1),
            turn_signal_on = random.random() < 0.03,
            rpm            = round(rpm),
            throttle_pct   = round(throttle, 1),
        )

    # ------------------------------------------------------------------ #
    #  OBD-II real hardware                                               #
    # ------------------------------------------------------------------ #

    def _read_obd(self) -> SensorReading:
        """
        Read from a real ELM327 OBD-II adapter.
        Requires:  pip install obd
        Connect the OBD-II dongle to the vehicle's port and set
        the correct serial port in the constructor (default auto-detect).

        Falls back to simulation if the connection fails.
        """
        try:
            import obd

            if self._obd_conn is None or not self._obd_conn.is_connected():
                self._obd_conn = obd.OBD()   # auto-detect port
                if not self._obd_conn.is_connected():
                    logger.warning("OBD-II not connected — falling back to simulation")
                    return self._simulate()

            speed_cmd  = self._obd_conn.query(obd.commands.SPEED)
            rpm_cmd    = self._obd_conn.query(obd.commands.RPM)
            throttle_cmd = self._obd_conn.query(obd.commands.THROTTLE_POS)

            speed = float(speed_cmd.value.magnitude)  if not speed_cmd.is_null()   else 0.0
            rpm   = float(rpm_cmd.value.magnitude)    if not rpm_cmd.is_null()     else 0.0
            thr   = float(throttle_cmd.value.magnitude) if not throttle_cmd.is_null() else 0.0

            return SensorReading(
                timestamp    = time.time(),
                speed_kmh    = round(speed, 1),
                rpm          = round(rpm),
                throttle_pct = round(thr, 1),
                # Acceleration / braking require IMU — estimated here
                acceleration = (speed - self._sim_speed) / self._interval,
            )

        except ImportError:
            logger.warning("obd library not installed — using simulation")
            return self._simulate()
        except Exception as exc:
            logger.warning(f"OBD read failed ({exc}) — using simulation")
            return self._simulate()

    # ------------------------------------------------------------------ #
    #  Derived flag computation                                            #
    # ------------------------------------------------------------------ #

    def _compute_flags(self, r: SensorReading):
        """Set boolean risk flags based on thresholds."""
        r.hard_brake    = r.acceleration < -self.HARD_BRAKE_THRESHOLD
        r.sudden_swerve = r.lateral_accel > self.SUDDEN_SWERVE_THRESHOLD
        r.lane_departure = (
            abs(r.lane_deviation) > self.LANE_DEPARTURE_CM
            and not r.turn_signal_on
        )


# ------------------------------------------------------------------ #
#  Standalone test                                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Sensor Module Test ===")
    for pattern in DrivingPattern:
        sensor = SensorModule(mode="simulated", pattern=pattern, update_hz=10)
        sensor.start()
        time.sleep(0.5)
        r = sensor.get_latest()
        sensor.stop()
        print(f"\n[{pattern.value}]")
        print(f"  Speed       : {r.speed_kmh:.1f} km/h")
        print(f"  Acceleration: {r.acceleration:.2f} m/s²")
        print(f"  Lane dev    : {r.lane_deviation:.1f} cm")
        print(f"  Hard brake  : {r.hard_brake}")
        print(f"  Swerve      : {r.sudden_swerve}")
