"""
dashboard.py — AutoSense Live Dashboard

macOS threading rules enforced:
  - start()        → marks ready only (no thread spawned on macOS)
  - run_blocking() → calls _run_tk() on the main thread (blocks)
  - _run_tk()      → calls self._post_build_hook() after Tk root is created,
                     allowing main.py to hook cv2.imshow into the event loop
  - stop()         → root.quit() via after() — never destroy() from outside
  - _on_close()    → root.quit() — never destroy()
  - _refresh_loop  → root.quit() if _running is False
  - log_alert()    → thread-safe via _alert_queue
"""
import time
import threading
import logging
import queue
import platform
from collections import deque
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    driver_state:   str   = "UNKNOWN"
    risk_score:     float = 0.0
    alert_level:    str   = "SAFE"
    ear:            float = 0.30
    mar:            float = 0.20
    head_yaw:       float = 0.0
    head_pitch:     float = 0.0
    speed_kmh:      float = 0.0
    acceleration:   float = 0.0
    lane_deviation: float = 0.0
    hard_brake:     bool  = False
    timestamp:      float = 0.0


class DashboardApp:
    REFRESH_MS  = 200
    HISTORY_LEN = 60 * 5

    COLOURS = {
        "bg"       : "#1a1a2e",
        "bg2"      : "#16213e",
        "bg3"      : "#0f3460",
        "accent"   : "#e94560",
        "text"     : "#eaeaea",
        "muted"    : "#8892a4",
        "SAFE"     : "#2ecc71",
        "WARNING"  : "#f39c12",
        "ALERT"    : "#e74c3c",
        "CRITICAL" : "#c0392b",
    }

    def __init__(self):
        self._queue       : queue.Queue[DashboardData] = queue.Queue(maxsize=50)
        self._alert_queue : queue.Queue                = queue.Queue()
        self._running     = False
        self._thread      : Optional[threading.Thread] = None
        self._tk_root     = None
        self._history     : deque = deque(maxlen=self.HISTORY_LEN)
        self._alert_log   : deque = deque(maxlen=20)
        self._current     = DashboardData(timestamp=time.time())
        self._is_macos    = platform.system() == "Darwin"

        # Optional hook called by _run_tk() after root is created.
        self._post_build_hook: Optional[Callable] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def safe_update(self, data: dict):
        """Called from any thread — converts dict to DashboardData and pushes to queue."""
        dd = DashboardData(
            driver_state = data.get("driver_state", "UNKNOWN"),
            risk_score   = data.get("risk_score", 0.0),
            alert_level  = data.get("alert_level", "SAFE"),
            speed_kmh    = data.get("speed", 0.0),
            timestamp    = time.time(),
        )
        self.update(dd)

    def start(self):
        """
        macOS  → marks as ready; caller MUST call run_blocking() from main thread.
        Others → spawns a background daemon thread.
        """
        self._running = True
        if self._is_macos:
            logger.info("Dashboard ready (macOS: call run_blocking() from main thread)")
        else:
            self._thread = threading.Thread(
                target=self._run_tk, daemon=True, name="Dashboard"
            )
            self._thread.start()
            logger.info("Dashboard started in background thread")

    def run_blocking(self):
        """macOS only — call from the MAIN thread. Blocks until window closes."""
        if self._is_macos:
            self._run_tk()

    def stop(self):
        """Signal dashboard to quit (safe to call from any thread)."""
        self._running = False
        if self._tk_root:
            try:
                self._tk_root.after(0, self._tk_root.quit)
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Dashboard stopped")

    def update(self, data: DashboardData):
        """Thread-safe data push."""
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(data)
            except Exception:
                pass

    def log_alert(self, level: str, message: str):
        """Thread-safe alert log entry."""
        ts = time.strftime("%H:%M:%S")
        try:
            self._alert_queue.put_nowait(f"[{ts}] {level}: {message}")
        except queue.Full:
            pass

    # ------------------------------------------------------------------ #
    #  Tkinter                                                             #
    # ------------------------------------------------------------------ #

    def _run_tk(self):
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError:
            logger.warning("tkinter not available — dashboard disabled")
            return

        try:
            root = tk.Tk()
            self._tk_root = root
            root.title("AutoSense")
            root.geometry("1000x680")
            root.configure(bg=self.COLOURS["bg"])
            root.resizable(True, True)
            self._build_ui(root, tk, ttk)
            root.after(self.REFRESH_MS, lambda: self._refresh_loop(root))
            root.protocol("WM_DELETE_WINDOW", self._on_close)

            # Fire the post-build hook after a short delay so the
            # window has fully appeared before imshow tries to attach
            if self._post_build_hook:
                root.after(500, self._post_build_hook)

            root.mainloop()
        except Exception as exc:
            logger.warning(f"Dashboard error: {exc}")

    def _build_ui(self, root, tk, ttk):
        C   = self.COLOURS
        BG  = C["bg"]
        BG2 = C["bg2"]

        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)

        # Header
        hdr = tk.Frame(root, bg=C["bg3"], pady=8)
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(hdr, text="🚗  AutoSense",
                 font=("Helvetica", 16, "bold"),
                 bg=C["bg3"], fg=C["text"]).pack()

        # Left panel
        left = tk.Frame(root, bg=BG, padx=12, pady=12)
        left.grid(row=1, column=0, sticky="nsew")

        sf = tk.Frame(left, bg=BG2, padx=16, pady=16)
        sf.pack(fill="x", pady=(0, 10))
        tk.Label(sf, text="DRIVER STATE",
                 font=("Helvetica", 10), bg=BG2, fg=C["muted"]).pack()
        self._state_label = tk.Label(
            sf, text="INITIALISING",
            font=("Helvetica", 28, "bold"), bg=BG2, fg=C["SAFE"]
        )
        self._state_label.pack()

        gf = tk.Frame(left, bg=BG2, padx=16, pady=12)
        gf.pack(fill="x", pady=(0, 10))
        tk.Label(gf, text="RISK SCORE",
                 font=("Helvetica", 10), bg=BG2, fg=C["muted"]).pack()
        self._risk_label = tk.Label(
            gf, text="0.000",
            font=("Helvetica", 22, "bold"), bg=BG2, fg=C["text"]
        )
        self._risk_label.pack()
        self._risk_bar = ttk.Progressbar(
            gf, length=300, mode="determinate", maximum=100
        )
        self._risk_bar.pack(fill="x", pady=4)

        vf = tk.LabelFrame(left, text="  Vision Metrics  ",
                           font=("Helvetica", 10), bg=BG, fg=C["muted"],
                           labelanchor="n", bd=1, relief="flat")
        vf.pack(fill="x", pady=(0, 10))
        self._vis_vars = self._metric_rows(
            vf, ["EAR", "MAR", "Head Yaw", "Head Pitch"], BG, tk
        )

        sf2 = tk.LabelFrame(left, text="  Sensor Metrics  ",
                            font=("Helvetica", 10), bg=BG, fg=C["muted"],
                            labelanchor="n", bd=1, relief="flat")
        sf2.pack(fill="x")
        self._sen_vars = self._metric_rows(
            sf2,
            ["Speed (km/h)", "Acceleration", "Lane Dev (cm)", "Hard Brake"],
            BG, tk
        )

        # Right panel
        right = tk.Frame(root, bg=BG, padx=12, pady=12)
        right.grid(row=1, column=1, sticky="nsew")

        tk.Label(right, text="Risk Score History",
                 font=("Helvetica", 10), bg=BG, fg=C["muted"]).pack()
        self._canvas = tk.Canvas(right, height=220, bg=BG2, highlightthickness=0)
        self._canvas.pack(fill="x", pady=(4, 10))

        tk.Label(right, text="Alert Log",
                 font=("Helvetica", 10), bg=BG, fg=C["muted"]).pack(anchor="w")
        lf = tk.Frame(right, bg=BG)
        lf.pack(fill="both", expand=True)
        sb = tk.Scrollbar(lf)
        sb.pack(side="right", fill="y")
        self._log_box = tk.Text(
            lf, bg=BG2, fg=C["text"], font=("Courier", 9),
            state="disabled", yscrollcommand=sb.set,
            relief="flat", wrap="word"
        )
        self._log_box.pack(side="left", fill="both", expand=True)
        sb.config(command=self._log_box.yview)
        for tag, col in [("ALERT",    C["ALERT"]),
                         ("WARNING",  C["WARNING"]),
                         ("CRITICAL", C["CRITICAL"]),
                         ("SAFE",     C["SAFE"])]:
            self._log_box.tag_config(tag, foreground=col)

        self._status_var = tk.StringVar(value="Ready")
        tk.Label(root, textvariable=self._status_var,
                 font=("Helvetica", 9), bg=C["bg3"], fg=C["muted"], pady=4
                 ).grid(row=2, column=0, columnspan=2, sticky="ew")

    def _metric_rows(self, parent, names, bg, tk):
        C     = self.COLOURS
        vars_ = {}
        for name in names:
            row = tk.Frame(parent, bg=bg)
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=name, width=18, anchor="w",
                     font=("Helvetica", 10), bg=bg, fg=C["muted"]).pack(side="left")
            var = tk.StringVar(value="—")
            tk.Label(row, textvariable=var,
                     font=("Helvetica", 10, "bold"),
                     bg=bg, fg=C["text"]).pack(side="right")
            vars_[name] = var
        return vars_

    # ------------------------------------------------------------------ #
    #  Refresh loop                                                        #
    # ------------------------------------------------------------------ #

    def _refresh_loop(self, root):
        if not self._running:
            root.quit()
            return

        # Drain alert queue
        while True:
            try:
                self._alert_log.appendleft(self._alert_queue.get_nowait())
            except queue.Empty:
                break

        # Drain data queue
        updated = False
        while True:
            try:
                data = self._queue.get_nowait()
                self._current = data
                self._history.append((data.timestamp, data.risk_score))
                updated = True
            except queue.Empty:
                break

        if updated:
            self._redraw(self._current)

        try:
            root.after(self.REFRESH_MS, lambda: self._refresh_loop(root))
        except Exception:
            pass

    def _redraw(self, data: DashboardData):
        C   = self.COLOURS
        col = C.get(data.alert_level, C["SAFE"])
        self._state_label.configure(text=data.driver_state.upper(), fg=col)
        self._risk_label.configure(text=f"{data.risk_score:.3f}")
        self._risk_bar["value"] = round(data.risk_score * 100, 1)

        for k, v in {
            "EAR"        : f"{data.ear:.3f}",
            "MAR"        : f"{data.mar:.3f}",
            "Head Yaw"   : f"{data.head_yaw:.1f}°",
            "Head Pitch" : f"{data.head_pitch:.1f}°",
        }.items():
            if k in self._vis_vars:
                self._vis_vars[k].set(v)

        for k, v in {
            "Speed (km/h)"  : f"{data.speed_kmh:.1f}",
            "Acceleration"  : f"{data.acceleration:+.2f} m/s²",
            "Lane Dev (cm)" : f"{data.lane_deviation:.1f}",
            "Hard Brake"    : "YES" if data.hard_brake else "no",
        }.items():
            if k in self._sen_vars:
                self._sen_vars[k].set(v)

        self._draw_chart()
        self._refresh_log()
        self._status_var.set(
            f"Last update: {time.strftime('%H:%M:%S')}  |  "
            f"State: {data.driver_state.upper()}  |  "
            f"Risk: {data.risk_score:.3f}  |  Level: {data.alert_level}"
        )

    def _draw_chart(self):
        C      = self.COLOURS
        canvas = self._canvas
        canvas.delete("all")
        w, h   = canvas.winfo_width(), canvas.winfo_height()
        if w < 10 or h < 10:
            return

        pl, pr, pt, pb = 40, 10, 10, 30
        pw = w - pl - pr
        ph = h - pt - pb

        for level, label in [(0.35, "0.35"), (0.65, "0.65"), (1.0, "1.0")]:
            y = pt + ph * (1 - level)
            canvas.create_line(pl, y, w - pr, y,
                               fill=C["bg3"], width=1, dash=(4, 4))
            canvas.create_text(pl - 4, y, text=label,
                               fill=C["muted"], font=("Helvetica", 8), anchor="e")

        if len(self._history) < 2:
            return

        hist    = list(self._history)
        t_min   = hist[0][0]
        t_range = max(hist[-1][0] - t_min, 1.0)

        def xy(ts, val):
            return (pl + ((ts - t_min) / t_range) * pw,
                    pt + ph * (1 - val))

        pts = [xy(ts, v) for ts, v in hist]
        for i in range(1, len(pts)):
            val = hist[i][1]
            col = (C["SAFE"] if val < 0.35
                   else C["WARNING"] if val < 0.65
                   else C["ALERT"])
            canvas.create_line(
                pts[i-1][0], pts[i-1][1],
                pts[i][0],   pts[i][1],
                fill=col, width=2
            )
        canvas.create_text(pl + pw // 2, h - 6,
                           text="time →", fill=C["muted"], font=("Helvetica", 8))

    def _refresh_log(self):
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        for entry in self._alert_log:
            tag = next(
                (lv for lv in ("CRITICAL", "ALERT", "WARNING", "SAFE")
                 if lv in entry), None
            )
            self._log_box.insert("end", entry + "\n", tag or "")
        self._log_box.configure(state="disabled")

    def _on_close(self):
        """User clicked the window X button."""
        self._running = False
        if self._tk_root:
            self._tk_root.quit()
