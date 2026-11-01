# ============================================================================
# app7.py  –  AgriBot Weed Detection Rover Dashboard
# ============================================================================
# Works with the ESP32 firmware "agribot_fixed.ino":
#   /red        rover moves forward (patrol)
#   /green      rover stops -> cutter 5 s -> pause 1 s -> pump 5 s -> rover
#               moves forward again BY ITSELF (the firmware does this)
#   /stop       everything off
#   /cutter_on  /pump_on   manual tools (firmware auto-stops after 10 s)
#   /status     "State: PATROLLING | STOPPED | CUTTING_WATERING"
#
# Design:
#  * ESP32-CAM starts automatically when you pick it. No START/STOP.
#  * Camera + YOLO run in a background thread. Only the NEWEST frame is ever
#    processed (old frames are dropped) so latency never builds up.
#  * Video is shown through a tiny built-in MJPEG server (port 8765), NOT by
#    re-running Streamlit for every frame. This removes the white screen and
#    gives smooth, low-latency video.
#  * AUTOMATIC: weed confirmed -> /green is sent ONCE. Detection is then paused
#    while the firmware runs its 11 s sequence + resume time. Nothing else is
#    sent (a /red or a second /green would cancel the sequence).
#  * MANUAL: detections are shown but never move the rover. Buttons control it.
# ============================================================================

import os
import sys
import pathlib
import hashlib
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── Cross-platform PyTorch pickle fix ──────────────────────────────────────
if os.name == "nt":
    sys.modules["pathlib._local"] = pathlib
    pathlib.PosixPath = pathlib.WindowsPath

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")                       # no WARN spam

import cv2
import numpy as np
import requests
import streamlit as st
import torch

try:
    cv2.setLogLevel(0)
except Exception:
    pass

try:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
except Exception:
    pass

# ── Paths / settings ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "weed.pt")
YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")
IMAGE_SIZE  = 416     # YOLO inference size (use 640 if small weeds are missed)
PROC_WIDTH  = 640     # frames wider than this are shrunk before inference
STREAM_PORT = 8765    # built-in MJPEG video port
JPEG_QUALITY = 70

# Must match the ESP32 firmware (agribot_fixed.ino)
CUTTER_SECONDS = 5.0
PAUSE_SECONDS  = 1.0
PUMP_SECONDS   = 5.0
SEQUENCE_SECONDS = CUTTER_SECONDS + PAUSE_SECONDS + PUMP_SECONDS   # 11 s

# ============================================================================
# PAGE CONFIG & DARK THEME
# ============================================================================
st.set_page_config(
    page_title="AgriBot – Weed Detection Rover",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"] { background-color:#080d1a !important; }
.stApp                        { color:#f1f5f9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
[data-testid="stSidebar"]     { background-color:#0f172a; border-right:1px solid #1e293b; }
[data-testid="stHeader"]      { background:transparent; }
h1,h2,h3,h4                   { color:#f8fafc !important; font-weight:700 !important; }
[data-testid="stWidgetLabel"] p{ color:#cbd5e1 !important; }

.banner {
    padding:12px 20px; border-radius:8px; margin-bottom:12px;
    display:flex; align-items:center; justify-content:space-between;
}
.banner-weed   { background:#064e3b; border:1px solid #059669; }
.banner-patrol { background:#0f1d38; border:1px solid #1e3a5f; }
.banner-manual { background:#3b2a05; border:1px solid #b45309; }

.mcard {
    background:#0f172a; border:1px solid #1e293b;
    border-radius:8px; padding:12px 16px; margin-bottom:10px;
}
.mcard-label { color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
.mcard-value { color:#f8fafc; font-size:22px; font-weight:700; }

.logfeed { font-size:12px; color:#cbd5e1; max-height:150px; overflow-y:auto; }

.topbar {
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 18px; background:#111e38; border:1px solid #1e3a5f;
    border-radius:8px; margin-bottom:16px;
}
.pill {
    padding:4px 12px; border-radius:9999px;
    font-size:12px; font-weight:700; letter-spacing:.4px;
}
.pill-online  { background:#064e3b; color:#34d399; border:1px solid #059669; }
.pill-offline { background:#450a0a; color:#f87171; border:1px solid #dc2626; }
.pill-patrol  { background:#1e3a8a; color:#93c5fd; border:1px solid #2563eb; }
.pill-weed    { background:#064e3b; color:#10b981; border:1px solid #059669; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOAD
# ============================================================================
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"weed.pt not found: {MODEL_PATH}")
    if not os.path.isdir(YOLOV5_PATH):
        raise FileNotFoundError(f"yolov5 dir not found: {YOLOV5_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = torch.hub.load(YOLOV5_PATH, "custom", path=MODEL_PATH,
                         source="local", device=device)
    return mdl, device

try:
    model, DEVICE = load_model()
except Exception as e:
    st.error(f"❌ Cannot load model: {e}")
    st.stop()

# ============================================================================
# MJPEG VIDEO SERVER  (video bypasses Streamlit reruns completely)
# ============================================================================
def _make_placeholder() -> bytes:
    img = np.full((360, 640, 3), (26, 17, 8), np.uint8)
    cv2.putText(img, "Waiting for camera...", (150, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (184, 163, 148), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class FrameHub:
    """Holds the newest JPEG frame and wakes every viewer when it changes."""

    def __init__(self):
        self.cv = threading.Condition()
        self.jpg = None
        self.seq = 0
        self.placeholder = _make_placeholder()

    def publish(self, bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.cv:
                self.jpg = buf.tobytes()
                self.seq += 1
                self.cv.notify_all()

    def clear(self):
        with self.cv:
            self.jpg = None
            self.seq += 1
            self.cv.notify_all()

    def wait(self, last_seq, timeout=2.0):
        with self.cv:
            if self.seq == last_seq:
                self.cv.wait(timeout)
            return (self.jpg or self.placeholder), self.seq


class _StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.split("?")[0] != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        hub = self.server.hub
        last = -1
        try:
            while True:
                # Re-sends the current frame every 2 s when idle. This also lets
                # us notice when the browser has gone away.
                jpg, last = hub.wait(last, 2.0)
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except OSError:
            pass


def start_stream_server(hub, port):
    server = ThreadingHTTPServer(("0.0.0.0", port), _StreamHandler)
    server.daemon_threads = True
    server.hub = hub
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

# ============================================================================
# CAMERA GRABBER  – always keeps only the NEWEST frame (no latency build-up)
# ============================================================================
class _Grabber(threading.Thread):
    def __init__(self, source, stop_event):
        super().__init__(daemon=True)
        self.source = source
        self.stop_event = stop_event
        self.cancel = threading.Event()
        self.cv = threading.Condition()
        self.frame = None
        self.seq = 0
        self.error = None
        self.finished = False

    def _should_stop(self):
        return self.stop_event.is_set() or self.cancel.is_set()

    def run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot connect to the ESP32-CAM stream: {self.source}. "
                    "Check the ESP32-CAM IP and that this URL plays in a browser.")

            fails = 0
            while not self._should_stop():
                ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails >= 50:
                        raise RuntimeError("Camera stream stopped returning frames.")
                    time.sleep(0.02)
                    continue
                fails = 0
                with self.cv:
                    self.frame = frame
                    self.seq += 1
                    self.cv.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            if cap is not None:
                cap.release()
            self.finished = True
            with self.cv:
                self.cv.notify_all()

    def latest(self, last_seq, timeout=0.5):
        with self.cv:
            if (self.frame is None or self.seq == last_seq) and not self.finished:
                self.cv.wait(timeout)
            return self.frame, self.seq

# ============================================================================
# CONTROLLER  – one shared instance (cache_resource) for the whole app
# ============================================================================
class Controller:
    def __init__(self, model):
        self.model = model
        self.model_lock = threading.Lock()
        self.lock = threading.RLock()
        self.hub = FrameHub()
        self.stream_error = None

        self.cfg = {"esp_ip": "", "conf": 0.25, "persistence": 2,
                    "cooldown": 3.0, "mode": "AUTOMATIC"}

        self.src_key = None
        self.stop_event = None
        self.thread = None
        self.snap = self._blank_snap()

        self.treat_started = 0.0      # when the firmware sequence began
        self.treat_until = 0.0        # detection stays paused until this time
        self.cmd_block_until = 0.0    # stops duplicate sends while one is in flight

        self.rover = {"online": False, "state": "UNKNOWN"}
        self.activity = []

        threading.Thread(target=self._poll_status, daemon=True).start()

    # ── small helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _blank_snap():
        return {"count": 0, "peak": 0.0, "fps": 0.0, "error": None, "has_frame": False}

    def configure(self, **kw):
        with self.lock:
            self.cfg.update(kw)

    def get_cfg(self):
        with self.lock:
            return dict(self.cfg)

    def get_mode(self):
        with self.lock:
            return self.cfg["mode"]

    def set_mode(self, mode):
        with self.lock:
            self.cfg["mode"] = mode

    def snapshot(self):
        with self.lock:
            return dict(self.snap)

    def get_rover(self):
        with self.lock:
            return dict(self.rover)

    def treatment_phase(self):
        """None when idle, else (phase_name, seconds_left) – timer estimate."""
        now = time.monotonic()
        with self.lock:
            if now >= self.treat_until:
                return None
            el = now - self.treat_started
        if el < CUTTER_SECONDS:
            return "CUTTING", CUTTER_SECONDS - el
        if el < CUTTER_SECONDS + PAUSE_SECONDS:
            return "PAUSE", CUTTER_SECONDS + PAUSE_SECONDS - el
        if el < SEQUENCE_SECONDS:
            return "WATERING", SEQUENCE_SECONDS - el
        return "RESUMING", 0.0

    def treatment_active(self):
        return self.treatment_phase() is not None

    # ── activity log (lives here so background threads can write to it) ────
    def log(self, action, status):
        with self.lock:
            self.activity.insert(0, {
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": action, "status": status})
            del self.activity[12:]

    def render_log(self):
        with self.lock:
            rows = list(self.activity[:8])
        html = '<div class="logfeed">'
        for r in rows:
            good = r["status"] == "OK"
            c = "#34d399" if good else "#f87171"
            html += (f'<div><span style="color:#64748b">[{r["time"]}]</span> '
                     f'<b>{r["action"]}</b>: <span style="color:{c}">{r["status"]}</span></div>')
        return html + "</div>"

    # ── ESP32 commands ──────────────────────────────────────────────────────
    def _send(self, endpoint, ip):
        if not ip or not ip.strip():
            self.log(endpoint.upper(), "NO IP")
            return False
        url = f"http://{ip.strip().rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            r = requests.get(url, timeout=(0.5, 1.2))
            ok = r.status_code == 200
            self.log(endpoint.upper(), "OK" if ok else f"HTTP {r.status_code}")
            return ok
        except requests.RequestException as exc:
            self.log(endpoint.upper(), f"UNREACHABLE: {type(exc).__name__}")
            return False

    def command(self, endpoint, sync=False):
        """Send one command. Async by default so video/detection never wait."""
        ip = self.get_cfg()["esp_ip"]

        def run():
            ok = self._send(endpoint, ip)
            with self.lock:
                if ok:
                    if endpoint == "/green":
                        now = time.monotonic()
                        self.treat_started = now
                        # sequence + 1 s safety + user re-detect delay
                        self.treat_until = (now + SEQUENCE_SECONDS + 1.0
                                            + self.cfg["cooldown"])
                    else:
                        # /red /stop /cutter_on /pump_on all cancel a running sequence
                        self.treat_until = 0.0
                else:
                    self.cmd_block_until = time.monotonic() + 2.0   # retry in 2 s
            return ok

        if sync:
            return run()
        threading.Thread(target=run, daemon=True).start()
        return True

    def _poll_status(self):
        while True:
            ip = self.get_cfg()["esp_ip"].strip()
            if ip:
                try:
                    r = requests.get(f"http://{ip}/status", timeout=(0.5, 1.0))
                    state = "UNKNOWN"
                    for line in r.text.splitlines():
                        if line.startswith("State:"):
                            state = line.split(":", 1)[1].strip()
                    with self.lock:
                        self.rover = {"online": r.status_code == 200, "state": state}
                except requests.RequestException:
                    with self.lock:
                        self.rover = {"online": False, "state": "OFFLINE"}
            else:
                with self.lock:
                    self.rover = {"online": False, "state": "NO IP"}
            time.sleep(1.5)

    # ── detection ───────────────────────────────────────────────────────────
    def detect(self, frame_bgr, conf):
        """Raw YOLOv5 output. Returns (annotated_bgr, count, peak_conf)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # YOLOv5 expects RGB
        with self.model_lock:
            self.model.conf = conf
            with torch.inference_mode():
                results = self.model(rgb, size=IMAGE_SIZE)
        dets = results.xyxy[0].cpu().numpy()

        annotated = frame_bgr.copy()
        count, peak = 0, 0.0
        h, w = frame_bgr.shape[:2]
        for det in dets:
            x1, y1, x2, y2, dc, _ = det
            x1, y1, x2, y2 = (max(0, min(int(v), dim - 1))
                              for v, dim in zip([x1, y1, x2, y2], [w, h, w, h]))
            bw, bh = x2 - x1, y2 - y1
            if bw < 5 or bh < 5:
                continue
            if (bw * bh) / float(w * h) > 0.92:     # reject whole-frame boxes
                continue
            count += 1
            peak = max(peak, float(dc))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 127), 2)
            label = f"WEED {dc:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(annotated, (x1, y1 - 18), (x1 + tw + 6, y1), (0, 255, 127), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated, count, peak

    # ── live source management (auto-start, no buttons) ────────────────────
    def ensure_source(self, kind, target):
        key = (kind, target)
        with self.lock:
            alive = self.thread is not None and self.thread.is_alive()
            if alive and self.src_key == key:
                return
        self.stop_source()
        source = str(target)
        stop = threading.Event()
        with self.lock:
            self.src_key = key
            self.stop_event = stop
            self.snap = self._blank_snap()
            self.thread = threading.Thread(
                target=self._run, args=(source, stop), daemon=True)
            self.thread.start()

    def stop_source(self):
        with self.lock:
            ev, th = self.stop_event, self.thread
        if ev is not None:
            ev.set()
        if th is not None and th.is_alive():
            th.join(timeout=3.0)
        with self.lock:
            self.thread = None
            self.stop_event = None
            self.src_key = None
            self.snap = self._blank_snap()
        self.hub.clear()

    def _run(self, source, stop):
        last_err = None
        while not stop.is_set():
            grab = _Grabber(source, stop)
            grab.start()
            try:
                self._infer_loop(grab, stop)
            except Exception as exc:
                msg = str(exc)
                with self.lock:
                    self.snap = {**self._blank_snap(), "error": msg}
                self.hub.clear()
                if msg != last_err:
                    self.log("CAMERA", f"ERROR: {msg[:50]}")
                    last_err = msg
            finally:
                grab.cancel.set()
            if not stop.is_set():
                stop.wait(5.0)          # auto-reconnect after a pause

    def _infer_loop(self, grab, stop):
        seq = -1
        positive = 0
        frames, t_fps, fps = 0, time.monotonic(), 0.0

        while not stop.is_set():
            frame, new_seq = grab.latest(seq, 0.5)
            if frame is None or new_seq == seq:
                if grab.finished:
                    raise RuntimeError(grab.error or "Camera closed.")
                continue
            seq = new_seq

            cfg = self.get_cfg()
            h, w = frame.shape[:2]
            if w > PROC_WIDTH:
                frame = cv2.resize(frame, (PROC_WIDTH, max(1, int(h * PROC_WIDTH / w))),
                                   interpolation=cv2.INTER_AREA)

            annotated, count, peak = self.detect(frame, cfg["conf"])
            now = time.monotonic()

            frames += 1
            if now - t_fps >= 1.0:
                fps, frames, t_fps = frames / (now - t_fps), 0, now

            # ── automatic control ──────────────────────────────────────────
            with self.lock:
                treating = now < self.treat_until
                blocked = now < self.cmd_block_until
            automatic = cfg["mode"] == "AUTOMATIC"

            if not automatic or treating:
                positive = 0            # fresh confirmation needed afterwards
            elif count > 0:
                positive += 1
            else:
                positive = 0

            if automatic and not treating and not blocked and positive >= cfg["persistence"]:
                positive = 0
                with self.lock:
                    self.cmd_block_until = now + 3.0
                self.command("/green")  # ONCE – firmware runs the whole sequence

            self.hub.publish(annotated)
            with self.lock:
                self.snap = {"count": count, "peak": peak, "fps": fps,
                             "error": None, "has_frame": True}

    def run_image_action(self, endpoint):
        self.command(endpoint)


@st.cache_resource
def get_controller():
    ctrl = Controller(model)
    try:
        start_stream_server(ctrl.hub, STREAM_PORT)
    except OSError as exc:
        ctrl.stream_error = f"Cannot start video server on port {STREAM_PORT}: {exc}"
    return ctrl

controller = get_controller()

# ============================================================================
# CALLBACKS  (run before the page re-runs, so they may change widget state)
# ============================================================================
def _ip():
    return st.session_state.get("esp_ip", "").strip()


def cb_mode_change():
    mode = st.session_state["control_mode_radio"]
    controller.configure(esp_ip=_ip())
    controller.set_mode(mode)
    if mode == "AUTOMATIC" and not controller.treatment_active():
        controller.command("/red")      # resume patrol after manual / all-stop


def cb_all_stop():
    controller.configure(esp_ip=_ip())
    controller.set_mode("MANUAL")       # automatic must NOT restart the rover
    st.session_state["control_mode_radio"] = "MANUAL"
    ok = controller.command("/stop", sync=True)
    st.session_state["flash"] = (
        ("success", "ALL STOP sent. Switched to MANUAL mode.") if ok else
        ("error", "ESP32 did not answer /stop. Check the IP and Wi-Fi. "
                  "Mode is MANUAL, so no automatic commands will be sent."))


def cb_command(endpoint, label):
    controller.configure(esp_ip=_ip())
    ok = controller.command(endpoint, sync=True)
    st.session_state["flash"] = (
        ("success", f"{label} sent.") if ok else
        ("warning", f"ESP32 did not respond to {endpoint}. Check the IP and Wi-Fi."))

# ============================================================================
# SIDEBAR
# ============================================================================
st.session_state.setdefault("control_mode_radio", controller.get_mode())

with st.sidebar:
    st.markdown("### 🛰️ Network")
    esp_ip = st.text_input("ESP32 Rover IP", value="192.168.1.105", key="esp_ip")
    cam_ip = st.text_input("ESP32-CAM Stream IP", value="192.168.1.110", key="cam_ip")

    st.markdown("---")
    st.markdown("### 🎯 Detection")
    conf_thr = st.slider(
        "Confidence threshold", min_value=0.10, max_value=0.85,
        value=0.25, step=0.05, key="conf_thr",
        help="0.20–0.35 works well for natural field weeds.")
    persist_frames = st.slider(
        "Frames to confirm a weed", min_value=1, max_value=8, value=2, key="persist",
        help="Consecutive frames with a weed before the rover is triggered. "
             "Lower = faster reaction, higher = fewer false alarms.")
    cooldown_s = st.slider(
        "Wait after treatment (s)", min_value=0, max_value=10, value=3, key="cooldown",
        help="After the 11 s cut+water sequence the rover drives forward. "
             "Detection stays paused this long so the same weed is not treated twice.")

    st.markdown("---")
    st.markdown("### 🤖 Control Mode")
    st.radio(
        "Operating mode", ["AUTOMATIC", "MANUAL"], key="control_mode_radio",
        on_change=cb_mode_change,
        help="AUTOMATIC: detections drive the rover. MANUAL: detections are only "
             "shown; use the buttons below.")

    st.markdown("---")
    st.markdown("### 🎮 Manual Controls")
    manual_ok = st.session_state["control_mode_radio"] == "MANUAL"

    st.button("🛑 ALL STOP", width="stretch", on_click=cb_all_stop)

    c1, c2 = st.columns(2)
    with c1:
        st.button("🚜 Patrol", width="stretch", disabled=not manual_ok,
                  on_click=cb_command, args=("/red", "Patrol"))
        st.button("🔧 Cutter ON", width="stretch", disabled=not manual_ok,
                  on_click=cb_command, args=("/cutter_on", "Cutter ON"))
    with c2:
        st.button("🌿 Cut+Water", width="stretch", disabled=not manual_ok,
                  on_click=cb_command, args=("/green", "Cut + Water sequence"))
        st.button("💧 Pump ON", width="stretch", disabled=not manual_ok,
                  on_click=cb_command, args=("/pump_on", "Pump ON"))
    if not manual_ok:
        st.caption("Switch to MANUAL to use the buttons. ALL STOP always works.")

    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])

# Push the latest settings to the background worker (no restart needed).
controller.configure(esp_ip=esp_ip.strip(), conf=conf_thr,
                     persistence=persist_frames, cooldown=float(cooldown_s))

# ============================================================================
# TOP STATUS BAR  (small fragment: only text refreshes, never the video)
# ============================================================================
@st.fragment(run_every=1.0)
def render_topbar():
    rover = controller.get_rover()
    phase = controller.treatment_phase()
    mode = controller.get_mode()
    if not rover["online"]:
        pill_cls, pill_txt = "pill-offline", "ESP32 OFFLINE"
    elif phase and phase[0] != "RESUMING":
        pill_cls, pill_txt = "pill-weed", f"{phase[0]} · {phase[1]:.0f}s"
    elif rover["state"] == "CUTTING_WATERING":
        pill_cls, pill_txt = "pill-weed", "CUTTING / WATERING"
    else:
        pill_cls, pill_txt = "pill-patrol", rover["state"]
    st.markdown(f"""
    <div class="topbar">
      <span style="font-size:20px;font-weight:700;color:#f8fafc;">
          🌱 AgriBot – Precision Weed Detection Rover
          <span style="font-size:13px;color:#94a3b8;margin-left:10px;">
              YOLOv5 · {DEVICE.upper()} · {mode}
          </span>
      </span>
      <span class="pill {pill_cls}">{pill_txt}</span>
    </div>""", unsafe_allow_html=True)

render_topbar()

# ============================================================================
# INPUT MODE
# ============================================================================
mode = st.radio(
    "Input source",
    ["📷 Upload Image", "📡 ESP32-CAM (Live)"],
    horizontal=True, key="input_source")

# ────────────────────────────────────────────────────────────────────────────
# MODE 1 – IMAGE UPLOAD
# ────────────────────────────────────────────────────────────────────────────
if mode == "📷 Upload Image":
    controller.stop_source()            # release the camera

    st.markdown("### 🖼️ Image Weed Detection")
    uploaded = st.file_uploader(
        "Upload a field / test image",
        type=["jpg", "jpeg", "png", "bmp", "webp"])

    if uploaded is not None:
        raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Cannot decode the uploaded image.")
        else:
            annotated, count, peak = controller.detect(img, conf_thr)

            col_orig, col_det = st.columns(2)
            with col_orig:
                st.subheader("Original")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
            with col_det:
                st.subheader(f"Detections  ·  {count} weed(s)")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width="stretch")

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Weeds found", str(count))
            m2.metric("Peak confidence", f"{peak:.1%}" if peak else "—")
            m3.metric("Rover action",
                      "🛑 STOP · CUT · WATER" if count > 0 else "🚜 PATROL FORWARD")

            automatic = controller.get_mode() == "AUTOMATIC"
            if automatic:
                # Send once per image / action / IP; reruns do not repeat it.
                image_id = hashlib.sha256(raw).hexdigest()
                action = "WEED" if count > 0 else "PATROL"
                key = f"{image_id}:{action}:{esp_ip.strip()}"
                if st.session_state.get("last_image_action_key") != key:
                    st.session_state["last_image_action_key"] = key
                    if count > 0:
                        controller.command("/green")
                    elif not controller.treatment_active():
                        controller.command("/red")   # never cancel a running sequence

            if count > 0:
                st.success(
                    f"🌱 **{count} weed(s)** found at up to **{peak:.1%}** confidence. "
                    + ("Cut + water command sent to the rover." if automatic
                       else "MANUAL mode: no command sent."))
            else:
                st.info("No weeds detected above the current threshold. "
                        + ("Patrol command sent. " if automatic else "MANUAL mode: no command sent. ")
                        + "Try lowering the confidence slider if you expect weeds.")

# ────────────────────────────────────────────────────────────────────────────
# MODE 2 – ESP32-CAM LIVE (starts automatically, no START / STOP buttons)
# ────────────────────────────────────────────────────────────────────────────
else:
    col_vid, col_info = st.columns([3, 1])

    with col_info:
        st.markdown("#### 📡 Source")
        url = st.text_input("MJPEG URL", value=f"http://{cam_ip.strip()}:81/stream")
        kind, target = "esp32cam", url.strip()

    # Starts the camera + detector (or keeps it running if nothing changed).
    controller.ensure_source(kind, target)

    @st.fragment(run_every=0.5)
    def render_banner():
        snap = controller.snapshot()
        phase = controller.treatment_phase()
        if snap["error"]:
            st.error(f"{snap['error']}  –  retrying automatically…")
        elif phase:
            name, left = phase
            detail = ("rover moving forward again" if name == "RESUMING"
                      else f"{name} · {left:.0f}s left")
            st.markdown(f"""
            <div class="banner banner-weed"><span style="font-size:15px;font-weight:700;color:#34d399;">
              🌱 WEED CONFIRMED – {detail}
            </span></div>""", unsafe_allow_html=True)
        elif controller.get_mode() == "MANUAL":
            st.markdown("""
            <div class="banner banner-manual"><span style="font-size:15px;font-weight:700;color:#fbbf24;">
              🎮 MANUAL MODE – detections are shown, the rover is controlled by the buttons
            </span></div>""", unsafe_allow_html=True)
        elif snap["has_frame"]:
            st.markdown("""
            <div class="banner banner-patrol"><span style="font-size:15px;font-weight:700;color:#93c5fd;">
              🚜 PATROLLING FORWARD – watching for weeds
            </span></div>""", unsafe_allow_html=True)
        else:
            st.info("Starting camera and detector…")

    @st.fragment(run_every=0.5)
    def render_metrics():
        snap = controller.snapshot()
        rover = controller.get_rover()
        phase = controller.treatment_phase()
        st.markdown("#### 📊 Live Metrics")
        state_txt = rover["state"] if rover["online"] else "ESP32 OFFLINE"
        color = "#34d399" if phase else ("#f87171" if not rover["online"] else "#94a3b8")
        for label, value, style in [
            ("Rover State (from ESP32)", state_txt, f"color:{color};font-size:14px;"),
            ("Detections (this frame)", str(snap["count"]), ""),
            ("Peak Confidence", f"{snap['peak']:.1%}", ""),
            ("Processing FPS", f"{snap['fps']:.1f}", ""),
        ]:
            st.markdown(f"""
            <div class="mcard"><div class="mcard-label">{label}</div>
            <div class="mcard-value" style="{style}">{value}</div></div>""",
                        unsafe_allow_html=True)
        st.markdown("#### 📜 Activity Log")
        st.markdown(controller.render_log(), unsafe_allow_html=True)

    with col_vid:
        render_banner()
        if controller.stream_error:
            st.error(controller.stream_error)
        try:
            host = st.context.headers.get("Host", "localhost")
            if ":" in host and not host.endswith("]"):
                host = host.rsplit(":", 1)[0]
        except Exception:
            host = "localhost"
        stream_url = f"http://{host}:{STREAM_PORT}/stream"
        st.markdown(
            f'<img src="{stream_url}" alt="live video" '
            f'style="width:100%;border-radius:8px;border:1px solid #1e293b;'
            f'background:#0f172a;min-height:240px;">',
            unsafe_allow_html=True)
        st.caption(f"Video not showing? Open [{stream_url}]({stream_url}) in a new tab.")

    with col_info:
        render_metrics()
