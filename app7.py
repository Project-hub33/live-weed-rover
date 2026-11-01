# ============================================================================
# app4.py  –  AgriBot Weed Detection Rover Dashboard
# ============================================================================
# Key design decisions:
#  • Raw YOLOv5 output ONLY – no secondary colour / plant filter.
#    (The model was trained on these exact agricultural field images;
#     re-filtering by colour destroys correct detections.)
#  • Image mode   → detect & display every detection above the confidence
#    slider, no suppression at all.
#  • Live streams → same raw YOLO output, but temporal hysteresis (N
#    consecutive frames) stops single-frame noise triggering the rover,
#    without hiding real detections on-screen.
#  • Async ESP32 HTTP calls via daemon threads – never block the video loop.
#  • Features: Image inspection, Webcam live, ESP32-CAM live.
#    (Video upload removed as requested.)
# ============================================================================

import os
import sys
import pathlib
import html as html_lib
import threading
import time
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

# ── Cross-platform PyTorch pickle fix ──────────────────────────────────────
if os.name == "nt":
    sys.modules["pathlib._local"] = pathlib
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch

# Avoid CPU oversubscription on laptops running YOLO on CPU.
try:
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 2)))
except Exception:
    pass

_MODEL_LOCK = threading.Lock()

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "weed.pt")
YOLOV5_PATH  = os.path.join(BASE_DIR, "yolov5")
IMAGE_SIZE   = 320  # Lower latency for live inference; use 416/640 if small weeds are missed.

# ============================================================================
# PAGE CONFIG & DARK THEME
# ============================================================================
st.set_page_config(
    page_title="AgriBot – Weed Detection Rover",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Persistent per-session runtime: Streamlit reruns the script after button clicks.
# Keeping thread/event/state objects in session_state prevents the UI from losing
# the live worker or reading a newly-created empty frame state on every rerun.
if "_agribot_runtime" not in st.session_state:
    st.session_state["_agribot_runtime"] = {
        "log_lock": threading.Lock(),
        "activity": [],
        "live_lock": threading.Lock(),
        "live_stop": threading.Event(),
        "worker": None,
        "live": {
            "running": False, "status": "Stopped", "frame_rgb": None,
            "count": 0, "peak": 0.0, "confirmed": False, "fps": 0.0,
            "camera_fps": 0.0, "error": "", "last_source": "",
            "command_status": "No rover command sent yet", "rover_action": "IDLE",
        },
    }
_RUNTIME = st.session_state["_agribot_runtime"]

st.markdown("""
<style>
/* ── base ── */
.stApp                        { background-color:#080d1a; color:#f1f5f9; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
[data-testid="stSidebar"]     { background-color:#0f172a; border-right:1px solid #1e293b; }
[data-testid="stHeader"]      { background:transparent; }
h1,h2,h3,h4                   { color:#f8fafc !important; font-weight:700 !important; }
[data-testid="stWidgetLabel"] p{ color:#cbd5e1 !important; }

/* ── status banner ── */
.banner {
    padding:12px 20px; border-radius:8px; margin-bottom:16px;
    display:flex; align-items:center; justify-content:space-between;
}
.banner-weed   { background:#064e3b; border:1px solid #059669; }
.banner-patrol { background:#0f1d38; border:1px solid #1e3a5f; }

/* ── metric card ── */
.mcard {
    background:#0f172a; border:1px solid #1e293b;
    border-radius:8px; padding:12px 16px; margin-bottom:10px;
}
.mcard-label { color:#94a3b8; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
.mcard-value { color:#f8fafc; font-size:22px; font-weight:700; }

/* ── log feed ── */
.logfeed { font-size:12px; color:#cbd5e1; max-height:150px; overflow-y:auto; }

/* ── top bar ── */
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
# DETECTION  –  pure YOLO, no colour filter
# ============================================================================
def run_detection(frame_bgr: np.ndarray, conf: float):
    """
    Run YOLOv5 on a BGR frame.
    Returns:
        annotated : BGR frame with green bounding boxes on every detected weed
        count     : number of detections
        peak_conf : highest confidence score (0.0 if none)
    """
    with _MODEL_LOCK:
        model.conf = float(conf)
        with torch.inference_mode():
            results = model(frame_bgr, size=IMAGE_SIZE)
    dets        = results.xyxy[0].cpu().numpy()   # [x1,y1,x2,y2,conf,cls]

    annotated  = frame_bgr.copy()
    count      = 0
    peak_conf  = 0.0
    h, w       = frame_bgr.shape[:2]

    for det in dets:
        x1, y1, x2, y2, dc, _ = det
        x1, y1, x2, y2 = (max(0, min(int(v), dim - 1))
                           for v, dim in zip([x1, y1, x2, y2], [w, h, w, h]))
        bw, bh = x2 - x1, y2 - y1
        if bw < 5 or bh < 5:
            continue
        if (bw * bh) / float(w * h) > 0.92:   # reject whole-frame boxes
            continue

        count     += 1
        peak_conf  = max(peak_conf, float(dc))

        # ── draw box + label ──────────────────────────────────────────────
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 127), 2)
        label = f"WEED {dc:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(annotated, (x1, y1 - 18), (x1 + tw + 6, y1), (0, 255, 127), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated, count, peak_conf

# ============================================================================
# ASYNC ESP32 CALLER
# ============================================================================
_LOG_LOCK = _RUNTIME["log_lock"]
_ACTIVITY = _RUNTIME["activity"]


def _log(action: str, status: str):
    with _LOG_LOCK:
        _ACTIVITY.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "status": status,
        })
        del _ACTIVITY[12:]


def render_log():
    with _LOG_LOCK:
        rows = list(_ACTIVITY[:8])
    html = '<div class="logfeed">'
    for row in rows:
        color = "#34d399" if row["status"] in ("OK", "DISPATCHED") else "#f87171"
        html += (f'<div><span style="color:#64748b">[{row["time"]}]</span> '
                 f'<b>{row["action"]}</b>: '
                 f'<span style="color:{color}">{row["status"]}</span></div>')
    return html + "</div>"


def _rover_base_url(value: str) -> str:
    """Accept a rover IP, IP:port, or base URL without duplicating the scheme."""
    raw = (value or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Enter the rover ESP32 IP shown in the rover Serial Monitor.")
    if not raw.lower().startswith(("http://", "https://")):
        raw = "http://" + raw
    parsed = urlsplit(raw)
    if not parsed.hostname:
        raise ValueError("Invalid rover address. Enter an IP such as 192.168.125.123.")
    # Rover firmware's WebServer normally uses port 80. Preserve an explicit port.
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = parsed.port
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((parsed.scheme, netloc, "", "", ""))


def _esp_worker(endpoint: str, ip: str):
    """Send a rover command and keep the real HTTP response visible in the UI."""
    try:
        base_url = _rover_base_url(ip)
    except Exception as exc:
        detail = f"ROVER ADDRESS ERROR: {exc}"
        _log(endpoint.upper(), detail)
        with _RUNTIME["live_lock"]:
            _RUNTIME["live"]["command_status"] = detail
        return
    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        response = requests.get(url, timeout=(1.2, 2.5))
        body = " ".join(response.text.split())[:100]
        status = f"HTTP {response.status_code}: {body or 'empty response'}"
        _log(endpoint.upper(), "OK" if response.status_code == 200 else status)
        # Update live status without triggering a Streamlit rerun from the worker.
        lock = _RUNTIME["live_lock"]
        with lock:
            _RUNTIME["live"]["command_status"] = status
            if response.status_code == 200:
                _RUNTIME["live"]["rover_action"] = {
                    "/red": "PATROLLING", "/green": "CUTTER + PUMP ON", "/stop": "STOPPED",
                    "/cutter_on": "CUTTER ON", "/pump_on": "PUMP ON",
                }.get(endpoint, endpoint)
    except Exception as exc:
        detail = f"UNREACHABLE: {type(exc).__name__}: {str(exc)[:100]}"
        _log(endpoint.upper(), detail)
        with _RUNTIME["live_lock"]:
            _RUNTIME["live"]["command_status"] = detail


def send_esp32(endpoint: str, ip: str):
    if ip and ip.strip():
        threading.Thread(target=_esp_worker, args=(endpoint, ip), daemon=True).start()


# ============================================================================
# CAMERA URL NORMALIZATION
# Accepts an IP, IP:port, full URL, or full MJPEG URL without duplicating :81.
# ============================================================================
def normalize_camera_url(value: str) -> str:
    """Accept an IP, IP:port, or full stream URL without duplicating ports."""
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Enter the ESP32-CAM IP address or stream URL.")
    if not raw.lower().startswith(("http://", "https://")):
        raw = "http://" + raw
    parsed = urlsplit(raw)
    if not parsed.hostname:
        raise ValueError("Invalid camera address. Example: 192.168.125.132 or http://192.168.125.132:81/stream")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid camera port in address: {value}") from exc
    port = port or 81
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    path = parsed.path if parsed.path and parsed.path != "/" else "/stream"
    return urlunsplit((parsed.scheme or "http", f"{host}:{port}", path, parsed.query, ""))


def _camera_candidates(source: str):
    """Try the configured stream first, then the common alternate ESP32-CAM port.

    Standard AI-Thinker ESP32-CAM examples often expose MJPEG at :81/stream;
    some firmwares expose the stream on port 80. We only try the same host.
    """
    first = normalize_camera_url(source)
    parsed = urlsplit(first)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = parsed.port or 81
    alternate_port = 80 if port == 81 else 81
    alternate = urlunsplit((parsed.scheme, f"{host}:{alternate_port}", parsed.path or "/stream", parsed.query, ""))
    return list(dict.fromkeys([first, alternate]))


# ============================================================================
# BACKGROUND CAMERA + INFERENCE WORKERS
# ============================================================================
_LIVE_LOCK = _RUNTIME["live_lock"]
_LIVE_STOP = _RUNTIME["live_stop"]
_LIVE = _RUNTIME["live"]


def _set_live(**kwargs):
    with _LIVE_LOCK:
        _LIVE.update(kwargs)


def _open_capture(source):
    """Open a local webcam with Windows backend fallbacks, or a URL if requested."""
    if isinstance(source, str) and source.lower().startswith(("http://", "https://")):
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
    elif os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    last_cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(source, backend)
            last_cap = cap
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                for prop_name, value in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000), ("CAP_PROP_READ_TIMEOUT_MSEC", 2000)):
                    prop = getattr(cv2, prop_name, None)
                    if prop is not None:
                        try:
                            cap.set(prop, value)
                        except Exception:
                            pass
                # Request a modest webcam resolution to reduce capture and inference lag.
                if not (isinstance(source, str) and source.lower().startswith(("http://", "https://"))):
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
    if last_cap is not None:
        last_cap.release()
    return cv2.VideoCapture()  # closed capture; caller displays a useful error


def _live_worker(source, confidence, persistence, rover_ip):
    cap = None
    pos_streak = 0
    neg_streak = 0
    confirmed = False
    # Commands are sent automatically from live detections. Do not start
    # patrol until camera frames are actually arriving and clear frames persist.
    last_cmd = None
    frames_since_fps = 0
    fps_start = time.perf_counter()
    camera_frames = 0
    camera_fps_start = time.perf_counter()
    last_infer_seq = -1
    latest_raw = None
    latest_seq = 0
    last_frame_at = 0.0
    camera_loss_stop_sent = False
    capture_lock = threading.Lock()
    capture_done = threading.Event()
    capture_error = {"message": ""}

    def reader():
        nonlocal cap, latest_raw, latest_seq, camera_frames, camera_fps_start

        retry_delay = 0.5
        is_http_stream = isinstance(source, str) and source.lower().startswith(("http://", "https://"))

        def publish_frame(frame):
            nonlocal latest_raw, latest_seq, camera_frames, camera_fps_start, last_frame_at, camera_loss_stop_sent
            if frame is None:
                return
            # Resize before inference to reduce latency and CPU load.
            h, w = frame.shape[:2]
            max_width = 480
            if w > max_width:
                new_h = max(1, int(h * max_width / w))
                frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
            with capture_lock:
                latest_raw = frame
                latest_seq += 1
                last_frame_at = time.perf_counter()
                camera_loss_stop_sent = False
            camera_frames += 1
            elapsed = time.perf_counter() - camera_fps_start
            if elapsed >= 1.0:
                _set_live(camera_fps=camera_frames / elapsed)
                camera_frames = 0
                camera_fps_start = time.perf_counter()

        while not _LIVE_STOP.is_set():
            try:
                if is_http_stream:
                    # Read MJPEG directly over HTTP. Try the configured port first,
                    # then the same host on the common alternate port (81 <-> 80).
                    last_error = None
                    got_frame_on_candidate = False
                    for candidate in _camera_candidates(str(source)):
                        if _LIVE_STOP.is_set():
                            break
                        _set_live(status=f"Connecting to {candidate}…", error="")
                        try:
                            with requests.get(
                                candidate, stream=True, timeout=(1.8, 8.0),
                                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                            ) as response:
                                response.raise_for_status()
                                content_type = response.headers.get("Content-Type", "").lower()
                                _set_live(status=f"HTTP connected ({candidate}) — waiting for JPEG frames", error="")
                                buffer = bytearray()
                                frames_before = latest_seq
                                for chunk in response.iter_content(chunk_size=2048):
                                    if _LIVE_STOP.is_set():
                                        break
                                    if not chunk:
                                        continue
                                    buffer.extend(chunk)
                                    # Parse JPEG start/end markers; multipart boundaries vary by firmware.
                                    while True:
                                        soi = buffer.find(b"\xff\xd8")
                                        if soi < 0:
                                            if len(buffer) > 1_500_000:
                                                del buffer[:-2]
                                            break
                                        eoi = buffer.find(b"\xff\xd9", soi + 2)
                                        if eoi < 0:
                                            if soi > 0:
                                                del buffer[:soi]
                                            break
                                        jpeg = bytes(buffer[soi:eoi + 2])
                                        del buffer[:eoi + 2]
                                        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                        if frame is not None:
                                            publish_frame(frame)
                                            got_frame_on_candidate = True
                                    if got_frame_on_candidate:
                                        # Stay on the working candidate until the connection drops.
                                        continue
                                if _LIVE_STOP.is_set():
                                    break
                                raise ConnectionError(f"Stream ended without continuing frames ({candidate}; content-type={content_type or 'unknown'})")
                        except Exception as candidate_exc:
                            last_error = candidate_exc
                            if got_frame_on_candidate or _LIVE_STOP.is_set():
                                break
                            continue
                    if _LIVE_STOP.is_set():
                        break
                    if last_error is not None:
                        raise ConnectionError(f"Tried {', '.join(_camera_candidates(str(source)))}; last error: {last_error}")
                    raise ConnectionError("No camera stream candidate returned video frames")
                else:
                    _set_live(status="Connecting to webcam…", error="")
                    cap = _open_capture(source)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        if isinstance(source, int):
                            raise ConnectionError(f"Cannot open webcam index {source}. Close other camera apps, check Windows camera permission, and try device index 1 or 2.")
                        raise ConnectionError(f"Cannot open video source: {source}")
                    _set_live(status="Connected — receiving webcam frames", error="")
                    while not _LIVE_STOP.is_set():
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            raise ConnectionError("Webcam frame read failed; reconnecting")
                        publish_frame(frame)
                    cap.release()
                    cap = None
                    break
            except Exception as exc:
                if _LIVE_STOP.is_set():
                    break
                message = f"Camera stream error: {exc}"
                _set_live(status="Stream interrupted — reconnecting", error=message)
                if _LIVE_STOP.wait(retry_delay):
                    break
                retry_delay = min(retry_delay * 1.5, 3.0)
            else:
                retry_delay = 0.5

        if cap is not None:
            cap.release()
        capture_done.set()

    reader_thread = threading.Thread(target=reader, name="agribot-camera-reader", daemon=True)
    reader_thread.start()

    # Keep the cutter/pump command alive independently of YOLO inference speed.
    # The supplied rover firmware turns tools off after 10 seconds unless /green is refreshed.
    def rover_heartbeat():
        last_green = time.monotonic()
        while not _LIVE_STOP.wait(0.25):
            with _LIVE_LOCK:
                weed_confirmed = bool(_LIVE.get("confirmed", False))
            now = time.monotonic()
            if weed_confirmed and now - last_green >= 3.0:
                send_esp32("/green", rover_ip)
                _log("AUTO WEED HEARTBEAT (/green)", "DISPATCHED")
                last_green = now

    heartbeat_thread = threading.Thread(target=rover_heartbeat, name="agribot-rover-heartbeat", daemon=True)
    heartbeat_thread.start()
    _set_live(running=True, status="Starting camera…", error="", last_source=str(source), fps=0.0, camera_fps=0.0)

    try:
        while not _LIVE_STOP.is_set():
            with capture_lock:
                frame = None if latest_raw is None else latest_raw.copy()
                seq = latest_seq
                frame_age = (time.perf_counter() - last_frame_at) if last_frame_at else None
                should_failsafe_stop = (frame_age is not None and frame_age > 2.5 and
                                        last_cmd in ("WEED", "PATROL") and
                                        not camera_loss_stop_sent)
                if should_failsafe_stop:
                    camera_loss_stop_sent = True
            if should_failsafe_stop:
                # Fail safe if the camera disconnects after rover commands were sent.
                send_esp32("/stop", rover_ip)
                _log("CAMERA LOST — AUTO STOP", "DISPATCHED")
                last_cmd = "STOP"
                confirmed = False
                _set_live(confirmed=False, status="Camera lost — rover stop requested")
            if frame is None or seq == last_infer_seq:
                if capture_done.is_set():
                    break
                time.sleep(0.01)
                continue
            last_infer_seq = seq
            annotated, count, peak = run_detection(frame, confidence)

            if count > 0:
                pos_streak += 1
                neg_streak = 0
                if pos_streak >= persistence:
                    confirmed = True
            else:
                neg_streak += 1
                pos_streak = 0
                if neg_streak >= persistence:
                    confirmed = False

            now = time.monotonic()
            if confirmed:
                # /green activates cutter + pump in the supplied ESP32 firmware.
                # Refresh periodically because that firmware has a 10-second tool timeout.
                if last_cmd != "WEED":
                    last_cmd = "WEED"
                    send_esp32("/green", rover_ip)
                    _log("AUTO WEED ACTION (/green)", "DISPATCHED")
            elif neg_streak >= persistence and last_cmd != "PATROL":
                # A valid clear frame starts/resumes patrol automatically.
                # No patrol command is sent when the camera is disconnected.
                was_weed = last_cmd == "WEED"
                last_cmd = "PATROL"
                send_esp32("/red", rover_ip)
                _log("AUTO PATROL" if not was_weed else "PATROL RESUMED", "DISPATCHED")

            frames_since_fps += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1.0:
                inference_fps = frames_since_fps / elapsed
                frames_since_fps = 0
                fps_start = time.perf_counter()
            else:
                with _LIVE_LOCK:
                    inference_fps = _LIVE["fps"]

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            status = "WEED CONFIRMED — CUTTING / WATERING" if confirmed else (
                "Possible weed — confirming" if count else "PATROLLING — no weed detected")
            _set_live(
                frame_rgb=rgb, count=count, peak=peak, confirmed=confirmed,
                fps=inference_fps, status=status, error=""
            )
    except Exception as exc:
        _set_live(status="Live processing error", error=str(exc))
    finally:
        _LIVE_STOP.set()
        reader_thread.join(timeout=2.5)
        heartbeat_thread.join(timeout=1.0)
        _set_live(running=False, status="Stopped", frame_rgb=None)


def start_live(source, confidence, persistence, rover_ip):
    worker = _RUNTIME.get("worker")
    if worker is not None and worker.is_alive():
        return
    _LIVE_STOP.clear()
    _set_live(running=True, status="Starting camera…", error="", frame_rgb=None,
              count=0, peak=0.0, confirmed=False, fps=0.0, camera_fps=0.0,
              last_source=str(source), command_status="Waiting for first rover command response",
              rover_action="CONNECTING")
    worker = threading.Thread(
        target=_live_worker,
        args=(source, confidence, persistence, rover_ip),
        name="agribot-live-worker", daemon=True
    )
    _RUNTIME["worker"] = worker
    worker.start()


def stop_live(rover_ip=None):
    _LIVE_STOP.set()
    _set_live(running=False, status="Stopping…")
    if rover_ip:
        send_esp32("/stop", rover_ip)


# ============================================================================
# SESSION STATE + SIDEBAR
# ============================================================================
for key, value in (("rover_state", "IDLE"),):
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.markdown("### 🛰️ Network")
    esp_ip = st.text_input(
        "ESP32 Rover IP (from rover Serial Monitor)", value="",
        placeholder="Enter the rover's actual IP, e.g. 192.168.125.xxx",
        help="This must be the rover ESP32 IP, not the ESP32-CAM IP. Both devices must be reachable from this computer.",
    )
    cam_ip = st.text_input("ESP32-CAM Stream IP", value="192.168.125.132")
    webcam_index = st.number_input("Webcam device index", min_value=0, max_value=5, value=0, step=1,
                                   help="Try 0 first. If the webcam does not open, try 1 or 2.")
    if not esp_ip.strip():
        st.warning("Enter the rover ESP32's actual IP from its Serial Monitor. Automatic commands cannot reach the rover without it.")
    st.markdown("---")
    st.markdown("### 🎯 Detection")
    conf_thr = st.slider("Confidence threshold", 0.10, 0.85, 0.25, 0.05,
                         help="0.20–0.35 often works well for field weeds.")
    persist_frames = st.slider("Live-stream persistence (frames)", 1, 8, 1,
                                help="Consecutive frames required before changing rover mode. 1 is fastest; larger values reduce false triggers.")
    st.markdown("---")
    st.markdown("### 🎮 Manual Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🛑 ALL STOP", use_container_width=True):
            stop_live(esp_ip)
            st.session_state["rover_state"] = "STOPPED"
    with c2:
        if st.button("🚜 Patrol", use_container_width=True):
            send_esp32("/red", esp_ip)
            st.session_state["rover_state"] = "PATROLLING"
    c3, c4 = st.columns(2)
    with c3:
        if st.button("🔧 Cutter ON", use_container_width=True):
            send_esp32("/cutter_on", esp_ip)
    with c4:
        if st.button("💧 Pump ON", use_container_width=True):
            send_esp32("/pump_on", esp_ip)

with _LIVE_LOCK:
    _current_live_action = _LIVE.get("rover_action", "IDLE")
rover_state_display = _current_live_action if _current_live_action not in (None, "IDLE", "CONNECTING") else st.session_state["rover_state"]
st.markdown(f"""
<div class="topbar">
  <span style="font-size:20px;font-weight:700;color:#f8fafc;">
      🌱 AgriBot – Precision Weed Detection Rover
      <span style="font-size:13px;color:#94a3b8;margin-left:10px;">YOLOv5 · {DEVICE.upper()}</span>
  </span>
  <span class="pill {'pill-weed' if rover_state_display not in ['IDLE','PATROLLING','STOPPED'] else 'pill-patrol'}">{rover_state_display}</span>
</div>
""", unsafe_allow_html=True)

mode = st.radio("Input source", ["📷 Upload Image", "🎥 Webcam (Live)", "📡 ESP32-CAM (Live)"], horizontal=True)

if mode == "📷 Upload Image":
    # If the user changes mode while streaming, stop the background worker.
    current_worker = _RUNTIME.get("worker")
    if current_worker is not None and current_worker.is_alive():
        stop_live(esp_ip)
    st.markdown("### 🖼️ Image Weed Detection")
    uploaded = st.file_uploader("Upload a field / test image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is not None:
        raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Cannot decode the uploaded image.")
        else:
            annotated, count, peak = run_detection(img, conf_thr)
            # Automatically dispatch once per uploaded image/threshold combination.
            # The rover IP must be configured; this is not a manual action button.
            import hashlib
            image_signature = hashlib.sha256(raw).hexdigest() + f":{conf_thr:.2f}"
            previous_signature = st.session_state.get("_agribot_last_image_command_signature")
            if image_signature != previous_signature:
                if esp_ip.strip():
                    endpoint = "/green" if count > 0 else "/red"
                    send_esp32(endpoint, esp_ip)
                    st.session_state["_agribot_last_image_command_signature"] = image_signature
                    _log("IMAGE AUTO ACTION " + endpoint.upper(), "DISPATCHED")
                else:
                    st.warning("Image was analyzed, but no rover command was sent. Enter the rover ESP32 IP in the sidebar to enable automatic control.")
            original_col, detect_col = st.columns(2)
            with original_col:
                st.subheader("Original")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with detect_col:
                st.subheader(f"Detections · {count} weed(s)")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Weeds found", str(count))
            m2.metric("Peak confidence", f"{peak:.1%}" if peak else "—")
            m3.metric("Rover action", "STOP · CUT · WATER" if count else "PATROL FORWARD")
            if count:
                st.success(f"{count} weed(s) detected at up to {peak:.1%} confidence.")
                st.info("Automatic image action: /green was dispatched to stop the rover and turn the cutter and pump on. The firmware tool timeout still applies unless live detection refreshes /green.")
            else:
                st.info("No weeds detected above the current threshold; /red was dispatched to start patrol if the rover IP is configured. Try lowering confidence if a weed is visible.")

else:
    st.markdown("### 📡 Live Weed Detection")
    col_vid, col_info = st.columns([3, 1])
    with col_info:
        st.markdown("#### Stream settings")
        if mode == "📡 ESP32-CAM (Live)":
            try:
                default_url = normalize_camera_url(cam_ip)
            except ValueError:
                default_url = "http://192.168.125.132:81/stream"
            stream_url = st.text_input(
                "Camera IP or MJPEG URL",
                value=default_url,
                key="agribot_stream_url_v2",
                help="Enter an IP, IP:port, or full URL. The dashboard tries the selected port and then the alternate common port (81/80) on the same IP.",
            )
            try:
                source = normalize_camera_url(stream_url)
                st.caption(f"Using stream: {source}")
            except ValueError as exc:
                source = ""
                st.error(str(exc))
        else:
            source = int(webcam_index)
            st.caption(f"Using local webcam device index {source}. If unavailable, try another index above.")
        if st.button("▶ START", type="primary", use_container_width=True):
            if source:
                start_live(source, conf_thr, persist_frames, esp_ip)
        if st.button("⏹ STOP", use_container_width=True):
            stop_live(esp_ip)
            st.session_state["rover_state"] = "STOPPED"
        st.caption("Automatic live mode: after the first valid frame, a clear frame triggers /red; a persistent weed detection triggers /green (drive stops, cutter and pump turn on). /green is refreshed while weeds remain. Camera loss requests /stop. Enter the rover ESP32 IP separately from the camera IP.")

    with col_vid:
        # Render the ESP32-CAM's continuous MJPEG stream directly in the browser.
        # This is separate from the Python reader used to feed frames to YOLO.
        # The component is intentionally OUTSIDE the fast-refreshing fragment so
        # it is not repeatedly recreated/reconnected every 150 ms.
        if mode == "📡 ESP32-CAM (Live)" and source:
            st.markdown("#### Direct camera stream")
            safe_stream_url = html_lib.escape(source, quote=True)
            components.html(
                f'''<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
html, body {{ margin:0; padding:0; background:#080d1a; overflow:hidden; }}
.stream-wrap {{ width:100%; height:420px; display:flex; align-items:center; justify-content:center; background:#020617; border:1px solid #1e293b; border-radius:8px; overflow:hidden; }}
.stream-wrap img {{ width:100%; height:100%; object-fit:contain; display:block; }}
#error {{ color:#fca5a5; font:14px sans-serif; padding:16px; text-align:center; }}
</style>
</head>
<body>
<div class="stream-wrap">
  <img src="{safe_stream_url}" alt="Live ESP32-CAM stream"
       onerror="document.getElementById('error').style.display='block';">
  <div id="error" style="display:none; position:absolute;">Browser cannot reach the camera stream URL.</div>
</div>
</body>
</html>''',
                height=430,
                scrolling=False,
            )
            st.caption("Direct camera stream from the ESP32-CAM. YOLO processing below reads frames from the same stream and draws detection boxes.")

        st.markdown("#### YOLO detection output")
        # Fragment-owned elements are recreated in the fragment context, avoiding
        # StreamlitInvalidLayoutContextError from writing into outside placeholders.
        @st.fragment(run_every=0.25)
        def render_live_panel():
            with _LIVE_LOCK:
                snapshot = dict(_LIVE)
            if snapshot["error"]:
                st.warning(snapshot["error"])
            st.markdown(f"**Detection status:** {snapshot['status']}")
            if snapshot["frame_rgb"] is not None:
                st.image(snapshot["frame_rgb"], use_container_width=True)
            else:
                st.info("Press START to connect to the camera and begin detection.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Detections", snapshot["count"])
            m2.metric("Peak confidence", f"{snapshot['peak']:.1%}" if snapshot["peak"] else "—")
            m3.metric("Inference FPS", f"{snapshot['fps']:.1f}")
            m4, m5 = st.columns(2)
            m4.metric("Camera receive FPS", f"{snapshot['camera_fps']:.1f}")
            m5.metric("Rover action", snapshot.get("rover_action", "IDLE"))
            st.caption(f"Last rover command response: {snapshot.get('command_status', 'No command yet')}")
            if snapshot["count"] == 0 and snapshot["running"]:
                st.caption("No weed box detected in this processed frame. If a weed is visible, lower confidence or check the model/training labels.")
            st.markdown("#### Activity log")
            st.markdown(render_log(), unsafe_allow_html=True)
        render_live_panel()
