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
import threading
import time
from datetime import datetime

# ── Cross-platform PyTorch pickle fix ──────────────────────────────────────
if os.name == "nt":
    sys.modules["pathlib._local"] = pathlib
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import requests
import streamlit as st
import torch

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "weed.pt")
YOLOV5_PATH  = os.path.join(BASE_DIR, "yolov5")
IMAGE_SIZE   = 640

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
    model.conf = conf
    results     = model(frame_bgr, size=IMAGE_SIZE)
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
# ASYNC ESP32 CALLER  –  daemon thread, never blocks the video loop
# ============================================================================
_LOG_LOCK = threading.Lock()
_ACTIVITY = []   # list of dicts: {time, action, status}

def _esp_worker(endpoint: str, ip: str):
    url = f"http://{ip.strip().rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        r = requests.get(url, timeout=0.9)
        ok = r.status_code == 200
        _log(endpoint.upper(), "OK" if ok else f"HTTP {r.status_code}")
    except Exception:
        _log(endpoint.upper(), "UNREACHABLE")

def send_esp32(endpoint: str, ip: str):
    if not ip or not ip.strip():
        return
    threading.Thread(target=_esp_worker, args=(endpoint, ip), daemon=True).start()

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
    for r in rows:
        c = "#34d399" if r["status"] == "OK" else "#f87171"
        html += (f'<div><span style="color:#64748b">[{r["time"]}]</span> '
                 f'<b>{r["action"]}</b>: <span style="color:{c}">{r["status"]}</span></div>')
    html += "</div>"
    return html

# ============================================================================
# SESSION STATE
# ============================================================================
for key, val in [("live_on", False), ("last_cmd", None), ("rover_state", "IDLE")]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🛰️ Network")
    esp_ip = st.text_input("ESP32 Rover IP",    value="192.168.1.105")
    cam_ip = st.text_input("ESP32-CAM Stream IP", value="192.168.1.110")

    st.markdown("---")
    st.markdown("### 🎯 Detection")
    conf_thr = st.slider(
        "Confidence threshold",
        min_value=0.10, max_value=0.85, value=0.25, step=0.05,
        help="0.20–0.35 works well for natural field weeds."
    )
    persist_frames = st.slider(
        "Live-stream persistence (frames)",
        min_value=1, max_value=8, value=3,
        help="Consecutive positive frames before triggering the rover."
    )

    st.markdown("---")
    st.markdown("### 🎮 Manual Controls")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if st.button("🛑 ALL STOP", use_container_width=True):
            send_esp32("/stop", esp_ip)
            st.session_state["rover_state"] = "STOPPED"
    with r1c2:
        if st.button("🚜 Patrol", use_container_width=True):
            send_esp32("/red", esp_ip)
            st.session_state["rover_state"] = "PATROLLING"

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        if st.button("🔧 Cutter ON", use_container_width=True):
            send_esp32("/cutter_on", esp_ip)
    with r2c2:
        if st.button("💧 Pump ON", use_container_width=True):
            send_esp32("/pump_on", esp_ip)

# ============================================================================
# TOP STATUS BAR
# ============================================================================
rover_state_display = st.session_state["rover_state"]
st.markdown(f"""
<div class="topbar">
  <span style="font-size:20px;font-weight:700;color:#f8fafc;">
      🌱 AgriBot – Precision Weed Detection Rover
      <span style="font-size:13px;color:#94a3b8;margin-left:10px;">
          YOLOv5 · {DEVICE.upper()}
      </span>
  </span>
  <span class="pill {'pill-weed' if rover_state_display not in ['IDLE','PATROLLING','STOPPED'] else 'pill-patrol'}">
      {rover_state_display}
  </span>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# INPUT MODE
# ============================================================================
mode = st.radio(
    "Input source",
    ["📷 Upload Image", "🎥 Webcam (Live)", "📡 ESP32-CAM (Live)"],
    horizontal=True,
)

# ────────────────────────────────────────────────────────────────────────────
# MODE 1 – IMAGE UPLOAD
# ────────────────────────────────────────────────────────────────────────────
if mode == "📷 Upload Image":
    st.markdown("### 🖼️ Image Weed Detection")
    uploaded = st.file_uploader(
        "Upload a field / test image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded is not None:
        raw = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if img is None:
            st.error("Cannot decode the uploaded image.")
        else:
            annotated, count, peak = run_detection(img, conf_thr)

            col_orig, col_det = st.columns(2)
            with col_orig:
                st.subheader("Original")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         use_container_width=True)
            with col_det:
                st.subheader(f"Detections  ·  {count} weed(s)")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Weeds found",      str(count))
            m2.metric("Peak confidence",  f"{peak:.1%}" if peak else "—")
            m3.metric("Rover action",
                      "🛑 STOP · CUT · WATER" if count > 0 else "🚜 PATROL FORWARD")

            if count > 0:
                st.success(f"🌱 **{count} weed(s)** confirmed at up to "
                           f"**{peak:.1%}** confidence.")
                if st.button("📡 Send Weed Action to ESP32", type="primary"):
                    send_esp32("/green", esp_ip)
                    st.session_state["rover_state"] = "CUTTING / WATERING"
                    st.info("Command dispatched to rover.")
            else:
                st.info("No weeds detected above the current threshold. "
                        "Try lowering the confidence slider if you expect weeds.")

# ────────────────────────────────────────────────────────────────────────────
# SHARED LIVE-STREAM LOGIC
# ────────────────────────────────────────────────────────────────────────────
elif mode in ["🎥 Webcam (Live)", "📡 ESP32-CAM (Live)"]:

    col_vid, col_info = st.columns([3, 1])

    with col_info:
        st.markdown("#### 📡 Stream")

        if mode == "📡 ESP32-CAM (Live)":
            default_url = f"http://{cam_ip.strip()}:81/stream"
            stream_url = st.text_input("MJPEG URL", value=default_url)
            src = stream_url.strip()
        else:
            src = 0   # default system webcam

        bc1, bc2 = st.columns(2)
        with bc1:
            start = st.button("▶ START", type="primary", use_container_width=True)
        with bc2:
            stop  = st.button("⏹ STOP",  use_container_width=True)

        if start and not st.session_state["live_on"]:
            st.session_state["live_on"]   = True
            st.session_state["last_cmd"]  = None
            st.rerun()

        if stop and st.session_state["live_on"]:
            st.session_state["live_on"]   = False
            send_esp32("/stop", esp_ip)
            st.session_state["rover_state"] = "STOPPED"
            st.rerun()

        st.markdown("---")
        st.markdown("#### 📊 Live Metrics")
        box_state = st.empty()
        box_count = st.empty()
        box_conf  = st.empty()

        st.markdown("---")
        st.markdown("#### 📜 Activity Log")
        box_log   = st.empty()

    with col_vid:
        banner     = st.empty()
        frame_slot = st.empty()

    # ── actual stream loop ──────────────────────────────────────────────────
    if st.session_state["live_on"]:
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            st.error("❌ Cannot open video source. Check camera / URL.")
            st.session_state["live_on"] = False
        else:
            pos_streak = 0   # consecutive frames with weed
            neg_streak = 0   # consecutive clear frames
            confirmed  = False

            while st.session_state["live_on"]:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.02)
                    continue

                annotated, count, peak = run_detection(frame, conf_thr)

                # ── temporal filter ────────────────────────────────────────
                if count > 0:
                    pos_streak += 1;  neg_streak = 0
                    if pos_streak >= persist_frames:
                        confirmed = True
                else:
                    neg_streak += 1;  pos_streak = 0
                    if neg_streak >= persist_frames:
                        confirmed = False

                # ── ESP32 dispatch on state change ─────────────────────────
                last = st.session_state["last_cmd"]
                if confirmed and last != "WEED":
                    st.session_state["last_cmd"]    = "WEED"
                    st.session_state["rover_state"] = "CUTTING / WATERING"
                    send_esp32("/green", esp_ip)
                elif not confirmed and last != "PATROL" and last is not None:
                    st.session_state["last_cmd"]    = "PATROL"
                    st.session_state["rover_state"] = "PATROLLING"
                    send_esp32("/red", esp_ip)

                # ── render frame ───────────────────────────────────────────
                frame_slot.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

                # ── status banner ──────────────────────────────────────────
                if confirmed:
                    banner.markdown("""
                    <div class="banner banner-weed">
                      <span style="font-size:15px;font-weight:700;color:#34d399;">
                          🌱 WEED CONFIRMED – ROVER STOPPED · CUTTING · WATERING
                      </span>
                    </div>""", unsafe_allow_html=True)
                elif count > 0:
                    banner.markdown("""
                    <div class="banner" style="background:#1a2e1a;border:1px solid #15803d;">
                      <span style="font-size:15px;font-weight:700;color:#86efac;">
                          ⚠️ Possible weed – waiting for persistent detection…
                      </span>
                    </div>""", unsafe_allow_html=True)
                else:
                    banner.markdown("""
                    <div class="banner banner-patrol">
                      <span style="font-size:15px;font-weight:700;color:#93c5fd;">
                          🚜 PATROLLING FORWARD – No weed detected
                      </span>
                    </div>""", unsafe_allow_html=True)

                # ── sidebar metrics ────────────────────────────────────────
                vc = "#34d399" if confirmed else "#94a3b8"
                box_state.markdown(f"""
                <div class="mcard">
                  <div class="mcard-label">Rover State</div>
                  <div class="mcard-value" style="color:{vc};font-size:14px;">
                    {'🛑 STOP · CUT · WATER' if confirmed else '🚜 FORWARD PATROL'}
                  </div>
                </div>""", unsafe_allow_html=True)

                box_count.markdown(f"""
                <div class="mcard">
                  <div class="mcard-label">Detections (this frame)</div>
                  <div class="mcard-value">{count}</div>
                </div>""", unsafe_allow_html=True)

                box_conf.markdown(f"""
                <div class="mcard">
                  <div class="mcard-label">Peak Confidence</div>
                  <div class="mcard-value">{peak:.1%}</div>
                </div>""", unsafe_allow_html=True)

                box_log.markdown(render_log(), unsafe_allow_html=True)

                time.sleep(0.01)   # tiny yield – keeps UI responsive at ~30 fps

            cap.release()
