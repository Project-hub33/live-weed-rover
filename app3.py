import hashlib
import os
import pathlib
import sys
import tempfile
import threading
import time

if os.name == "nt":
    sys.modules["pathlib._local"] = pathlib
    pathlib.PosixPath = pathlib.WindowsPath

import cv2
import numpy as np
import requests
import streamlit as st
import torch


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weed.pt")
YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")
IMAGE_SIZE = 640
MODEL_LOCK = threading.Lock()

st.set_page_config(
    page_title="Weed Detection and Pump Control",
    page_icon="🌱",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e5e7eb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] {
        color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌱 Weed Detection & ESP32 Pump")
st.caption(
    "Run continuous detection from your computer webcam or ESP32-CAM, "
    "or analyze an uploaded image or video."
)


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.isdir(YOLOV5_PATH):
        raise FileNotFoundError(f"YOLOv5 folder not found: {YOLOV5_PATH}")

    return torch.hub.load(
        YOLOV5_PATH,
        "custom",
        path=MODEL_PATH,
        source="local",
        device="cpu",
    )


def detect(frame, model, confidence):
    with MODEL_LOCK:
        model.conf = confidence
        results = model(frame, size=IMAGE_SIZE)
        detections = results.xyxy[0].cpu().numpy()
        annotated = results.render()[0]

    highest_confidence = 0.0
    for detection in detections:
        detection_confidence = float(detection[4])
        class_id = int(detection[5])
        class_name = model.names[class_id]
        if "weed" in str(class_name).lower():
            highest_confidence = max(highest_confidence, detection_confidence)

    return annotated, highest_confidence


def stabilize_detection(candidate, state):
    if candidate:
        state["positive"] += 1
        state["negative"] = 0
        if state["positive"] >= 5:
            state["confirmed"] = True
    else:
        state["negative"] += 1
        state["positive"] = 0
        if state["negative"] >= 5:
            state["confirmed"] = False

    return state["confirmed"]


def pump_command(weed_detected, controller_ip, on_path, off_path):
    endpoint = on_path if weed_detected else off_path
    endpoint = endpoint.strip()
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    url = f"http://{controller_ip.strip().rstrip('/')}{endpoint}"

    try:
        response = requests.get(url, timeout=2.0)
        response.raise_for_status()
    except requests.RequestException as error:
        return False, f"Pump command failed at {url}: {error}"

    return True, f"ESP32 replied with HTTP {response.status_code} at {url}."


def close_live_camera(send_off=True):
    camera = st.session_state.get("live_camera")
    if camera is not None:
        camera.release()
        st.session_state["live_camera"] = None

    if send_off and st.session_state.get("live_active"):
        config = st.session_state.get("live_config", {})
        succeeded, message = pump_command(
            False,
            config.get("controller_ip", ""),
            config.get("pump_on_path", "/weed"),
            config.get("pump_off_path", "/no_weed"),
        )
        st.session_state["live_command_message"] = message
        st.session_state["live_command_error"] = (
            None if succeeded else message
        )

    st.session_state["live_active"] = False
    st.session_state["live_source"] = None


with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "Detection confidence",
        min_value=0.35,
        max_value=0.95,
        value=0.65,
        step=0.05,
    )
    st.subheader("ESP32 pump controller")
    controller_ip = st.text_input(
        "Pump controller IP",
        value="192.168.1.105",
        help="IP of the ESP32 that controls the pump; separate from the camera IP.",
    )
    pump_on_path = st.text_input("Pump ON endpoint", value="/weed")
    pump_off_path = st.text_input("Pump OFF endpoint", value="/no_weed")
    st.subheader("ESP32-CAM")
    camera_ip = st.text_input(
        "ESP32-CAM IP",
        value="192.168.1.110",
        help="IP shown by the ESP32-CAM after it connects to Wi-Fi.",
    )
    camera_stream_url = st.text_input(
        "ESP32-CAM live stream URL",
        value=f"http://{camera_ip.strip()}:81/stream",
        help="Typical ESP32-CAM MJPEG stream endpoint is port 81 at /stream.",
    )
    st.caption(
        "The ESP32 pump firmware must implement the configured HTTP endpoints. "
        "The camera and pump controller may be different devices."
    )
    st.warning(
        "This model has one weed class only. It cannot explicitly distinguish "
        "weeds from people or reflections."
    )

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the weed detection model: {error}")
    st.stop()

input_mode = st.radio(
    "Input source",
    [
        "Webcam (live)",
        "ESP32-CAM (live)",
        "Upload image",
        "Upload video",
    ],
    horizontal=True,
)

live_modes = {"Webcam (live)", "ESP32-CAM (live)"}
if (
    st.session_state.get("live_active")
    and st.session_state.get("live_source") != input_mode
):
    close_live_camera()
if (
    not st.session_state.get("live_active")
    and st.session_state.get("live_command_error")
):
    st.error(st.session_state["live_command_error"])


def render_live_detection():
    if not st.session_state.get("live_active"):
        return

    camera = st.session_state.get("live_camera")
    config = st.session_state.get("live_config", {})
    if camera is None:
        st.session_state["live_error"] = (
            "Live camera is not available. Stop and start the stream again."
        )
        close_live_camera()
        st.rerun()

    frame_read, frame = camera.read()
    if not frame_read:
        st.session_state["live_error"] = (
            "Could not read a live frame. Check the camera and stream URL."
        )
        close_live_camera()
        st.rerun()

    annotated, highest_confidence = detect(
        frame,
        model,
        config["confidence"],
    )
    detection_state = st.session_state["live_detection_state"]
    weed_detected = stabilize_detection(
        highest_confidence > 0,
        detection_state,
    )
    last_command = st.session_state.get("live_last_command")
    now = time.monotonic()
    last_attempt = st.session_state.get("live_last_attempt_at", 0.0)

    if (
        last_command is not weed_detected
        and now - last_attempt >= 2.0
    ):
        st.session_state["live_last_attempt_at"] = now
        succeeded, message = pump_command(
            weed_detected,
            config["controller_ip"],
            config["pump_on_path"],
            config["pump_off_path"],
        )
        st.session_state["live_command_message"] = message
        if succeeded:
            st.session_state["live_last_command"] = weed_detected
            st.session_state["live_command_error"] = None
        else:
            st.session_state["live_command_error"] = message

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Live detection",
        use_container_width=True,
    )

    if weed_detected:
        st.success(f"WEED DETECTED — {highest_confidence:.1%}. Pump ON.")
    elif highest_confidence > 0:
        st.warning(
            f"Possible weed — {highest_confidence:.1%}. "
            "Waiting for 5 consecutive frames before turning the pump on."
        )
    else:
        st.info("No weed confirmed. Pump is OFF.")

    message = st.session_state.get("live_command_message")
    if message:
        if "failed" in message.lower():
            st.error(message)
        else:
            st.caption(message)


if input_mode in live_modes:
    if not st.session_state.get("live_active"):
        if st.session_state.get("live_error"):
            st.error(st.session_state["live_error"])

        if input_mode == "Webcam (live)":
            st.info(
                "Uses webcam device 0 on the computer running Streamlit."
            )
        else:
            st.info(f"ESP32-CAM stream: `{camera_stream_url}`")

        if st.button("Start live detection", type="primary"):
            if not controller_ip.strip():
                st.error("Enter the pump controller IP before starting.")
                st.stop()
            if not pump_on_path.strip() or not pump_off_path.strip():
                st.error("Configure both pump endpoints before starting.")
                st.stop()

            if input_mode == "Webcam (live)":
                camera = cv2.VideoCapture(0)
            else:
                camera = cv2.VideoCapture(camera_stream_url.strip())

            if not camera.isOpened():
                camera.release()
                st.session_state["live_error"] = (
                    "Could not open the selected live camera. Check its IP, "
                    "stream URL, Wi-Fi connection, and camera availability."
                )
                st.error(st.session_state["live_error"])
            else:
                succeeded, message = pump_command(
                    False,
                    controller_ip,
                    pump_on_path,
                    pump_off_path,
                )
                st.session_state["live_camera"] = camera
                st.session_state["live_error"] = None
                st.session_state["live_active"] = True
                st.session_state["live_source"] = input_mode
                st.session_state["live_config"] = {
                    "confidence": confidence,
                    "controller_ip": controller_ip,
                    "pump_on_path": pump_on_path,
                    "pump_off_path": pump_off_path,
                }
                st.session_state["live_detection_state"] = {
                    "positive": 0,
                    "negative": 0,
                    "confirmed": False,
                }
                st.session_state["live_last_command"] = (
                    False if succeeded else None
                )
                st.session_state["live_last_attempt_at"] = time.monotonic()
                st.session_state["live_command_message"] = message
                st.session_state["live_command_error"] = (
                    None if succeeded else message
                )
                st.rerun()
    else:
        st.session_state["live_config"] = {
            "confidence": confidence,
            "controller_ip": controller_ip,
            "pump_on_path": pump_on_path,
            "pump_off_path": pump_off_path,
        }
        if st.button("Stop live detection"):
            close_live_camera()
            st.rerun()

        @st.fragment(run_every=0.2)
        def live_detection_fragment():
            render_live_detection()

        live_detection_fragment()

elif input_mode == "Upload image":
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="uploaded_image",
    )

    current_fingerprint = None
    if uploaded_image is not None:
        current_fingerprint = hashlib.sha256(
            uploaded_image.getvalue() + f"|{confidence:.2f}".encode("utf-8")
        ).hexdigest()

    if uploaded_image is not None and st.button(
        "Analyze image and update pump",
        type="primary",
    ):
        image_bytes = uploaded_image.getvalue()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            st.error("Could not read the selected image.")
        elif not controller_ip.strip():
            st.error("Enter the pump controller IP before analyzing.")
        elif not pump_on_path.strip() or not pump_off_path.strip():
            st.error("Configure both pump endpoints before analyzing.")
        else:
            annotated, highest_confidence = detect(image, model, confidence)
            weed_detected = highest_confidence > 0
            succeeded, message = pump_command(
                weed_detected,
                controller_ip,
                pump_on_path,
                pump_off_path,
            )
            st.session_state["image_result"] = {
                "fingerprint": current_fingerprint,
                "original": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                "annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                "confidence": highest_confidence,
                "weed_detected": weed_detected,
                "command_succeeded": succeeded,
                "message": message,
            }

    result = st.session_state.get("image_result")
    if result and result["fingerprint"] != current_fingerprint:
        result = None
    if result:
        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(result["original"], use_container_width=True)
        with right:
            st.subheader("Detection")
            st.image(result["annotated"], use_container_width=True)

        if result["weed_detected"]:
            if result["command_succeeded"]:
                st.success(
                    f"WEED DETECTED — {result['confidence']:.1%}. Pump ON."
                )
            else:
                st.error(
                    f"WEED DETECTED — {result['confidence']:.1%}, but the "
                    "pump command failed."
                )
        elif result["command_succeeded"]:
            st.info("No weed detected above the threshold. Pump OFF.")
        else:
            st.error("No weed detected, but the pump OFF command failed.")
        st.caption(result["message"])

elif input_mode == "Upload video":
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="uploaded_video",
    )
    if uploaded_video is not None and st.button(
        "Analyze video and control pump",
        type="primary",
    ):
        if not controller_ip.strip():
            st.error("Enter the pump controller IP before analyzing.")
        elif not pump_on_path.strip() or not pump_off_path.strip():
            st.error("Configure both pump endpoints before analyzing.")
        else:
            temporary_path = None
            capture = None
            try:
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(uploaded_video.getvalue())
                    temporary_path = temporary_file.name

                capture = cv2.VideoCapture(temporary_path)
                if not capture.isOpened():
                    st.error("Could not open the uploaded video.")
                else:
                    state = {
                        "positive": 0,
                        "negative": 0,
                        "confirmed": False,
                    }
                    last_command = None
                    preview = st.empty()
                    status = st.empty()
                    progress = st.progress(0)
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_number = 0
                    peak_confidence = 0.0

                    while True:
                        frame_read, frame = capture.read()
                        if not frame_read:
                            break

                        annotated, highest_confidence = detect(
                            frame,
                            model,
                            confidence,
                        )
                        peak_confidence = max(
                            peak_confidence,
                            highest_confidence,
                        )
                        weed_detected = stabilize_detection(
                            highest_confidence > 0,
                            state,
                        )
                        if weed_detected != last_command:
                            succeeded, message = pump_command(
                                weed_detected,
                                controller_ip,
                                pump_on_path,
                                pump_off_path,
                            )
                            if not succeeded:
                                st.error(message)
                            last_command = weed_detected

                        frame_number += 1
                        if frame_number % 5 == 0:
                            preview.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                caption=f"Video frame {frame_number}",
                                use_container_width=True,
                            )
                            status.info(
                                f"Frames processed: {frame_number} | "
                                f"Pump: {'ON' if weed_detected else 'OFF'}"
                            )
                            if total_frames > 0:
                                progress.progress(
                                    min(frame_number / total_frames, 1.0)
                                )

                    succeeded, message = pump_command(
                        False,
                        controller_ip,
                        pump_on_path,
                        pump_off_path,
                    )
                    if succeeded:
                        status.warning(
                            "Video ended; pump OFF command sent for safety."
                        )
                    else:
                        status.error(message)
                    progress.progress(1.0)
                    st.caption(message)
                    st.success(
                        f"Video complete. Highest weed confidence: "
                        f"{peak_confidence:.1%}."
                    )
            except OSError as error:
                st.error(f"Could not process the uploaded video: {error}")
            finally:
                if capture is not None:
                    capture.release()
                if temporary_path is not None and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
