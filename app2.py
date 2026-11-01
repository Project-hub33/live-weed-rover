# ============================================================
# app2.py
# WEED DETECTION ROVER
# ============================================================

# ============================================================
# IMPORTANT WINDOWS / YOLOv5 FIX
# MUST BE FIRST
# ============================================================

import os
import sys
import pathlib

if os.name == "nt":

    # Fix pathlib module used while unpickling
    sys.modules["pathlib._local"] = pathlib

    # Linux PosixPath -> WindowsPath
    pathlib.PosixPath = pathlib.WindowsPath


# ============================================================
# IMPORTS
# ============================================================

import cv2
import torch
import numpy as np
import requests
import streamlit as st


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "weed.pt"
)

YOLOV5_PATH = os.path.join(
    BASE_DIR,
    "yolov5"
)


# ============================================================
# ESP32
# ============================================================

ESP32_IP = "192.168.1.105"


# ============================================================
# YOLO
# ============================================================

IMAGE_SIZE = 640


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Weed Detection Rover",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }

    .title {
        font-size: 38px;
        font-weight: bold;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stHeader"] {
        color: #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title">🌱 Weed Detection Rover</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")


confidence = st.sidebar.slider(
    "🎯 Detection Confidence",
    min_value=0.35,
    max_value=0.95,
    value=0.65,
    step=0.05
)


st.sidebar.write(
    f"Threshold: **{confidence:.2f}**"
)

st.sidebar.caption(
    "Live detections must persist for 5 frames before activating the rover."
)
st.sidebar.warning(
    "The loaded model has one weed class only. It cannot explicitly reject people "
    "or reflections; add negative examples and retrain to address persistent false positives."
)


st.sidebar.divider()


ESP32_IP = st.sidebar.text_input(
    "ESP32 Controller IP",
    value=ESP32_IP
)


st.sidebar.divider()


mode = st.sidebar.radio(
    "📹 Input Mode",
    [
        "Add Image",
        "Add Video",
        "Webcam",
        "ESP32-CAM"
    ]
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"weed.pt not found:\n\n{MODEL_PATH}"
    )

    st.stop()


if not os.path.exists(YOLOV5_PATH):

    st.error(
        f"YOLOv5 folder not found:\n\n{YOLOV5_PATH}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    print()
    print("==============================")
    print("LOADING YOLOv5")
    print("==============================")

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "YOLOv5:",
        YOLOV5_PATH
    )

    print(
        "Python:",
        sys.version
    )


    # IMPORTANT:
    # Explicitly load on CPU.
    #
    # This avoids another common problem where a
    # model trained with CUDA is loaded on a machine
    # without the same CUDA environment.

    model = torch.hub.load(
        YOLOV5_PATH,
        "custom",
        path=MODEL_PATH,
        source="local",
        device="cpu"
    )


    print()
    print("MODEL LOADED SUCCESSFULLY")

    print(
        "Classes:",
        model.names
    )

    print("==============================")
    print()

    return model


# ============================================================
# MODEL LOAD
# ============================================================

try:

    model = load_model()

    st.sidebar.success(
        "✅ YOLOv5 loaded"
    )

    st.sidebar.write(
        "Classes:"
    )

    st.sidebar.write(
        model.names
    )


except Exception as e:

    st.error(
        "❌ Failed to load YOLOv5 model"
    )

    st.code(
        str(e)
    )

    st.info(
        "The error occurs while loading weed.pt, "
        "before image detection starts."
    )

    st.stop()


# ============================================================
# ESP32 COMMAND
# ============================================================

def send_esp_command(command):

    try:

        url = (
            f"http://{ESP32_IP}/{command}"
        )

        response = requests.get(
            url,
            timeout=0.5
        )

        print(
            f"ESP32 → /{command} "
            f"→ {response.status_code}"
        )

        return True

    except Exception:

        print(
            f"ESP32 unavailable → /{command}"
        )

        return False


# ============================================================
# WEED
# ============================================================

def weed_found():

    print()
    print("🌱 WEED DETECTED")
    print("🛑 STOP")
    print("🔧 CUTTER ON")
    print("💧 PUMP ON")

    send_esp_command(
        "weed"
    )


# ============================================================
# NO WEED
# ============================================================

def no_weed():

    print()
    print("🔴 NO WEED")
    print("🚜 FORWARD")
    print("🔧 CUTTER OFF")
    print("💧 PUMP OFF")

    send_esp_command(
        "no_weed"
    )


# ============================================================
# DETECTION
# ============================================================

def detect(frame):

    # Set threshold
    model.conf = confidence


    # Run YOLO
    results = model(
        frame,
        size=IMAGE_SIZE
    )


    detections = (
        results.xyxy[0]
        .cpu()
        .numpy()
    )


    weed = False

    highest_conf = 0.0


    # --------------------------------------------------------
    # CHECK EACH DETECTION
    # --------------------------------------------------------

    for detection in detections:

        x1, y1, x2, y2, conf, class_id = detection

        conf = float(conf)

        class_id = int(class_id)


        class_name = model.names[
            class_id
        ]


        print(
            f"Detected: "
            f"{class_name} "
            f"{conf * 100:.1f}%"
        )


        if "weed" in class_name.lower():

            weed = True


            if conf > highest_conf:

                highest_conf = conf


    # --------------------------------------------------------
    # DRAW BOXES
    # --------------------------------------------------------

    output = results.render()[0]


    return (
        output,
        weed,
        highest_conf
    )


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


# ============================================================
# IMAGE
# ============================================================

if mode == "Add Image":

    st.header("🖼️ Add Image")


    uploaded = st.file_uploader(
        "Upload weed image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded:

        data = uploaded.read()


        image = cv2.imdecode(
            np.frombuffer(
                data,
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )


        if image is None:

            st.error(
                "Could not read image."
            )

            st.stop()


        # Detect
        output, weed, conf = detect(
            image
        )


        # ESP32
        if weed:

            weed_found()

        else:

            no_weed()


        # Display

        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "Original"
            )

            st.image(
                cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


        with col2:

            st.subheader(
                "YOLO Detection"
            )

            st.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


        st.divider()


        if weed:

            st.success(
                f"🌱 WEED DETECTED — "
                f"{conf * 100:.1f}%"
            )

            st.write(
                "🛑 Rover: STOP"
            )

            st.write(
                "🔧 Cutter: ON"
            )

            st.write(
                "💧 Pump: ON"
            )

        else:

            st.error(
                "🔴 NO WEED DETECTED"
            )

            st.write(
                "🚜 Rover: FORWARD"
            )

            st.write(
                "🔧 Cutter: OFF"
            )

            st.write(
                "💧 Pump: OFF"
            )


# ============================================================
# VIDEO
# ============================================================

elif mode == "Add Video":

    st.header("🎥 Add Video")


    uploaded = st.file_uploader(
        "Upload video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )


    if uploaded:

        temp_file = os.path.join(
            BASE_DIR,
            "temp_video.mp4"
        )


        with open(
            temp_file,
            "wb"
        ) as f:

            f.write(
                uploaded.read()
            )


        cap = cv2.VideoCapture(
            temp_file
        )


        if not cap.isOpened():

            st.error(
                "Could not open video."
            )

            st.stop()


        frame_box = st.empty()

        status_box = st.empty()


        previous_state = None

        detection_state = {
            "positive": 0,
            "negative": 0,
            "confirmed": False
        }


        while True:

            ret, frame = cap.read()


            if not ret:

                break


            output, candidate, conf = detect(
                frame
            )

            weed = stabilize_detection(
                candidate,
                detection_state
            )


            # Only send ESP32 when state changes

            if weed != previous_state:

                if weed:

                    weed_found()

                else:

                    no_weed()


                previous_state = weed


            frame_box.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


            if weed:

                status_box.success(
                    f"🌱 WEED — "
                    f"{conf * 100:.1f}%"
                )

            elif candidate:

                status_box.warning(
                    "Possible weed — waiting for 5 consecutive frames."
                )

            else:

                status_box.error(
                    "🔴 NO WEED"
                )


        cap.release()


        # Safety
        no_weed()


        st.success(
            "Video completed."
        )


# ============================================================
# WEBCAM
# ============================================================

elif mode == "Webcam":

    st.header("📷 Webcam")


    start = st.checkbox(
        "▶ Start Webcam"
    )


    if start:

        camera = cv2.VideoCapture(
            0
        )


        if not camera.isOpened():

            st.error(
                "Could not open webcam."
            )

            st.stop()


        frame_box = st.empty()

        status_box = st.empty()


        previous_state = None

        detection_state = {
            "positive": 0,
            "negative": 0,
            "confirmed": False
        }


        while True:

            ret, frame = camera.read()


            if not ret:

                st.error(
                    "Could not read webcam."
                )

                break


            output, candidate, conf = detect(
                frame
            )

            weed = stabilize_detection(
                candidate,
                detection_state
            )


            # Send command only when state changes

            if weed != previous_state:

                if weed:

                    weed_found()

                else:

                    no_weed()


                previous_state = weed


            frame_box.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


            if weed:

                status_box.success(
                    f"🌱 WEED — "
                    f"{conf * 100:.1f}%"
                )

            elif candidate:

                status_box.warning(
                    "Possible weed — waiting for 5 consecutive frames."
                )

            else:

                status_box.error(
                    "🔴 NO WEED"
                )


        camera.release()


# ============================================================
# ESP32-CAM
# ============================================================

elif mode == "ESP32-CAM":

    st.header("📡 ESP32-CAM")


    cam_ip = st.text_input(
        "ESP32-CAM IP",
        value="192.168.1.110"
    )


    stream_url = (
        f"http://{cam_ip}:81/stream"
    )


    st.write(
        f"Stream: `{stream_url}`"
    )


    start = st.checkbox(
        "▶ Start ESP32-CAM"
    )


    if start:

        cap = cv2.VideoCapture(
            stream_url
        )


        if not cap.isOpened():

            st.error(
                "Could not connect to ESP32-CAM."
            )

            st.stop()


        frame_box = st.empty()

        status_box = st.empty()


        previous_state = None

        detection_state = {
            "positive": 0,
            "negative": 0,
            "confirmed": False
        }


        while True:

            ret, frame = cap.read()


            if not ret:

                st.error(
                    "ESP32-CAM stream stopped."
                )

                break


            output, candidate, conf = detect(
                frame
            )

            weed = stabilize_detection(
                candidate,
                detection_state
            )


            # Send command only when state changes

            if weed != previous_state:

                if weed:

                    weed_found()

                else:

                    no_weed()


                previous_state = weed


            frame_box.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )


            if weed:

                status_box.success(
                    f"🌱 WEED — "
                    f"{conf * 100:.1f}%"
                )

            elif candidate:

                status_box.warning(
                    "Possible weed — waiting for 5 consecutive frames."
                )

            else:

                status_box.error(
                    "🔴 NO WEED"
                )


        cap.release()