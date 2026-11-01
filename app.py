import os
import cv2
import torch
import numpy as np
import streamlit as st

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "weed.pt")
YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Weed Detection",
    layout="wide"
)

st.title("🌱 Weed Detection System")


# ============================================================
# CONFIDENCE SLIDER
# ============================================================

confidence = st.slider(
    "🎯 Detection Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.20,
    step=0.05
)

st.write(f"Current confidence threshold: **{confidence:.2f}**")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = torch.hub.load(
        YOLOV5_PATH,
        "custom",
        path=MODEL_PATH,
        source="local"
    )

    return model


try:

    model = load_model()

    st.success("YOLO model loaded successfully.")

    st.write("Model classes:", model.names)

except Exception as e:

    st.error("Failed to load model.")
    st.exception(e)
    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload one weed image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.read()

    image = cv2.imdecode(
        np.frombuffer(file_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Could not read image.")
        st.stop()


    # ========================================================
    # SET CONFIDENCE
    # ========================================================

    model.conf = confidence


    # ========================================================
    # RUN YOLO
    # ========================================================

    print("\n==============================")
    print("RUNNING YOLO")
    print("==============================")

    print(f"Confidence threshold: {confidence:.2f}")

    results = model(
        image,
        size=640
    )


    # ========================================================
    # GET DETECTIONS
    # ========================================================

    detections = results.xyxy[0].cpu().numpy()

    print(f"Total detections: {len(detections)}")


    weed_found = False
    highest_conf = 0.0


    # ========================================================
    # CHECK DETECTIONS
    # ========================================================

    for detection in detections:

        x1, y1, x2, y2, conf, class_id = detection

        conf = float(conf)
        class_id = int(class_id)

        class_name = model.names[class_id]

        print(
            f"Detected: {class_name} | "
            f"Confidence: {conf:.3f}"
        )


        if "weed" in class_name.lower():

            weed_found = True

            if conf > highest_conf:
                highest_conf = conf


    # ========================================================
    # ACTION
    # ========================================================

    print("\n------------------------------")

    if weed_found:

        print("🌱 WEED DETECTED")
        print(f"📊 Confidence: {highest_conf * 100:.1f}%")
        print("🛑 Rover stopped")
        print("🔧 Cutter activated")
        print("💧 Pump activated")

    else:

        print("🔴 NO WEED DETECTED")
        print("🚜 Rover moving forward")

    print("------------------------------")


    # ========================================================
    # DISPLAY
    # ========================================================

    annotated_image = results.render()[0]

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    original_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Original")

        st.image(
            original_image,
            use_container_width=True
        )


    with col2:

        st.subheader("Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    if weed_found:

        st.success(
            f"🌱 WEED DETECTED — "
            f"{highest_conf * 100:.1f}%"
        )

        st.info("🔧 Cutter activated")
        st.info("💧 Pump activated")

    else:

        st.error("🔴 NO WEED DETECTED")

        st.info("🚜 Rover moving forward")