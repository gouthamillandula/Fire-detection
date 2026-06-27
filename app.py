"""Streamlit application for detecting fire in uploaded images."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR))

from ultralytics import YOLO  # noqa: E402  (config path must be set first)


MODEL_PATH = Path(os.getenv("MODEL_PATH", "best.pt"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

DEFAULT_CONFIDENCE = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))


st.set_page_config(
    page_title="Fire Detection",
    page_icon="🔥",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the fire detection model...")
def load_model(model_path: str) -> YOLO:
    """Load and cache the YOLO model between Streamlit reruns."""
    return YOLO(model_path)


def run_detection(model: YOLO, image: Image.Image, confidence: float):
    """Run YOLO inference and draw only the highest-confidence detection."""
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )[0]
    boxes = result.boxes

    annotated_image = image.copy()
    if len(boxes):
        xyxy = boxes.xyxy.detach().cpu()
        x1 = float(xyxy[:, 0].min())
        y1 = float(xyxy[:, 1].min())
        x2 = float(xyxy[:, 2].max())
        y2 = float(xyxy[:, 3].max())
        score = float(boxes.conf.max().detach().cpu())

        padding = max(6, annotated_image.width // 100)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(annotated_image.width - 1, x2 + padding)
        y2 = min(annotated_image.height - 1, y2 + padding)

        draw = ImageDraw.Draw(annotated_image)
        color = (0, 90, 255)
        line_width = max(3, annotated_image.width // 180)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

        label = f"fire {score:.2f}"
        font = ImageFont.load_default()
        text_box = draw.textbbox((0, 0), label, font=font)
        label_width = text_box[2] - text_box[0]
        label_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - label_height - 8)
        draw.rectangle(
            (x1, label_y, x1 + label_width + 10, label_y + label_height + 8),
            fill=color,
        )
        draw.text((x1 + 5, label_y + 4), label, fill="white", font=font)

        return annotated_image, [score]

    return annotated_image, []


st.title("🔥 Fire Detection System")
st.caption("Upload an image to locate visible fire using a trained YOLO model.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.95,
        value=min(max(DEFAULT_CONFIDENCE, 0.10), 0.95),
        step=0.05,
        help="Higher values reduce false alarms but may miss faint or distant fire.",
    )
    st.write(f"Model: `{MODEL_PATH.name}`")
    st.write(f"Inference size: `{IMAGE_SIZE}px`")

if not MODEL_PATH.is_file():
    st.error(
        f"Trained model not found at `{MODEL_PATH}`. The dataset alone cannot "
        "perform inference. Train the model, then place `best.pt` in the project "
        "folder (or set the `MODEL_PATH` environment variable)."
    )
    st.code("yolo detect train data=data.yaml model=yolo11n.pt epochs=50 imgsz=640")
    st.info(
        "After training, copy `runs/detect/train/weights/best.pt` to this "
        "project folder and refresh the page."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"The model could not be loaded: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=("jpg", "jpeg", "png", "webp"),
    help="Supported formats: JPG, PNG, and WebP.",
)

if uploaded_file is None:
    st.info("Upload an image to begin detection.")
    st.stop()

try:
    uploaded_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

original_column, result_column = st.columns(2)
with original_column:
    st.subheader("Uploaded image")
    st.image(uploaded_image, use_container_width=True)

try:
    with st.spinner("Scanning the image for fire..."):
        annotated_image, boxes = run_detection(model, uploaded_image, confidence)
except Exception as exc:
    st.error(f"Detection failed: {exc}")
    st.stop()

detection_count = len(boxes)
with result_column:
    st.subheader("Detection result")
    st.image(annotated_image, use_container_width=True)

if detection_count:
    confidences = boxes
    highest_confidence = max(confidences)
    st.error(
        f"Fire detected: {detection_count} region(s) found. "
        f"Highest confidence: {highest_confidence:.1%}."
    )
    with st.expander("Detection details"):
        for index, score in enumerate(confidences, start=1):
            st.write(f"Fire region {index}: {score:.1%} confidence")
else:
    st.success(
        f"No fire was detected above the {confidence:.0%} confidence threshold."
    )

st.warning(
    "This computer-vision result can be wrong. Do not use it as the only fire "
    "safety or emergency alert system."
)

st.markdown(
    """
    <div style="text-align:center; margin-top: 2rem; color: #9ca3af;">
        Developed by Goutham
    </div>
    """,
    unsafe_allow_html=True,
)
