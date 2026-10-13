from collections import Counter
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = "runs/car_damage/weights/best.pt"

# Friendly names for the 6 damage types
NAMES = {
    "dent": "Dent (panel damage)",
    "scratch": "Scratch",
    "crack": "Crack",
    "glass shatter": "Broken glass",
    "lamp broken": "Broken lamp",
    "tire flat": "Flat tyre",
}


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


st.set_page_config(page_title="Car Damage Detector", layout="wide")
st.title("Car Damage Detector")
st.write("Upload a photo of a damaged car. The app will mark and name the damage.")

confidence = 0.2
uploaded = st.file_uploader("Upload car photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Checking the photo..."):
        result = load_model().predict(image, conf=confidence, imgsz=640, verbose=False)[0]
        MIN_SCORE = {"tire flat": 0.6, "lamp broken": 0.5}   # flat tyre needs 60% or more
        keep = [i for i, (c, s) in enumerate(zip(result.boxes.cls, result.boxes.conf))
                if s >= MIN_SCORE.get(result.names[int(c)], 0)]
        result = result[keep]

    st.image(result.plot()[:, :, ::-1], caption="Damage found", use_container_width=True)

    if len(result.boxes) == 0:
        st.success("No damage found in this photo.")
    else:
        found = [result.names[int(c)] for c in result.boxes.cls]
        st.subheader("Damage found")
        for name, count in Counter(found).most_common():
            st.write(f"- **{NAMES.get(name, name)}**: {count}")
        st.subheader("Details")
        for name, score in zip(found, result.boxes.conf.tolist()):
            st.write(f"{NAMES.get(name, name)} — {score:.0%} sure")