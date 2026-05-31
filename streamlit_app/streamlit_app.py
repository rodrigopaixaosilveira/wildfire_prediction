import streamlit as st
import numpy as np
import tensorflow as tf
import os
import random
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Wildfire CNN", layout="wide")

BASE_DIR = os.path.dirname(__file__)

MODEL_1_PATH = os.path.join(BASE_DIR, "model_1.keras")
MODEL_2_PATH = os.path.join(BASE_DIR, "model_2.keras")

AMOSTRAS_DIR = os.path.join(BASE_DIR, "amostras")

IMG_SIZE = (128, 128)

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    model_1 = tf.keras.models.load_model(MODEL_1_PATH)
    model_2 = tf.keras.models.load_model(MODEL_2_PATH)
    return model_1, model_2

model_1, model_2 = load_models()

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# RANDOM IMAGE
# =========================
def get_random_image():
    files = [
        f for f in os.listdir(AMOSTRAS_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    filename = random.choice(files)
    path = os.path.join(AMOSTRAS_DIR, filename)

    img = Image.open(path).convert("RGB")

    # label pelo nome do arquivo
    if "wildfire" in filename.lower():
        label = 1
    else:
        label = 0

    return img, label, filename

# =========================
# UI
# =========================
st.title("🔥 Wildfire Detection - CNN Comparison")

st.write("Clique para selecionar uma imagem aleatória e realizar a predição.")

# session state init
if "image_data" not in st.session_state:
    st.session_state.image_data = None

# =========================
# BUTTON - RANDOM IMAGE
# =========================
if st.button("📷 Selecionar imagem aleatória"):
    st.session_state.image_data = get_random_image()

# =========================
# SHOW IMAGE
# =========================
if st.session_state.image_data:
    img, true_label, filename = st.session_state.image_data

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption=filename, use_container_width=True)

    # =========================
    # PREDICTION
    # =========================
    if st.button("🚀 Realizar predição"):
        input_img = preprocess_image(img)

        pred_1 = model_1.predict(input_img)[0][0]
        pred_2 = model_2.predict(input_img)[0][0]

        # labels
        label_1 = "🔥 Wildfire" if pred_1 > 0.5 else "🌿 No Wildfire"
        label_2 = "🔥 Wildfire" if pred_2 > 0.5 else "🌿 No Wildfire"

        # session state
        st.session_state.label1 = label_1
        st.session_state.conf1 = float(pred_1)

        st.session_state.label2 = label_2
        st.session_state.conf2 = float(pred_2)

    # =========================
    # SHOW RESULTS
    # =========================
    with col2:
        st.subheader("Resultados")

        if "conf1" in st.session_state:
            st.write("### Modelo 1")
            st.write(f"{st.session_state.label1} | Conf: {st.session_state.conf1:.2f}")

            st.write("---")

            st.write("### Modelo 2")
            st.write(f"{st.session_state.label2} | Conf: {st.session_state.conf2:.2f}")

# =========================
# METRICS
# =========================
st.markdown("## 📊 Comparação dos Modelos")

col1, col2 = st.columns(2)

with col1:
    st.metric("CNN 1 Accuracy", "0.9463")
    st.metric("CNN 1 Loss", "0.2851")

with col2:
    st.metric("CNN 2 Accuracy", "0.9712")
    st.metric("CNN 2 Loss", "0.1260")

st.success(
    "CNN 2 teve desempenho superior devido a melhor extração de features, "
    "possivelmente por maior profundidade e melhor generalização."
)