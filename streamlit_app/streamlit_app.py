import streamlit as st
import numpy as np
import tensorflow as tf
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Wildfire CNN", layout="wide")

BASE_DIR = os.path.dirname(__file__)

MODEL_1_PATH = os.path.join(BASE_DIR, "model_1.keras")
MODEL_2_PATH = os.path.join(BASE_DIR, "model_2.keras")

AMOSTRAS_DIR = os.path.join(BASE_DIR, "amostras_dataset_test")

IMG_SIZE = (128, 128)

# =========================
# LOAD MODELS (CACHE)
# =========================
@st.cache_resource
def load_models():
    model_1 = tf.keras.models.load_model(MODEL_1_PATH)
    model_2 = tf.keras.models.load_model(MODEL_2_PATH)
    return model_1, model_2

model_1, model_2 = load_models()

# =========================
# IMAGE PROCESSING
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
    files = [f for f in os.listdir(AMOSTRAS_DIR)
             if f.endswith((".jpg", ".png", ".jpeg"))]

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

st.write("Escolha uma imagem aleatória das amostras e compare os modelos.")

# Session state
if "image_data" not in st.session_state:
    st.session_state.image_data = None

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
    # PREDICTION BUTTON
    # =========================
    if st.button("🚀 Realizar predição"):
        input_img = preprocess_image(img)

        pred_1 = model_1.predict(input_img)[0][0]
        pred_2 = model_2.predict(input_img)[0][0]

        label_pred_1 = int(pred_1 > 0.5)
        label_pred_2 = int(pred_2 > 0.5)

        with col2:
            st.subheader("Resultados")

            st.write("### Modelo 1")
            st.write(f"Probabilidade: {pred_1:.4f}")
            st.write("Predição:", "🔥 Wildfire" if label_pred_1 == 1 else "🌿 No Wildfire")

            st.write("---")

            st.write("### Modelo 2")
            st.write(f"Probabilidade: {pred_2:.4f}")
            st.write("Predição:", "🔥 Wildfire" if label_pred_2 == 1 else "🌿 No Wildfire")

# =========================
# METRICS (EXEMPLO FIXO)
# =========================
st.markdown("## 📊 Comparação dos Modelos")

col1, col2 = st.columns(2)

with col1:
    st.metric("CNN 1 Accuracy", "0.9463")
    st.metric("CNN 1 Loss", "0.2851")

with col2:
    st.metric("CNN 2 Accuracy", "0.9712")
    st.metric("CNN 2 Loss", "0.1260")

st.success("CNN 2 teve desempenho significativamente melhor devido a maior profundidade e melhor extração de features.")

# =========================
# MATRIZ DE CONFUSÃO (EXEMPLO)
# =========================
st.markdown("## 📌 Matriz de Confusão")

# exemplos fictícios (substitua pelos seus reais se quiser)
y_true = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
y_pred_1 = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred_2 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]

cm1 = confusion_matrix(y_true, y_pred_1)
cm2 = confusion_matrix(y_true, y_pred_2)

col1, col2 = st.columns(2)

with col1:
    st.write("### CNN 1")
    fig, ax = plt.subplots()
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

with col2:
    st.write("### CNN 2")
    fig, ax = plt.subplots()
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax)
    st.pyplot(fig)