import streamlit as st
import tensorflow as tf
import numpy as np
import random
import os
from PIL import Image

# =========================
# CONFIGURAÇÃO
# =========================

st.set_page_config(
    page_title="Wildfire Detection",
    layout="wide"
)

# =========================
# CARREGAR MODELOS
# =========================

@st.cache_resource
def load_models():
    model_1 = tf.keras.models.load_model("model_1.keras")
    model_2 = tf.keras.models.load_model("model_2.keras")
    return model_1, model_2


model_1, model_2 = load_models()

# =========================
# FUNÇÃO DE PREDIÇÃO
# =========================

def predict_image(model, image):

    image = image.convert("RGB")
    image = image.resize((128, 128))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    label = "Wildfire" if pred > 0.5 else "No Wildfire"
    conf = pred * 100 if pred > 0.5 else (1 - pred) * 100

    return label, conf

# =========================
# IMAGEM ALEATÓRIA
# =========================

def get_random_image():

    base_dir = os.path.dirname(__file__)
    folder = os.path.join(base_dir, "amostras_dataset_test")

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    selected = random.choice(images)

    image_path = os.path.join(folder, selected)

    image = Image.open(image_path)

    true_label = "Wildfire" if "_wildfire" in selected.lower() else "No Wildfire"

    return image, true_label, selected

# =========================
# INTERFACE
# =========================

st.title("Detecção de Incêndios Florestais por Satélite")

tab1, tab2, tab3 = st.tabs([
    "CNN 1",
    "CNN 2",
    "Comparação"
])

# =========================
# TAB CNN 1
# =========================

with tab1:

    st.header("CNN 1")

    st.image("cnn1_accuracy.png")
    st.image("cnn1_loss.png")
    st.image("cnn1_confusion.png")

    st.subheader("Teste do modelo")

    if "img1" not in st.session_state:
        st.session_state.img1 = None
        st.session_state.label1 = None
        st.session_state.file1 = None
        st.session_state.pred1_done = False

    if st.button("Selecionar imagem aleatória (CNN 1)"):

        img, label, file = get_random_image()

        st.session_state.img1 = img
        st.session_state.label1 = label
        st.session_state.file1 = file
        st.session_state.pred1_done = False

    if st.session_state.img1:

        st.image(st.session_state.img1, width=400)

        st.write("Arquivo:", st.session_state.file1)
        st.write("Classe real:", st.session_state.label1)

        if st.button("Realizar predição (CNN 1)"):

            label, conf = predict_image(model_1, st.session_state.img1)

            st.session_state.pred1 = label
            st.session_state.conf1 = conf
            st.session_state.pred1_done = True

    if st.session_state.get("pred1_done"):

        st.success(f"Predição: {st.session_state.pred1}")
        st.info(f"Confiança: {st.session_state.conf1:.2f}%")


# =========================
# TAB CNN 2
# =========================

with tab2:

    st.header("CNN 2")

    st.image("cnn2_accuracy.png")
    st.image("cnn2_loss.png")
    st.image("cnn2_confusion.png")

    st.subheader("Teste do modelo")

    if "img2" not in st.session_state:
        st.session_state.img2 = None
        st.session_state.label2 = None
        st.session_state.file2 = None
        st.session_state.pred2_done = False

    if st.button("Selecionar imagem aleatória (CNN 2)"):

        img, label, file = get_random_image()

        st.session_state.img2 = img
        st.session_state.label2 = label
        st.session_state.file2 = file
        st.session_state.pred2_done = False

    if st.session_state.img2:

        st.image(st.session_state.img2, width=400)

        st.write("Arquivo:", st.session_state.file2)
        st.write("Classe real:", st.session_state.label2)

        if st.button("Realizar predição (CNN 2)"):

            label, conf = predict_image(model_2, st.session_state.img2)

            st.session_state.pred2 = label
            st.session_state.conf2 = conf
            st.session_state.pred2_done = True

    if st.session_state.get("pred2_done"):

        st.success(f"Predição: {st.session_state.pred2}")
        st.info(f"Confiança: {st.session_state.conf2:.2f}%")


# =========================
# TAB COMPARAÇÃO
# =========================

with tab3:

    st.header("Comparação dos Modelos")

    st.markdown("""
| Modelo | Accuracy | Loss |
|--------|----------|------|
| CNN 1 | 94.63% | 0.285 |
| CNN 2 | 97.13% | 0.126 |
""")

    st.markdown("""
A CNN 2 teve melhor desempenho devido a:

- maior profundidade (3 camadas convolucionais);
- uso de Dropout (reduz overfitting);
- melhor capacidade de generalização;
- menor loss final.

Já a CNN 1, apesar de boa performance, apresentou maior overfitting devido ao Flatten com muitos parâmetros.
""")