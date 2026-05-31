# 🔥 Wildfire Prediction using CNN (Deep Learning + Streamlit)

Este projeto utiliza Redes Neurais Convolucionais (CNNs) para detectar a presença de incêndios florestais em imagens de satélite.

Uma aplicação interativa foi desenvolvida com **Streamlit** para visualizar previsões de dois modelos diferentes.

---

## 🌍 Dataset

O conjunto de dados utilizado neste projeto foi obtido a partir do **Wildfire Prediction Dataset**, disponível no Kaggle:

https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset

O dataset foi construído com base em registros oficiais de incêndios florestais do governo do Canadá:
https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003

As coordenadas geográficas (latitude e longitude) dos focos de incêndio foram utilizadas para coletar imagens de satélite via API do MapBox, formando um dataset adequado para Deep Learning e Visão Computacional.

---

## 📓 Notebook (`.ipynb`)

O notebook contém:

- Análise exploratória do dataset
- Pré-processamento das imagens
- Construção de dois modelos CNN
- Treinamento e validação
- Avaliação de performance
- Salvamento dos modelos `.keras`

---

## 🧠 Modelos

Foram treinados dois modelos CNN:

- CNN 1 (baseline)
- CNN 2 (arquitetura mais profunda e otimizada)

Ambos realizam classificação binária:

- 🔥 Wildfire
- 🌿 No Wildfire

---

## 🚀 Aplicação Streamlit

A aplicação permite:

- Selecionar imagem aleatória do dataset
- Visualizar classe real
- Rodar predição com dois modelos CNN
- Ver confiança da previsão (%)
- Comparar desempenho dos modelos

---

## ▶️ Acesso ao App

👉 https://wildfireprediction-fiap.streamlit.app/
