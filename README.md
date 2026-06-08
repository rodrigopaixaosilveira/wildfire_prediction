# 🔥 Wildfire Prediction using CNN (Deep Learning + Streamlit)

Projeto desenvolvido para detecção de incêndios florestais em imagens de satélite utilizando Redes Neurais Convolucionais (CNNs).

Uma aplicação interativa foi criada com Streamlit para permitir a visualização e comparação das previsões realizadas por dois modelos de Deep Learning.

---

## 🚀 Acesso ao WebApp

Acesse a aplicação online:

👉 https://wildfireprediction-fiap.streamlit.app/

---

## 🎥 Vídeo de Demonstração

Vídeo apresentando a proposta da solução e o funcionamento do sistema:

👉 https://www.youtube.com/watch?v=XWra90GJp00

---

## 👥 Integrantes

| RM       | Nome           |
| -------- | -------------- |
| RM98827  | Andre Solér    |
| RM551869 | Fabrizio Maia  |
| RM96869  | Rodrigo Paixão |
| RM551684 | Victor Asfur   |
| RM550390 | Vitor Shimizu  |

---

## 🌍 Dataset

O conjunto de dados utilizado neste projeto foi obtido a partir do **Wildfire Prediction Dataset**, disponível no Kaggle:

https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset

O dataset foi construído com base em registros oficiais de incêndios florestais do governo do Canadá:

https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003

As coordenadas geográficas (latitude e longitude) dos focos de incêndio foram utilizadas para coletar imagens de satélite por meio da API do MapBox, formando um dataset adequado para aplicações de Deep Learning e Visão Computacional.

---

## 📓 Notebook (`.ipynb`)

O notebook contempla as seguintes etapas:

* Análise exploratória dos dados (EDA)
* Pré-processamento das imagens
* Construção de dois modelos CNN
* Treinamento e validação
* Avaliação de desempenho
* Salvamento dos modelos treinados (`.keras`)

---

## 🧠 Modelos Desenvolvidos

Foram treinados dois modelos de classificação binária:

### CNN 1

Modelo baseline utilizado como referência inicial.

### CNN 2

Arquitetura mais profunda e otimizada para melhorar a capacidade de generalização e desempenho.

### Classes Preditas

* 🔥 Wildfire
* 🌿 No Wildfire

---

## 💻 Aplicação Streamlit

A aplicação permite:

* Selecionar imagens aleatórias do dataset
* Visualizar a classe real da imagem
* Executar predições utilizando os dois modelos CNN
* Exibir a confiança da predição (%)
* Comparar o desempenho entre os modelos
* Avaliar visualmente os resultados obtidos
