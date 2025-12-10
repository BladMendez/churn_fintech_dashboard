import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ============================
# Cargar archivos del modelo
# ============================
modelo = joblib.load("modelo_knn_churn_final.pkl")
scaler = joblib.load("scaler_knn_churn.pkl")
features = joblib.load("features_knn_churn.pkl")
umbral = joblib.load("umbral_optimo_knn.pkl")

# Cargar dataset procesado FINAL (el que sí coincide con el modelo)
df = pd.read_csv("dataset_procesado_final.csv")

# ============================
# Dashboard
# ============================
st.title("Dashboard Analítico de Churn – Fintech KNN")
st.write("Visualización simple y clara del modelo de churn.")

# ---- 1. Métricas clave ----
st.subheader("Métricas Principales")

tasa_churn = df["Target"].mean()
st.metric("Tasa global de churn", f"{tasa_churn:.2%}")

st.write("**Modelo cargado:**", modelo)
st.write("**Umbral óptimo:**", umbral)

# ============================
# 2. Segmentación simple
# ============================

st.subheader("Distribución por Segmentos")

# SOLO columnas que existen en el dataset final
segmentos_validos = {
    "Nivel de Satisfacción": "Nivel_Satisfaccion",
    "Antigüedad": "Antiguedad",
    "Distancia al Almacén": "Distancia_Almacen",
    "Número de Dispositivos": "Numero_Dispositivos",
    "Monto Cashback": "Monto_Cashback",
}

opcion = st.selectbox("Selecciona un segmento:", list(segmentos_validos.keys()))
columna = segmentos_validos[opcion]

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby(columna)["Target"].mean().plot(kind="bar", ax=ax, color="skyblue")
ax.set_ylabel("Tasa de churn")
ax.set_title(f"Tasa de churn por {opcion}")

st.pyplot(fig)
