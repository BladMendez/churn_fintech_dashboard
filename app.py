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

df = pd.read_csv("dataset_ecommerce_limpio.csv")

st.subheader("Columnas detectadas en el dataset")
st.write(list(df.columns))

st.subheader("Columnas que el modelo espera (features.pkl)")
st.write(features)

st.subheader("Diagnóstico de diferencias")

cols_dataset = set(df.columns)
cols_model = set(features)

faltantes = cols_model - cols_dataset
sobrantes = cols_dataset - cols_model

st.write("🔴 Columnas que el modelo necesita y NO están en el dataset:", faltantes)
st.write("🟡 Columnas que están en el dataset pero NO las usa el modelo:", sobrantes)


# ============================
# Dashboard
# ============================
st.title(" Dashboard Analítico de Churn – Fintech KNN")
st.write("Visualización simple y clara del modelo de churn.")

# ---- 1. Métricas clave ----
st.subheader(" Métricas Principales")

tasa_churn = df["Target"].mean()
st.metric("Tasa global de churn", f"{tasa_churn:.2%}")

st.write("**Modelo cargado:**", modelo)
st.write("**Umbral óptimo:**", umbral)

# ---- 2. Importancia de variables (DESACTIVADA) ----
st.subheader(" Importancia de Variables (Permutation Importance)")

st.info(""" Este gráfico fue desactivado temporalmente porque el dataset usado en Streamlit 
no coincide con las columnas usadas al entrenar el modelo.
Cuando subamos el dataset procesado final, esta sección volverá a funcionar.
""")
# ---- 3. Segmentación simple ----
st.subheader(" Distribución por Segmentos")

segmento = st.selectbox(
    "Selecciona un segmento:",
    ["Estado Civil", "Nivel de Satisfacción", "Categoría Preferida"]
)

columna = {
    "Estado Civil": "Estado_Civil",
    "Nivel de Satisfacción": "Nivel_Satisfaccion",
    "Categoría Preferida": "Categoria_Preferida"
}[segmento]

fig, ax = plt.subplots(figsize=(7, 4))
df.groupby(columna)["Target"].mean().plot(kind="bar", ax=ax, color="skyblue")
ax.set_ylabel("Tasa de churn")
ax.set_title(f"Tasa de churn por {segmento}")

st.pyplot(fig)

