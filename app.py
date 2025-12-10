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

# ---- 2. Importancia de variables ----
st.subheader(" Importancia de Variables (Permutation Importance)")

# Función para graficar importancias
def grafico_importancia():
    import numpy as np
    from sklearn.inspection import permutation_importance

    X = df[features]
    y = df["Target"]

    X_scaled = scaler.transform(X)

    result = permutation_importance(modelo, X_scaled, y, n_repeats=10, random_state=42)
    imp = pd.DataFrame({
        "feature": features,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp["feature"], imp["importance"])
    ax.set_xlabel("Impacto en rendimiento del modelo")
    ax.set_title("Importancia de Variables")
    st.pyplot(fig)

grafico_importancia()

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
