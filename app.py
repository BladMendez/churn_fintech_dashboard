import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ====== Limitar el ancho máximo del contenido ======
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;     /* reduce el ancho total */
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .chart-container {
        max-width: 650px;     /* ancho fijo para la gráfica */
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ============================
# Cargar modelo y datos
# ============================
modelo = joblib.load("modelo_knn_churn_final.pkl")
scaler = joblib.load("scaler_knn_churn.pkl")
umbral = joblib.load("umbral_optimo_knn.pkl")
df = pd.read_csv("dataset_procesado_final.csv")

# ============================
# Crear columnas segmentadas
# ============================
df["Antiguedad_seg"] = pd.cut(df["Antiguedad"], [0,6,12,18,24,36,200],
                              labels=["0-6","7-12","13-18","19-24","25-36","36+"], include_lowest=True)

df["Distancia_seg"] = pd.cut(df["Distancia_Almacen"], [0,10,20,30,40,200],
                             labels=["0-10","11-20","21-30","31-40","40+"], include_lowest=True)

df["Cashback_seg"] = pd.qcut(df["Monto_Cashback"], 4,
                             labels=["Bajo","Medio Bajo","Medio Alto","Alto"])

# ============================
# Dashboard
# ============================
st.title("Dashboard Analítico de Churn – Fintech KNN")
st.write("Visualización simple y clara del churn y su comportamiento por segmentos.")

# ---- Métricas ----
st.subheader("Métricas Principales")
tasa_churn = df["Target"].mean()
st.metric("Tasa global de churn", f"{tasa_churn:.2%}")
st.write("**Modelo cargado:**", modelo)
st.write("**Umbral óptimo:**", umbral)

# ============================
# Segmentación
# ============================
segmento = st.selectbox(
    "Selecciona un segmento:",
    ["Nivel de Satisfacción", "Antigüedad", "Distancia al Almacén", 
     "Número de Dispositivos", "Monto Cashback"]
)

columna = {
    "Nivel de Satisfacción": "Nivel_Satisfaccion",
    "Antigüedad": "Antiguedad_seg",
    "Distancia al Almacén": "Distancia_seg",
    "Número de Dispositivos": "Numero_Dispositivos",
    "Monto Cashback": "Cashback_seg"
}[segmento]

# ============================
# Gráfica compacta
# ============================
st.subheader(f"Tasa de churn por {segmento}")

# Contenedor con ancho limitado
st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(5, 3))  # Gráfica más pequeña
df.groupby(columna)["Target"].mean().plot(kind="bar", color="#77c2ff", ax=ax)

ax.set_ylabel("Tasa de churn", fontsize=8)
ax.set_xlabel(columna, fontsize=8)
ax.tick_params(axis="both", labelsize=7)

plt.tight_layout(pad=0.5)
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)
