import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ====== Limitar el ancho máximo del contenido ======
st.markdown("""
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .chart-container {
        max-width: 650px;
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
features = joblib.load("features_knn_churn.pkl")

df = pd.read_csv("dataset_procesado_final.csv")

# ============================
# Crear columnas segmentadas
# ============================
df["Antiguedad_seg"] = pd.cut(
    df["Antiguedad"], [0,6,12,18,24,36,200],
    labels=["0-6","7-12","13-18","19-24","25-36","36+"],
    include_lowest=True
)

df["Distancia_seg"] = pd.cut(
    df["Distancia_Almacen"], [0,10,20,30,40,200],
    labels=["0-10","11-20","21-30","31-40","40+"],
    include_lowest=True
)

df["Cashback_seg"] = pd.qcut(
    df["Monto_Cashback"], 4,
    labels=["Bajo","Medio Bajo","Medio Alto","Alto"]
)

# ============================
# MÉTRICAS DEL MODELO
# ============================
X = df[features]
y = df["Target"]

X_scaled = scaler.transform(X)
proba = modelo.predict_proba(X_scaled)[:, 1]
y_pred = (proba >= umbral).astype(int)

acc = accuracy_score(y, y_pred)
roc = roc_auc_score(y, proba)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

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

# ---- NUEVAS MÉTRICAS PROFESIONALES ----
st.markdown("### Desempeño del Modelo")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("ROC-AUC", f"{roc:.3f}")
c3.metric("Precision", f"{prec:.2%}")
c4.metric("Recall", f"{rec:.2%}")
c5.metric("F1-score", f"{f1:.2%}")

st.markdown("#### Matriz de confusión")
cm_df = pd.DataFrame(
    cm,
    index=["Real 0 (No churn)", "Real 1 (Churn)"],
    columns=["Pred 0 (No churn)", "Pred 1 (Churn)"]
)
st.table(cm_df)

st.markdown("---")

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

st.markdown('<div class="chart-container">', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(5, 3))
df.groupby(columna)["Target"].mean().plot(kind="bar", color="#77c2ff", ax=ax)

ax.set_ylabel("Tasa de churn", fontsize=8)
ax.set_xlabel(columna, fontsize=8)
ax.tick_params(axis="both", labelsize=7)

plt.tight_layout(pad=0.5)
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)
