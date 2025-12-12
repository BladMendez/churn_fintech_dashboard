import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
import os

# ============================
# CONFIG GENERAL
# ============================
st.set_page_config(page_title="ChurnZero 2026 – Dashboard KNN", layout="wide")

# ============================
#  CSS — NUEVO DISEÑO
# ============================
st.markdown("""
<style>

:root {
    --cz-blue-dark: #0b1f3b;
    --cz-blue-mid:  #1757a6;
    --cz-green:     #39b54a;
    --cz-white:     #f2f2f2;
}

/* =======================
   FONDO DEL DASHBOARD
======================= */
body, .main {
    background: linear-gradient(135deg, var(--cz-blue-dark), var(--cz-blue-mid), var(--cz-green));
    background-attachment: fixed;
}

/* Ajusta el contenedor principal */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    border-radius: 12px;
}

/* =======================
   TÍTULO PRINCIPAL
======================= */
.fade-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    margin-top: -0.5rem;
    margin-bottom: 0.8rem;
    color: var(--cz-white);
    text-shadow: 0px 2px 4px rgba(0,0,0,0.4);
    animation: fadeInTitle 1.2s ease-out forwards;
}

@keyframes fadeInTitle {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* =======================
   SUBTÍTULOS Y TEXTOS
======================= */
h2, h3, h4, label, p, .stMetric {
    color: var(--cz-white) !important;
    font-weight: 500;
}

/* =======================
   TARJETAS DE MÉTRICAS
======================= */
.metric-card {
    background: rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 14px 18px;
    min-width: 190px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    backdrop-filter: blur(4px);
}

.metric-label {
    font-size: 0.8rem;
    color: #e4e4e4;
    text-transform: uppercase;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
}

/* =======================
   TABLAS
======================= */
thead th {
    color: black !important;
    background-color: #e8f0fa;
}

tbody td {
    background-color: white;
    color: black !important;
}

/* =======================
   GRÁFICAS
======================= */
.chart-container {
    max-width: 680px;
    margin-left: auto;
    margin-right: auto;
}

</style>
""", unsafe_allow_html=True)


# ============================
# LOAD MODEL + DATA
# ============================
@st.cache_resource
def cargar_modelo():
    return (
        joblib.load("modelo_knn_churn_final.pkl"),
        joblib.load("scaler_knn_churn.pkl"),
        joblib.load("umbral_optimo_knn.pkl"),
        joblib.load("features_knn_churn.pkl")
    )

@st.cache_data
def cargar_dataset():
    return pd.read_csv("dataset_ecommerce_limpio.csv")

@st.cache_data
def cargar_test_set():
    if os.path.exists("datos_test_knn.pkl"):
        return joblib.load("datos_test_knn.pkl")
    return None, None

modelo, scaler, umbral, features = cargar_modelo()
df = cargar_dataset()
X_test_scaled, y_test = cargar_test_set()

# Variables derivadas
df["Es_Nuevo"] = (df["Antiguedad"] < 5).astype(int)
df["Tiene_Queja"] = df["Queja"].astype(int)
df["Alto_Riesgo"] = ((df["Queja"] == 1) & (df["Antiguedad"] < 5)).astype(int)
df["Satisfaccion_Baja"] = (df["Nivel_Satisfaccion"] <= 2).astype(int)

X_full = df[features]
y_full = df["Target"]
X_scaled_full = scaler.transform(X_full)

# ============================
# PERMUTATION IMPORTANCE
# ============================
@st.cache_resource
def compute_permutation():
    return permutation_importance(
        modelo, X_scaled_full, y_full,
        n_repeats=20, random_state=42
    )

result = compute_permutation()
importancias = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean
}).sort_values("importance", ascending=True)

# ============================
# LOGO
# ============================
if os.path.exists("logo_churnzero_2026.png"):
    st.image("logo_churnzero_2026.png", width=180)

# ============================
# TITLE
# ============================
st.markdown("<h1 class='fade-title'>ChurnZero 2026 – Dashboard KNN</h1>", unsafe_allow_html=True)

# ============================
# MÉTRICAS PRINCIPALES
# ============================
st.subheader("Métricas Principales")

tasa_churn = df["Target"].mean()
st.metric("Tasa global de churn", f"{tasa_churn:.2%}")
st.write("**Modelo cargado:**", modelo)
st.write("**Umbral óptimo:**", f"{umbral:.6f}")

# ============================
# MÉTRICAS DEL MODELO
# ============================
st.subheader("Desempeño del Modelo")

if X_test_scaled is None:
    X_test_scaled = X_scaled_full
    y_test = y_full

proba = modelo.predict_proba(X_test_scaled)[:, 1]
y_pred = (proba >= umbral).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, proba)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("ROC-AUC", f"{roc:.3f}")
c3.metric("Precisión", f"{prec:.2%}")
c4.metric("Recall", f"{rec:.2%}")
c5.metric("F1-score", f"{f1:.2%}")

# ============================
# MATRIZ DE CONFUSIÓN
# ============================
st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
st.table(pd.DataFrame(cm,
    index=["Real 0", "Real 1"],
    columns=["Pred 0", "Pred 1"]
))

# ============================
# 👉 RESTAURACIÓN DE SEGMENTACIÓN (GRÁFICA DESPLEGABLE)
# ============================
st.subheader("Segmentación de Churn")

segmento = st.selectbox(
    "Selecciona un segmento:",
    [
        "Nivel de Satisfacción",
        "Antigüedad",
        "Distancia al Almacén",
        "Número de Dispositivos",
        "Monto Cashback",
    ]
)

# Bins iguales a los de tu notebook
df["Antiguedad_seg"] = pd.cut(df["Antiguedad"], [0, 6, 12, 18, 24, 36, 200],
    labels=["0-6", "7-12", "13-18", "19-24", "25-36", "36+"], include_lowest=True)

df["Distancia_seg"] = pd.cut(df["Distancia_Almacen"], [0,10,20,30,40,200],
    labels=["0-10","11-20","21-30","31-40","40+"], include_lowest=True)

df["Cashback_seg"] = pd.qcut(df["Monto_Cashback"], 4,
    labels=["Bajo","Medio Bajo","Medio Alto","Alto"])

columna = {
    "Nivel de Satisfacción": "Nivel_Satisfaccion",
    "Antigüedad": "Antiguedad_seg",
    "Distancia al Almacén": "Distancia_seg",
    "Número de Dispositivos": "Numero_Dispositivos",
    "Monto Cashback": "Cashback_seg",
}[segmento]

with st.container():
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(6,3))
    df.groupby(columna)["Target"].mean().plot(kind="bar", color="#a7d7ff", ax=ax)
    ax.set_ylabel("Tasa de churn")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# IMPORTANCIA DE VARIABLES
# ============================
st.subheader("Importancia de Variables")

with st.container():
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.barh(importancias["feature"], importancias["importance"], color="#a7d7ff")
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)
