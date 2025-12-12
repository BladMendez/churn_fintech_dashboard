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
# CONFIGURACIÓN VISUAL
# ============================
st.set_page_config(page_title="Churn Dashboard – KNN Fintech", layout="wide")

# ---------------- CSS CUSTOM ----------------
st.markdown("""
<style>
.block-container { max-width: 1000px; padding-top: 1rem; padding-bottom: 1rem; }
.chart-container { max-width: 650px; margin-left: auto; margin-right: auto; }

:root {
    --cz-blue-dark: #0b1f3b;
    --cz-blue-mid:  #1757a6;
    --cz-green:     #39b54a;
    --cz-light:     #e5e7eb;
}

/* Título animado */
.fade-title {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, var(--cz-blue-mid), var(--cz-green));
    -webkit-background-clip: text;
    color: transparent;
    opacity: 0;
    animation: fadeInTitle 2s ease-out forwards;
}
@keyframes fadeInTitle {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Tarjetas de métricas */
.metric-card {
    background: #020617;
    border-radius: 14px;
    padding: 12px 16px;
    min-width: 190px;
    border: 1px solid #1f2937;
    transition: all 0.18s ease-in-out;
    box-shadow: 0 4px 12px rgba(15,23,42,0.5);
}
.metric-card:hover {
    background: linear-gradient(135deg, var(--cz-blue-dark), var(--cz-green));
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 14px 30px rgba(15,23,42,0.9);
}
.metric-label { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ============================
# CACHE — evitar lag
# ============================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_knn_churn_final.pkl")
    scaler = joblib.load("scaler_knn_churn.pkl")
    umbral = joblib.load("umbral_optimo_knn.pkl")
    features = joblib.load("features_knn_churn.pkl")
    return modelo, scaler, umbral, features

@st.cache_data
def cargar_dataset():
    df = pd.read_csv("dataset_ecommerce_limpio.csv")
    return df

@st.cache_data
def cargar_test_set():
    if os.path.exists("datos_test_knn.pkl"):
        return joblib.load("datos_test_knn.pkl")
    return None, None

@st.cache_resource
def compute_permutation(modelo, X_scaled_full, y_full):
    return permutation_importance(
        modelo, X_scaled_full, y_full,
        n_repeats=20, random_state=42
    )

# ============================
# CARGA DE DATOS Y MODELO
# ============================
modelo, scaler, umbral, features = cargar_modelo()
df = cargar_dataset()
X_test_scaled, y_test = cargar_test_set()

# RECREAR VARIABLES EXACTAS
df["Es_Nuevo"] = (df["Antiguedad"] < 5).astype(int)
df["Tiene_Queja"] = df["Queja"].astype(int)
df["Alto_Riesgo"] = ((df["Queja"] == 1) & (df["Antiguedad"] < 5)).astype(int)
df["Satisfaccion_Baja"] = (df["Nivel_Satisfaccion"] <= 2).astype(int)

X_full = df[features]
y_full = df["Target"]
X_scaled_full = scaler.transform(X_full)

# IMPORTANCIA (cacheada)
result = compute_permutation(modelo, X_scaled_full, y_full)
importancias = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean
}).sort_values("importance", ascending=True)

# ============================
# LOGO (evita error si no existe)
# ============================
logo_path = "logo_churnzero_2026.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=160)
else:
    st.info(" Logo no encontrado. Asegúrate de subirlo al entorno.")

# ============================
# TÍTULO ANIMADO
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
st.markdown("### Desempeño del Modelo")

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
cm = confusion_matrix(y_test, y_pred)
st.markdown("#### Matriz de Confusión")
st.table(pd.DataFrame(cm,
    index=["Real 0", "Real 1"],
    columns=["Pred 0", "Pred 1"]
))

# ============================
# IMPORTANCIA DE VARIABLES
# ============================
st.subheader("Importancia de Variables")
fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(importancias["feature"], importancias["importance"], color="#77c2ff")
st.pyplot(fig)
