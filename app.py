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
# CONFIGURACIÓN GENERAL
# ============================
st.set_page_config(page_title="Churn Dashboard – KNN Fintech", layout="wide")

# ==== SOLO DISEÑO (CSS) ====
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .chart-container {
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Paleta basada en el logo ChurnZero */
    :root {
        --cz-blue-dark: #0b1f3b;
        --cz-blue-mid:  #1757a6;
        --cz-green:     #39b54a;
        --cz-light:     #e5e7eb;
    }

    /* Título con fade-in centrado */
    .fade-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, var(--cz-blue-mid), var(--cz-green));
        -webkit-background-clip: text;
        color: transparent;
        animation: fadeInTitle 1.8s ease-out forwards;
        opacity: 0;
    }
    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
    @keyframes fadeInTitle {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Tarjetas de métricas tipo cards */
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .metrics-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #020617;
        border-radius: 14px;
        padding: 12px 16px;
        min-width: 190px;
        border: 1px solid #1f2937;
        transition: all 0.18s ease-in-out;
        cursor: default;
        box-shadow: 0 4px 12px rgba(15,23,42,0.5);
    }
    .metric-card:hover {
        background: linear-gradient(135deg, var(--cz-blue-dark), var(--cz-green));
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 14px 30px rgba(15,23,42,0.9);
        border-color: rgba(255,255,255,0.18);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.15rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# CARGA DE MODELO Y ARCHIVOS
# ============================
modelo = joblib.load("modelo_knn_churn_final.pkl")
scaler = joblib.load("scaler_knn_churn.pkl")
umbral = joblib.load("umbral_optimo_knn.pkl")
features = joblib.load("features_knn_churn.pkl")

# ============================
# CARGAR TEST SET REAL
# ============================
def cargar_test_set():
    if os.path.exists("datos_test_knn.pkl"):
        X_test_scaled, y_test = joblib.load("datos_test_knn.pkl")
        return X_test_scaled, y_test
    return None, None

X_test_scaled, y_test = cargar_test_set()

# ============================
# RECONSTRUIR EL DATASET ORIGINAL EXACTO
# ============================
df = pd.read_csv("dataset_ecommerce_limpio.csv")

# Variables derivadas EXACTAS del notebook
df["Es_Nuevo"] = (df["Antiguedad"] < 5).astype(int)
df["Tiene_Queja"] = df["Queja"].astype(int)
df["Alto_Riesgo"] = ((df["Queja"] == 1) & (df["Antiguedad"] < 5)).astype(int)
df["Satisfaccion_Baja"] = (df["Nivel_Satisfaccion"] <= 2).astype(int)

X_full = df[features]
y_full = df["Target"]

# ============================
# PERMUTATION IMPORTANCE EXACTA DEL MODELO
# ============================
X_scaled_full = scaler.transform(X_full)

result = permutation_importance(
    modelo,
    X_scaled_full,
    y_full,
    n_repeats=20,
    random_state=42
)

importancias = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean
}).sort_values("importance", ascending=True)

# ============================
# CÁLCULO DE MÉTRICAS DEL MODELO
# ============================
if X_test_scaled is not None:
    origen_metricas = "Set de prueba (igual que en el notebook)"
    proba = modelo.predict_proba(X_test_scaled)[:, 1]
    y_pred = (proba >= umbral).astype(int)
else:
    origen_metricas = "Dataset completo (NO se encontró datos_test_knn.pkl)"
    X_test_scaled = X_scaled_full
    y_test = y_full
    proba = modelo.predict_proba(X_test_scaled)[:, 1]
    y_pred = (proba >= umbral).astype(int)

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, proba)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ============================
# DASHBOARD (SOLO DISEÑO CAMBIADO)
# ============================

# ---- Header con logo + título animado ----
tasa_churn = df["Target"].mean()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Asegúrate de tener este archivo en el repo
    st.image("logo_churnzero_2026.png", width=160)
    st.markdown(
        '<div class="fade-title">ChurnZero 2026 – Churn Dashboard</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Proyecto de ciencia de datos para detección temprana de churn usando KNN.</div>',
        unsafe_allow_html=True
    )

# ---- Métricas principales en tarjetas ----
st.markdown('<div class="section-title">Métricas principales</div>', unsafe_allow_html=True)

metrics_html = f"""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-label">Tasa global de churn</div>
        <div class="metric-value">{tasa_churn:.2%}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{acc:.2%}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-value">{roc:.3f}</div>
    </div>
</div>
"""
st.markdown(metrics_html, unsafe_allow_html=True)

# Detalle del modelo en expander (no cambia lógica)
with st.expander("Ver detalles del modelo cargado"):
    st.write("**Modelo cargado:**")
    st.code(str(modelo))
    st.write("**Umbral óptimo:**", f"{umbral:.6f}")
    st.caption(f"Métricas calculadas sobre: {origen_metricas}")

st.markdown("---")

# ---- Desempeño del Modelo (tarjetas) ----
st.markdown('<div class="section-title">Desempeño del modelo sobre clientes en riesgo (Churn)</div>', unsafe_allow_html=True)

perf_html = f"""
<div class="metrics-row">
    <div class="metric-card">
        <div class="metric-label">Precisión (Churn)</div>
        <div class="metric-value">{prec:.2%}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Recall (Churn)</div>
        <div class="metric-value">{rec:.2%}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">F1-score</div>
        <div class="metric-value">{f1:.2%}</div>
    </div>
</div>
"""
st.markdown(perf_html, unsafe_allow_html=True)

st.markdown("### Matriz de Confusión")
cm_df = pd.DataFrame(
    cm,
    index=["Real 0 (No churn)", "Real 1 (Churn)"],
    columns=["Pred 0 (No churn)", "Pred 1 (Churn)"]
)
st.table(cm_df)

st.markdown("---")

# ============================
# SEGMENTACIÓN
# ============================
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

df["Antiguedad_seg"] = pd.cut(
    df["Antiguedad"], [0, 6, 12, 18, 24, 36, 200],
    labels=["0-6", "7-12", "13-18", "19-24", "25-36", "36+"],
    include_lowest=True
)

df["Distancia_seg"] = pd.cut(
    df["Distancia_Almacen"], [0, 10, 20, 30, 40, 200],
    labels=["0-10", "11-20", "21-30", "31-40", "40+"],
    include_lowest=True
)

df["Cashback_seg"] = pd.qcut(
    df["Monto_Cashback"], 4,
    labels=["Bajo", "Medio Bajo", "Medio Alto", "Alto"]
)

columna = {
    "Nivel de Satisfacción": "Nivel_Satisfaccion",
    "Antigüedad": "Antiguedad_seg",
    "Distancia al Almacén": "Distancia_seg",
    "Número de Dispositivos": "Numero_Dispositivos",
    "Monto Cashback": "Cashback_seg",
}[segmento]

st.subheader(f"Tasa de churn por {segmento}")

st.markdown('<div class="chart-container">', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(5, 3))
df.groupby(columna)["Target"].mean().plot(kind="bar", color="#77c2ff", ax=ax)
ax.set_ylabel("Tasa de churn", fontsize=8)
ax.set_xlabel(columna, fontsize=8)
ax.tick_params(axis="both", labelsize=7)
plt.tight_layout(pad=0.5)
st.pyplot(fig)
st.markdown("</div>", unsafe_allow_html=True)

# ============================
# IMPORTANCIA DE VARIABLES REAL
# ============================
st.subheader("Importancia de Variables (Permutation Importance)")

fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
ax_imp.barh(importancias["feature"], importancias["importance"], color="#77c2ff")
ax_imp.set_xlabel("Impacto en el rendimiento del modelo", fontsize=8)
ax_imp.set_ylabel("Variable", fontsize=8)
ax_imp.tick_params(axis="both", labelsize=7)
plt.tight_layout(pad=0.5)
st.pyplot(fig_imp)
