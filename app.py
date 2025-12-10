import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
import os

# =====================================================================
# CONFIGURACIÓN GENERAL
# =====================================================================
st.set_page_config(page_title="Churn Dashboard – KNN Fintech", layout="wide")

# Estilos para compactar el diseño
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

# =====================================================================
# CARGA DE MODELOS Y ARCHIVOS
# =====================================================================
@st.cache_resource
def cargar_modelo_y_scaler():
    modelo = joblib.load("modelo_knn_churn_final.pkl")
    scaler = joblib.load("scaler_knn_churn.pkl")
    umbral = joblib.load("umbral_optimo_knn.pkl")
    features = joblib.load("features_knn_churn.pkl")
    return modelo, scaler, umbral, features

@st.cache_data
def cargar_datos():
    df = pd.read_csv("dataset_procesado_final.csv")
    importancias = pd.read_csv("importancia_variables_knn.csv")
    return df, importancias

@st.cache_data
def cargar_test_set():
    """
    Intenta cargar el test set oficial guardado en el notebook.
    Si no existe, Streamlit usará el dataset completo.
    """
    if os.path.exists("datos_test_knn.pkl"):
        try:
            X_test_scaled, y_test = joblib.load("datos_test_knn.pkl")
            return X_test_scaled, y_test
        except:
            st.error("Error al leer datos_test_knn.pkl. Revisa el archivo.")
    return None, None

# =====================================================================
# FUNCIONES PRINCIPALES
# =====================================================================
def calcular_metricas(modelo, umbral, scaler, df, features):
    """
    Calcula las métricas usando:
    ✔ El test set guardado (si existe)
    ✔ Todo el dataset (fallback)
    """

    X_test_scaled, y_test = cargar_test_set()

    if X_test_scaled is not None:
        origen = "Set de prueba (igual que en el notebook)"
        proba = modelo.predict_proba(X_test_scaled)[:, 1]
    else:
        origen = "Dataset completo (NO se encontró datos_test_knn.pkl)"
        X = df[features]
        y_test = df["Target"].values
        X_test_scaled = scaler.transform(X)
        proba = modelo.predict_proba(X_test_scaled)[:, 1]

    y_pred = (proba >= umbral).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, roc, prec, rec, f1, cm, origen

# =====================================================================
# CARGA DEL MODELO Y DATA
# =====================================================================
modelo, scaler, umbral, features = cargar_modelo_y_scaler()
df_original = pd.read_csv("dataset_ecommerce_limpio.csv")

# recrear exactamente las mismas features que en el modelo
df = df_original.copy()
df['Es_Nuevo'] = (df['Antiguedad'] < 5).astype(int)
df['Tiene_Queja'] = df['Queja'].astype(int)
df['Alto_Riesgo'] = ((df['Queja'] == 1) & (df['Antiguedad'] < 5)).astype(int)
df['Satisfaccion_Baja'] = (df['Nivel_Satisfaccion'] <= 2).astype(int)

# asegurar orden EXACTO
features = joblib.load("features_knn_churn.pkl")
X = df[features]
y = df["Target"]

# calcular importancia REAL como en el notebook
X_scaled = scaler.transform(X)
result = permutation_importance(modelo, X_scaled, y, n_repeats=10, random_state=42)

importancias = pd.DataFrame({
    "feature": features,
    "importance": result.importances_mean
})


# =====================================================================
# CREAR SEGMENTACIONES
# =====================================================================
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

# =====================================================================
# MÉTRICAS DEL MODELO
# =====================================================================
acc, roc, prec, rec, f1, cm, origen_metricas = calcular_metricas(
    modelo, umbral, scaler, df, features
)

# =====================================================================
# DASHBOARD
# =====================================================================
st.title("Dashboard Analítico de Churn – Fintech (KNN)")

# --------------------------
# Tasa global de churn
# --------------------------
st.subheader("Métricas Principales")
tasa_churn = df["Target"].mean()
st.metric("Tasa global de churn", f"{tasa_churn:.2%}")
st.write("**Modelo cargado:**", modelo)
st.write("**Umbral óptimo:**", f"{umbral:.6f}")
st.caption(f"Métricas calculadas sobre: **{origen_metricas}**")

# --------------------------
# Métricas del modelo
# --------------------------
st.markdown("### Desempeño del Modelo")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{acc:.2%}")
c2.metric("ROC-AUC", f"{roc:.3f}")
c3.metric("Precisión (Churn)", f"{prec:.2%}")
c4.metric("Recall (Churn)", f"{rec:.2%}")
c5.metric("F1-score", f"{f1:.2%}")

st.markdown("#### Matriz de Confusión")
cm_df = pd.DataFrame(
    cm,
    index=["Real 0 (No churn)", "Real 1 (Churn)"],
    columns=["Pred 0 (No churn)", "Pred 1 (Churn)"]
)
st.table(cm_df)

st.markdown("---")

# =====================================================================
# SEGMENTACIÓN
# =====================================================================
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

# =====================================================================
# IMPORTANCIA DE VARIABLES
# =====================================================================
st.subheader("Importancia de Variables (Permutation Importance)")

fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
importancias_sorted = importancias.sort_values("importance", ascending=True)

ax_imp.barh(importancias_sorted["feature"], importancias_sorted["importance"], color="#77c2ff")
ax_imp.set_xlabel("Impacto en el rendimiento del modelo", fontsize=8)
ax_imp.set_ylabel("Variable", fontsize=8)
ax_imp.tick_params(axis="both", labelsize=7)
plt.tight_layout(pad=0.5)

st.pyplot(fig_imp)
