"""
Script INDEPENDIENTE para generar el gráfico de barras resumen de SHAP.
Carga los datos y modelos por sí mismo, así que se puede ejecutar suelto
(no necesita el notebook).

Ejecutar desde la raíz del proyecto:
    python shap_grafico_resumen.py
"""

import os
import json
import joblib
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ─── Rutas ───
RUTA_DATOS = "Data/"
MODELS_DIR = "models/"
LGB_DIR = os.path.join(MODELS_DIR, "lightgbm_final")
FIG_DIR = "notebooks/ML_outputs/figuras_latex"
os.makedirs(FIG_DIR, exist_ok=True)

etiquetas_cluster = {
    0: "Montaña / clima frío",
    1: "Zona templada húmeda",
    2: "Valle cálido — arenoso",
    3: "Valle cálido — arcilloso",
    4: "Desierto árido",
}

features_base = [
    "tmax", "tmin", "ppt",
    "ph1to1h2o_r", "awc_r", "profundidad_efectiva_cm",
    "claytotal_r", "dbthirdbar_r", "sandtotal_r", "silttotal_r",
]

NOMBRES_VAR = {
    "tmax": "Temp. máxima", "tmin": "Temp. mínima", "ppt": "Precipitación",
    "ph1to1h2o_r": "pH", "awc_r": "Cap. agua",
    "profundidad_efectiva_cm": "Profundidad", "claytotal_r": "Arcilla",
    "dbthirdbar_r": "Densidad ap.", "sandtotal_r": "Arena", "silttotal_r": "Limo",
}
VARS_BASE_GRAFICO = list(NOMBRES_VAR.keys())


def guardar_figura(fig, nombre):
    ruta = os.path.join(FIG_DIR, f"{nombre}.pdf")
    fig.savefig(ruta, format="pdf", bbox_inches="tight", dpi=300)
    print(f"Figura guardada: {ruta}")


# ─── Cargar datos y modelos ───
print("Cargando datos y modelos...")
df = pl.read_parquet(os.path.join(RUTA_DATOS, "df_cluster.parquet"))

modelos = {}
features_por_cluster = {}
for c in etiquetas_cluster:
    modelos[c] = joblib.load(os.path.join(LGB_DIR, f"modelo_cluster_{c}_lgb.pkl"))
    with open(os.path.join(LGB_DIR, f"features_cluster_{c}.json")) as f:
        features_por_cluster[c] = json.load(f)


def preparar_datos_cluster(c):
    df_c = df.filter(pl.col("cluster") == c).to_dummies(columns=["crop_name"])
    feats = features_por_cluster[c]
    for col in feats:
        if col not in df_c.columns:
            df_c = df_c.with_columns(pl.lit(0).alias(col))
    X = df_c.select(feats).to_numpy()
    return X


# ─── Calcular importancia SHAP media por variable y cluster ───
print("Calculando valores SHAP por región...")
importancias = {}
for c in etiquetas_cluster:
    np.random.seed(42)
    X = preparar_datos_cluster(c)
    feats = features_por_cluster[c]
    n = min(2000, X.shape[0])
    idx = np.random.choice(X.shape[0], size=n, replace=False)
    explainer = shap.TreeExplainer(modelos[c])
    sv = explainer.shap_values(X[idx])
    imp = pd.Series(np.abs(sv).mean(axis=0), index=feats)
    importancias[c] = {v: imp.get(v, 0) for v in VARS_BASE_GRAFICO}
    print(f"  Cluster {c} procesado")

# ─── Construir DataFrame y graficar ───
df_imp = pd.DataFrame(importancias)
df_imp.index = [NOMBRES_VAR[v] for v in df_imp.index]
df_imp["media"] = df_imp.mean(axis=1)
df_imp = df_imp.sort_values("media", ascending=True)
df_imp_plot = df_imp.drop(columns="media")

fig, ax = plt.subplots(figsize=(10, 6))
# Paleta de colores bien diferenciados (uno por región)
colores = ["#69341bac", "#fee08b", "#4575b4", "#91cf60", "#d73027"]
etiquetas = [etiquetas_cluster[c] for c in df_imp_plot.columns]

df_imp_plot.plot(kind="barh", ax=ax, color=colores, width=0.8)
ax.set_xlabel("Importancia media (valor SHAP absoluto)")
ax.set_ylabel("")
#ax.set_title("Importancia de las variables por región")
ax.legend(etiquetas, title="Región", fontsize=8, loc="lower right")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
guardar_figura(fig, "val_shap_resumen")
plt.show()

print("\nListo. Gráfico guardado en:", FIG_DIR)