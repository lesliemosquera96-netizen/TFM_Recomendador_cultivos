"""
================================================================================
 PRECÁLCULO DEL IAI PARA LOS MAPAS COROPLÉTICOS
================================================================================
Recorre toda la base de California y calcula el IAI de los 12 cultivos para
cada punto, en el presente (2025) y en los escenarios futuros por año
(2030 y 2040, para SSP2-4.5 y SSP5-8.5). Guarda el resultado en formato largo
para pintar los mapas.

Reutiliza el motor de predicción (predictor.py). Se ejecuta una sola vez;
la app luego solo lee el resultado.

Formato de salida (largo): una fila por punto × cultivo × escenario
  lon | lat | crop_id | cultivo | escenario | iai
================================================================================
"""

import os
import numpy as np
import polars as pl

from predictor import (
    PredictorCultivos, VARS_CLUSTER, FINAL_CROPS,
    CROP_DICT_EN, CROP_DICT_ES,
)

# ─── Rutas ───
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RUTA_BASE = os.path.join(PROJECT_ROOT, "Data", "base_california_app.parquet")
RUTA_SALIDA = os.path.join(PROJECT_ROOT, "Data", "iai_mapas_california.parquet")


def precalcular():
    print("Cargando motor y base...")
    predictor = PredictorCultivos()
    base = pl.read_parquet(RUTA_BASE)

    # Solo puntos con datos completos
    completos = base.filter(pl.col("datos_completos") == True)
    print(f"Puntos a procesar: {completos.height:,}")

    # 1. Asignar cluster a todos los puntos de una vez
    #    (forzar float64 para evitar el "Buffer dtype mismatch" de K-Means)
    X_cluster = completos.select(VARS_CLUSTER).to_numpy().astype(np.float64)
    X_cluster_sc = predictor.scaler.transform(X_cluster).astype(np.float64)
    predictor.kmeans.cluster_centers_ = predictor.kmeans.cluster_centers_.astype(np.float64)
    clusters = predictor.kmeans.predict(X_cluster_sc)
    completos = completos.with_columns(pl.Series("cluster", clusters))

    # 2. Escenarios: presente (2025) + cada año y escenario por separado
    escenarios = {
        "2025": {
            "tmax": "tmax", "tmin": "tmin", "ppt": "ppt",
        },
        "2030 · SSP2-4.5": {
            "tmax": "tmax_245_2030", "tmin": "tmin_245_2030", "ppt": "pr_245_2030",
        },
        "2040 · SSP2-4.5": {
            "tmax": "tmax_245_2040", "tmin": "tmin_245_2040", "ppt": "pr_245_2040",
        },
        "2030 · SSP5-8.5": {
            "tmax": "tmax_585_2030", "tmin": "tmin_585_2030", "ppt": "pr_585_2030",
        },
        "2040 · SSP5-8.5": {
            "tmax": "tmax_585_2040", "tmin": "tmin_585_2040", "ppt": "pr_585_2040",
        },
    }

    # Variables de suelo (comunes a todos los escenarios)
    vars_suelo = ["ph1to1h2o_r", "awc_r", "profundidad_efectiva_cm",
                  "claytotal_r", "dbthirdbar_r", "sandtotal_r", "silttotal_r"]

    filas = []

    for esc_nombre, mapeo in escenarios.items():
        print(f"Procesando escenario: {esc_nombre}")

        # Construir el clima de este escenario
        df_esc = completos.clone()
        for var in ["tmax", "tmin", "ppt"]:
            df_esc = df_esc.with_columns(pl.col(mapeo[var]).alias(f"_{var}"))

        # Procesar por cluster (cada cluster usa su modelo y sus features)
        for c in range(5):
            sub = df_esc.filter(pl.col("cluster") == c)
            if sub.height == 0:
                continue

            feats = predictor.features[c]
            modelo = predictor.modelos[c]
            n = sub.height

            # Matriz base de features (clima + suelo), sin one-hot
            base_features = {
                "tmax": sub["_tmax"].to_numpy(),
                "tmin": sub["_tmin"].to_numpy(),
                "ppt": sub["_ppt"].to_numpy(),
            }
            for v in vars_suelo:
                base_features[v] = sub[v].to_numpy()

            lon = sub["lon"].to_numpy()
            lat = sub["lat"].to_numpy()

            # Para cada cultivo, construir X y predecir
            for crop_id in FINAL_CROPS:
                col_onehot = f"crop_name_{CROP_DICT_EN[crop_id]}"
                X = np.zeros((n, len(feats)), dtype=np.float32)
                for j, f in enumerate(feats):
                    if f in base_features:
                        X[:, j] = base_features[f]
                    elif f == col_onehot:
                        X[:, j] = 1.0
                iai = np.clip(modelo.predict(X), 0, 1)

                filas.append(pl.DataFrame({
                    "lon": lon,
                    "lat": lat,
                    "crop_id": np.full(n, crop_id, dtype=np.int32),
                    "escenario": [esc_nombre] * n,
                    "iai": iai.astype(np.float32),
                }))

    # Unir todo
    resultado = pl.concat(filas)
    # Añadir nombre en español
    resultado = resultado.with_columns(
        pl.col("crop_id").replace_strict(CROP_DICT_ES).alias("cultivo")
    )

    resultado.write_parquet(RUTA_SALIDA)
    print(f"\n[OK] Guardado en {RUTA_SALIDA}")
    print(f"     Filas: {resultado.height:,}")
    print(f"     ({resultado['lon'].n_unique():,} puntos x "
          f"{len(FINAL_CROPS)} cultivos x {len(escenarios)} escenarios)")
    print(f"     Escenarios: {list(escenarios.keys())}")


if __name__ == "__main__":
    precalcular()