"""
================================================================================
 MOTOR DE PREDICCIÓN DEL SISTEMA DE RECOMENDACIÓN DE CULTIVOS
================================================================================
Dado un punto (suelo + clima presente y futuro), asigna su región (cluster) y
predice el Índice de Idoneidad Agrícola (IAI) de los 12 cultivos, tanto en el
presente (2025) como en los escenarios futuros (2030 y 2040, por separado).

La asignación de región se hace una sola vez con las variables del presente,
ya que la región biofísica de una parcela no cambia entre presente y futuro.
================================================================================
"""

import os
import json
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ─── Rutas ───
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")
LGB_DIR = os.path.join(MODELS_DIR, "lightgbm_final")

# ─── Orden EXACTO de las variables del clustering (del notebook 05) ───
VARS_CLUSTER = [
    "tmax", "tmin", "ppt", "vpdmean",
    "ph1to1h2o_r", "awc_r", "claytotal_r",
    "dbthirdbar_r", "sandtotal_r", "silttotal_r",
]

# ─── Cultivos (código CDL -> nombre en inglés, para el one-hot del modelo) ───
CROP_DICT_EN = {
    75: "Almonds", 69: "Grapes", 204: "Pistachios", 76: "Walnuts",
    54: "Tomatoes", 3: "Rice", 221: "Strawberries", 212: "Oranges",
    36: "Alfalfa", 24: "Wheat", 227: "Lettuce", 2: "Cotton",
}
CROP_DICT_ES = {
    75: "Almendras", 69: "Uvas", 204: "Pistachos", 76: "Nueces",
    54: "Tomates", 3: "Arroz", 221: "Fresas", 212: "Naranjas",
    36: "Alfalfa", 24: "Trigo", 227: "Lechuga", 2: "Algodón",
}
FINAL_CROPS = list(CROP_DICT_EN.keys())

# ─── Etiquetas de las regiones ───
ETIQUETAS_CLUSTER = {
    0: "Montaña / clima frío",
    1: "Zona templada húmeda",
    2: "Valle cálido — suelo arenoso",
    3: "Valle cálido — suelo arcilloso",
    4: "Desierto árido",
}


class PredictorCultivos:
    """Motor de predicción: carga los modelos una vez y predice por punto."""

    def __init__(self, models_dir=MODELS_DIR, lgb_dir=LGB_DIR):
        # Clustering
        self.scaler = joblib.load(os.path.join(models_dir, "scaler_cluster.pkl"))
        self.kmeans = joblib.load(os.path.join(models_dir, "kmeans_clusters.pkl"))

        # Modelos LightGBM por cluster + sus features
        self.modelos = {}
        self.features = {}
        for c in range(5):
            self.modelos[c] = joblib.load(
                os.path.join(lgb_dir, f"modelo_cluster_{c}_lgb.pkl"))
            with open(os.path.join(lgb_dir, f"features_cluster_{c}.json")) as f:
                self.features[c] = json.load(f)

    # ──────────────────────────────────────────────────────────────
    #  Asignar región (cluster) a partir de las variables del presente
    # ──────────────────────────────────────────────────────────────
    def asignar_cluster(self, datos: dict) -> int:
        x = np.array([[datos[v] for v in VARS_CLUSTER]], dtype=np.float64)
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.astype(np.float64)
        x_sc = self.scaler.transform(x).astype(np.float64)
        return int(self.kmeans.predict(x_sc)[0])

    # ──────────────────────────────────────────────────────────────
    #  Predecir el IAI de los 12 cultivos para un conjunto de variables
    # ──────────────────────────────────────────────────────────────
    def _predecir_iai(self, cluster: int, clima_suelo: dict) -> dict:
        feats = self.features[cluster]
        modelo = self.modelos[cluster]

        resultados = {}
        for crop_id in FINAL_CROPS:
            col_onehot = f"crop_name_{CROP_DICT_EN[crop_id]}"
            x = np.zeros((1, len(feats)), dtype=np.float32)
            for j, f in enumerate(feats):
                if f in clima_suelo:
                    x[0, j] = clima_suelo[f]
                elif f == col_onehot:
                    x[0, j] = 1.0
            iai = float(np.clip(modelo.predict(x)[0], 0, 1))
            resultados[crop_id] = iai
        return resultados

    # ──────────────────────────────────────────────────────────────
    #  Predicción completa: presente y futuro por año, para un punto
    # ──────────────────────────────────────────────────────────────
    def predecir_punto(self, punto: dict) -> dict:
        """
        Devuelve un dict con:
          - cluster, etiqueta
          - iai:   {momento: {crop_id: iai}}
          - clima: {momento: {tmax, tmin, ppt}}
        Los momentos son: "2025", "2030-245", "2040-245", "2030-585", "2040-585".
        """
        # 1. Asignar cluster con las variables del PRESENTE
        cluster = self.asignar_cluster(punto)

        # 2. Suelo (común a todos los momentos)
        suelo = {
            "ph1to1h2o_r": punto["ph1to1h2o_r"],
            "awc_r": punto["awc_r"],
            "profundidad_efectiva_cm": punto["profundidad_efectiva_cm"],
            "claytotal_r": punto["claytotal_r"],
            "dbthirdbar_r": punto["dbthirdbar_r"],
            "sandtotal_r": punto["sandtotal_r"],
            "silttotal_r": punto["silttotal_r"],
        }

        iai = {}
        clima = {}

        # 3. Presente (2025)
        clima["2025"] = {
            "tmax": punto["tmax"], "tmin": punto["tmin"], "ppt": punto["ppt"]
        }
        iai["2025"] = self._predecir_iai(cluster, {**clima["2025"], **suelo})

        # 4. Futuro: cada año y escenario por separado
        for esc in ["245", "585"]:
            for anio in [2030, 2040]:
                clave = f"{anio}-{esc}"
                cl = {
                    "tmax": punto[f"tmax_{esc}_{anio}"],
                    "tmin": punto[f"tmin_{esc}_{anio}"],
                    "ppt": punto[f"pr_{esc}_{anio}"],
                }
                clima[clave] = cl
                iai[clave] = self._predecir_iai(cluster, {**cl, **suelo})

        return {
            "cluster": cluster,
            "etiqueta": ETIQUETAS_CLUSTER[cluster],
            "iai": iai,
            "clima": clima,
        }

    # ──────────────────────────────────────────────────────────────
    #  Utilidad: ranking ordenado de cultivos
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def ranking(iai_dict: dict, top=None) -> list:
        """Devuelve [(nombre_es, iai), ...] ordenado de mayor a menor."""
        items = [(CROP_DICT_ES[cid], iai) for cid, iai in iai_dict.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top] if top else items


# ─── Prueba rápida del motor ───
if __name__ == "__main__":
    import polars as pl

    predictor = PredictorCultivos()
    print("Motor cargado correctamente.")

    base = pl.read_parquet(
        os.path.join(os.path.dirname(__file__), "..", "..", "Data", "base_california_app.parquet"))
    completos = base.filter(pl.col("datos_completos") == True)
    punto = completos.row(0, named=True)

    resultado = predictor.predecir_punto(punto)
    print(f"\nCluster asignado: {resultado['cluster']} ({resultado['etiqueta']})")
    print("\nMomentos disponibles:", list(resultado["iai"].keys()))
    print("\nRanking presente (2025):")
    for nombre, iai in predictor.ranking(resultado["iai"]["2025"], top=5):
        print(f"  {nombre}: {iai:.3f}")
    print("\nRanking 2040 severo (SSP5-8.5):")
    for nombre, iai in predictor.ranking(resultado["iai"]["2040-585"], top=5):
        print(f"  {nombre}: {iai:.3f}")