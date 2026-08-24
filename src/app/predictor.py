"""
================================================================================
 MOTOR DE PREDICCIÓN DEL SISTEMA DE RECOMENDACIÓN DE CULTIVOS
================================================================================
Dado un punto (suelo + clima presente y futuro), asigna su región (cluster) y
predice el Índice de Idoneidad Agrícola (IAI) de los 12 cultivos, tanto en el
presente (2025) como en los escenarios futuros.

La asignación de región se hace una sola vez con las variables del presente,
ya que la región biofísica de una parcela no cambia entre presente y futuro.
================================================================================
"""

import os
import json
import joblib
import numpy as np

# ─── Rutas (ajusta a tu estructura) ───
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LGB_DIR = os.path.join(MODELS_DIR, "lightgbm_final_v3")

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
        """
        datos: dict con las 10 variables de VARS_CLUSTER (valores del presente).
        Devuelve el número de cluster (0-4).
        """
        x = np.array([[datos[v] for v in VARS_CLUSTER]], dtype=np.float64)
        x_sc = self.scaler.transform(x)
        return int(self.kmeans.predict(x_sc)[0])

    # ──────────────────────────────────────────────────────────────
    #  Predecir el IAI de los 12 cultivos para un conjunto de variables
    # ──────────────────────────────────────────────────────────────
    def _predecir_iai(self, cluster: int, clima_suelo: dict) -> dict:
        """
        Predice el IAI de los 12 cultivos para un cluster dado y un conjunto
        de variables de clima + suelo.
        clima_suelo: dict con tmax, tmin, ppt, ph1to1h2o_r, awc_r,
                     profundidad_efectiva_cm, claytotal_r, dbthirdbar_r,
                     sandtotal_r, silttotal_r.
        Devuelve: {crop_id: iai}
        """
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
                # las demás one-hot quedan en 0
            iai = float(np.clip(modelo.predict(x)[0], 0, 1))
            resultados[crop_id] = iai
        return resultados

    # ──────────────────────────────────────────────────────────────
    #  Predicción completa: presente y futuro (por año) para un punto
    # ──────────────────────────────────────────────────────────────
    def predecir_punto(self, punto: dict) -> dict:
        """
        Devuelve un dict con:
          - cluster, etiqueta
          - iai: {momento: {crop_id: iai}} donde momento es
                 "2025", "2030-245", "2040-245", "2030-585", "2040-585"
          - clima: {momento: {tmax, tmin, ppt}} para los mismos momentos
        """
        # Verificar variables del clustering
        faltantes = [v for v in VARS_CLUSTER
                     if punto.get(v) is None or
                     (isinstance(punto.get(v), float) and np.isnan(punto.get(v)))]
        if faltantes:
            return {"error": f"Faltan datos para predecir: {', '.join(faltantes)}"}

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
        clima["2025"] = {"tmax": punto["tmax"], "tmin": punto["tmin"], "ppt": punto["ppt"]}
        iai["2025"] = self._predecir_iai(cluster, {**clima["2025"], **suelo})

        # 4. Futuro por año y escenario
        for esc in ["245", "585"]:
            for anio in [2030, 2040]:
                cl = {
                    "tmax": punto[f"tmax_{esc}_{anio}"],
                    "tmin": punto[f"tmin_{esc}_{anio}"],
                    "ppt": punto[f"pr_{esc}_{anio}"],
                }
                clave = f"{anio}-{esc}"
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

    # Cargar la base y probar con un punto
    base = pl.read_parquet(
        os.path.join(os.path.dirname(__file__), "..", "Data", "base_california_app.parquet"))
    completos = base.filter(pl.col("datos_completos") == True)
    punto = completos.row(0, named=True)

    resultado = predictor.predecir_punto(punto)
    print(f"\nCluster asignado: {resultado['cluster']} ({resultado['etiqueta']})")
    print("\nRanking presente:")
    for nombre, iai in predictor.ranking(resultado["iai_presente"], top=5):
        print(f"  {nombre}: {iai:.3f}")
    print("\nRanking futuro severo (SSP5-8.5):")
    for nombre, iai in predictor.ranking(resultado["iai_futuro"]["585"], top=5):
        print(f"  {nombre}: {iai:.3f}") have valid feature names")

# ─── Rutas (ajusta a tu estructura) ───

# El predictor está en src/app/, los modelos en TFM/models/ (subir 2 niveles)
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
        """
        datos: dict con las 10 variables de VARS_CLUSTER (valores del presente).
        Devuelve el número de cluster (0-4).
        """
        x = np.array([[datos[v] for v in VARS_CLUSTER]], dtype=np.float64)
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.astype(np.float64)
        x_sc = self.scaler.transform(x).astype(np.float64)
        return int(self.kmeans.predict(x_sc)[0])

    # ──────────────────────────────────────────────────────────────
    #  Predecir el IAI de los 12 cultivos para un conjunto de variables
    # ──────────────────────────────────────────────────────────────
    def _predecir_iai(self, cluster: int, clima_suelo: dict) -> dict:
        """
        Predice el IAI de los 12 cultivos para un cluster dado y un conjunto
        de variables de clima + suelo.
        clima_suelo: dict con tmax, tmin, ppt, ph1to1h2o_r, awc_r,
                     profundidad_efectiva_cm, claytotal_r, dbthirdbar_r,
                     sandtotal_r, silttotal_r.
        Devuelve: {crop_id: iai}
        """
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
                # las demás one-hot quedan en 0
            iai = float(np.clip(modelo.predict(x)[0], 0, 1))
            resultados[crop_id] = iai
        return resultados

    # ──────────────────────────────────────────────────────────────
    #  Predicción completa: presente y futuro para un punto
    # ──────────────────────────────────────────────────────────────
    def predecir_punto(self, punto: dict, promediar_futuro=True) -> dict:
        """
        punto: dict con TODAS las variables del punto:
          - Suelo: ph1to1h2o_r, awc_r, profundidad_efectiva_cm, claytotal_r,
                   dbthirdbar_r, sandtotal_r, silttotal_r
          - Clima presente: tmax, tmin, ppt, vpdmean
          - Clima futuro: tmax_245_2030, tmin_245_2030, ppt_245_2030, ... etc.
            (para cada escenario 245/585 y año 2030/2040)

        Devuelve un dict con:
          - cluster, etiqueta
          - iai_presente: {crop_id: iai}
          - iai_futuro: {escenario: {crop_id: iai}}
        """
        # 1. Asignar cluster con las variables del PRESENTE
        cluster = self.asignar_cluster(punto)

        # 2. Variables de suelo (comunes a presente y futuro)
        suelo = {
            "ph1to1h2o_r": punto["ph1to1h2o_r"],
            "awc_r": punto["awc_r"],
            "profundidad_efectiva_cm": punto["profundidad_efectiva_cm"],
            "claytotal_r": punto["claytotal_r"],
            "dbthirdbar_r": punto["dbthirdbar_r"],
            "sandtotal_r": punto["sandtotal_r"],
            "silttotal_r": punto["silttotal_r"],
        }

        # 3. IAI presente
        clima_pres = {"tmax": punto["tmax"], "tmin": punto["tmin"], "ppt": punto["ppt"]}
        iai_presente = self._predecir_iai(cluster, {**clima_pres, **suelo})

        # 4. IAI futuro por escenario
        iai_futuro = {}
        for esc in ["245", "585"]:
            if promediar_futuro:
                # Promediar 2030 y 2040
                tmax = np.mean([punto[f"tmax_{esc}_2030"], punto[f"tmax_{esc}_2040"]])
                tmin = np.mean([punto[f"tmin_{esc}_2030"], punto[f"tmin_{esc}_2040"]])
                ppt = np.mean([punto[f"pr_{esc}_2030"], punto[f"pr_{esc}_2040"]])
                clima_fut = {"tmax": tmax, "tmin": tmin, "ppt": ppt}
                iai_futuro[esc] = self._predecir_iai(cluster, {**clima_fut, **suelo})
            else:
                # Por año
                for anio in [2030, 2040]:
                    clima_fut = {
                        "tmax": punto[f"tmax_{esc}_{anio}"],
                        "tmin": punto[f"tmin_{esc}_{anio}"],
                        "ppt": punto[f"pr_{esc}_{anio}"],
                    }
                    iai_futuro[f"{esc}_{anio}"] = self._predecir_iai(
                        cluster, {**clima_fut, **suelo})

        return {
            "cluster": cluster,
            "etiqueta": ETIQUETAS_CLUSTER[cluster],
            "iai_presente": iai_presente,
            "iai_futuro": iai_futuro,
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

    # Cargar la base y probar con un punto
    base = pl.read_parquet(
        os.path.join(os.path.dirname(__file__), "..", "..", "Data", "base_california_app.parquet"))
    completos = base.filter(pl.col("datos_completos") == True)
    punto = completos.row(0, named=True)

    resultado = predictor.predecir_punto(punto)
    print(f"\nCluster asignado: {resultado['cluster']} ({resultado['etiqueta']})")
    print("\nRanking presente:")
    for nombre, iai in predictor.ranking(resultado["iai_presente"], top=5):
        print(f"  {nombre}: {iai:.3f}")
    print("\nRanking futuro severo (SSP5-8.5):")
    for nombre, iai in predictor.ranking(resultado["iai_futuro"]["585"], top=5):
        print(f"  {nombre}: {iai:.3f}")
