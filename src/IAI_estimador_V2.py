import os
import sys
import polars as pl

# 1. MATRIZ PARAMÉTRICA EXPERTA (FAO ECOCROP / GAEZ)
# Estructura: (Absolute_Min, Optimal_Min, Optimal_Max, Absolute_Max)

CROP_AGRONOMIC_SETTINGS = {
    75: {
        "name": "Almonds", "t_base": 7.2,
        "t_range": (10.0, 12.0, 35.0, 40.0),
        "p_range": (250.0, 600.0, 900.0, 1500.0),
        "ph_range": (5.5, 6.5, 7.0, 8.0),
        "depth_range": (20.0, 50.0, 150.0, 300.0), # Ajustado límite superior
        "awc_opt": 0.15
    },
    2: {
        "name": "Cotton", "t_base": 15.0,
        "t_range": (15.0, 22.0, 36.0, 42.0),
        "p_range": (450.0, 750.0, 1200.0, 1500.0),
        "ph_range": (5.0, 6.5, 7.5, 9.5),
        "depth_range": (50.0, 150.0, 500.0, 600.0), # >150cm óptimo
        "awc_opt": 0.15
    },
    69: {
        "name": "Grapes", "t_base": 10.0,
        "t_range": (10.0, 18.0, 30.0, 38.0),
        "p_range": (400.0, 700.0, 850.0, 1200.0),
        "ph_range": (4.5, 5.5, 7.5, 8.5),
        "depth_range": (20.0, 150.0, 500.0, 600.0), # >150cm óptimo
        "awc_opt": 0.12
    },
    204: {
        "name": "Pistachios", "t_base": 7.0,
        "t_range": (12.0, 25.0, 35.0, 40.0),
        "p_range": (250.0, 400.0, 700.0, 1100.0),
        "ph_range": (6.0, 7.0, 8.0, 8.5),
        "depth_range": (20.0, 50.0, 150.0, 300.0),
        "awc_opt": 0.12
    },
    76: {
        "name": "Walnuts", "t_base": 7.0,
        "t_range": (7.0, 15.0, 30.0, 40.0),
        "p_range": (400.0, 800.0, 1700.0, 2200.0),
        "ph_range": (4.5, 5.5, 6.5, 8.3),
        "depth_range": (150.0, 151.0, 500.0, 600.0), # Exige alta profundidad
        "awc_opt": 0.15
    },
    54: {
        "name": "Tomatoes", "t_base": 10.0,
        "t_range": (7.0, 20.0, 27.0, 35.0),
        "p_range": (400.0, 600.0, 1300.0, 1800.0),
        "ph_range": (5.0, 5.5, 6.8, 7.5),
        "depth_range": (20.0, 21.0, 50.0, 150.0),
        "awc_opt": 0.10
    },
    3: {
        "name": "Rice", "t_base": 10.0,
        "t_range": (10.0, 20.0, 30.0, 36.0),
        "p_range": (1000.0, 1500.0, 2000.0, 4000.0),
        "ph_range": (4.5, 5.5, 7.0, 9.0),
        "depth_range": (20.0, 50.0, 150.0, 300.0),
        "awc_opt": 0.15
    },
    227: {
        "name": "Lettuce", "t_base": 4.5,
        "t_range": (5.0, 12.0, 21.0, 30.0),
        "p_range": (900.0, 1100.0, 1400.0, 4100.0),
        "ph_range": (4.2, 6.0, 7.0, 7.5),
        "depth_range": (20.0, 21.0, 50.0, 150.0), # Homologado por raíz corta
        "awc_opt": 0.10
    },
    221: {
        "name": "Strawberries", "t_base": 7.0,
        "t_range": (6.0, 11.0, 24.0, 28.0),
        "p_range": (300.0, 600.0, 900.0, 1700.0),
        "ph_range": (4.5, 6.0, 6.8, 8.2),
        "depth_range": (20.0, 21.0, 50.0, 150.0),
        "awc_opt": 0.10
    },
    212: {
        "name": "Oranges", "t_base": 12.8,
        "t_range": (13.0, 20.0, 30.0, 38.0),
        "p_range": (450.0, 1200.0, 2000.0, 2700.0),
        "ph_range": (4.0, 5.0, 6.0, 8.3),
        "depth_range": (50.0, 150.0, 500.0, 600.0), # >150cm óptimo
        "awc_opt": 0.15
    },
    36: {
        "name": "Alfalfa", "t_base": 5.0,
        "t_range": (5.0, 21.0, 27.0, 45.0),
        "p_range": (350.0, 600.0, 1200.0, 2700.0),
        "ph_range": (2.3, 6.5, 7.5, 8.7),
        "depth_range": (50.0, 150.0, 500.0, 600.0), # >150cm óptimo
        "awc_opt": 0.12
    },
    24: {
        "name": "Wheat", "t_base": 0.0,
        "t_range": (5.0, 15.0, 23.0, 27.0),
        "p_range": (300.0, 750.0, 900.0, 1600.0),
        "ph_range": (5.5, 6.0, 7.0, 8.5),
        "depth_range": (20.0, 50.0, 150.0, 300.0),
        "awc_opt": 0.12
    },
    
}


# Traducción del requerimiento de horas de frío (literatura: UC Davis, extensiones
# universitarias) a umbrales de temperatura mínima anual, usada como proxy del frío
# invernal disponible.
#   - "tmin_opt": por debajo de este valor, el frío es suficiente (subíndice = 1)
#   - "tmin_max": por encima, el frío es insuficiente (subíndice = MIN_FRIO)
# Los cultivos sin requerimiento de frío ("ninguno") reciben subíndice 1 siempre.
#
# Requerimientos de horas de frío de referencia:
#   Nueces 600-1200 h (alto) · Pistachos 800-1000 h (alto)
#   Almendras 200-500 h (moderado) · Uvas 100-500 h (bajo)
#   Cítricos y cultivos anuales: sin requerimiento
CHILL_SETTINGS = {
    # Umbrales recalibrados: el óptimo incluye la tmin presente (~9.4),
    # la penalización se activa al subir hacia los valores futuros (11-13+).
    # Refleja que California hoy tiene frío suficiente para estos cultivos,
    # y el déficit aparece con el calentamiento.
    76:  {"chill": "alto",     "tmin_opt": 10.0, "tmin_max": 13.5},  # Nueces
    204: {"chill": "alto",     "tmin_opt": 10.5, "tmin_max": 14.0},  # Pistachos
    75:  {"chill": "moderado", "tmin_opt": 11.0, "tmin_max": 15.0},  # Almendras
    69:  {"chill": "bajo",     "tmin_opt": 12.0, "tmin_max": 16.0},  # Uvas
    2:   {"chill": "ninguno"},
    54:  {"chill": "ninguno"},
    3:   {"chill": "ninguno"},
    227: {"chill": "ninguno"},
    221: {"chill": "ninguno"},
    212: {"chill": "ninguno"},
    36:  {"chill": "ninguno"},
    24:  {"chill": "ninguno"},
}
 
MIN_FRIO = 0.30  # subíndice mínimo cuando falta frío (el cultivo es subóptimo, no inviable)
 
# Pesos de los subíndices (justificados por literatura; validar con sensibilidad)
PESOS = {
    "temp":   0.30,   # temperatura de crecimiento (calor)
    "frio":   0.30,   # frío invernal
    "precip": 0.25,   # precipitación
    "suelo":  0.15,   # suelo (menor peso: se considera constante en el tiempo)
}
 
 
def build_trapezoidal_expr(col_name: str, limits: tuple) -> pl.Expr:
    """
    Función de pertenencia trapezoidal (FAO ECOCROP).
    Vale 1 en el rango óptimo, decae linealmente hacia los extremos y 0 fuera
    del rango viable. Penaliza tanto el déficit como el exceso.
    """
    abs_min, opt_min, opt_max, abs_max = limits
    return (
        pl.when(pl.col(col_name) < abs_min).then(0.0)
        .when(pl.col(col_name).is_between(abs_min, opt_min, closed="left"))
            .then((pl.col(col_name) - abs_min) / (opt_min - abs_min))
        .when(pl.col(col_name).is_between(opt_min, opt_max, closed="both"))
            .then(1.0)
        .when(pl.col(col_name).is_between(opt_max, abs_max, closed="right"))
            .then((abs_max - pl.col(col_name)) / (abs_max - opt_max))
        .otherwise(0.0)
    )
 
 
def build_chill_expr(crop_id: int) -> pl.Expr:
    """
    Subíndice de frío invernal a partir de la temperatura mínima anual.
    Para cultivos con requerimiento de frío, decae cuando la tmin sube (menos
    frío invernal disponible), reflejando el efecto del calentamiento sobre la
    ruptura de dormancia. Para cultivos sin requerimiento, vale 1.
    """
    cfg = CHILL_SETTINGS[crop_id]
    if cfg["chill"] == "ninguno":
        return pl.lit(1.0)
    t_opt = cfg["tmin_opt"]
    t_max = cfg["tmin_max"]
    c = pl.col("tmin")
    return (
        pl.when(c <= t_opt).then(1.0)
        .when(c >= t_max).then(MIN_FRIO)
        .otherwise(1.0 - (1.0 - MIN_FRIO) * (c - t_opt) / (t_max - t_opt))
    )
 
 
def calculate_iai(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula el IAI y sus cuatro subíndices para cada parcela y cultivo.
    Requiere las columnas: tmax, tmin, ppt, ph1to1h2o_r,
    profundidad_efectiva_cm, awc_r, crop_id.
    """
    i_temp_cases = pl.when(False).then(0.0)
    i_frio_cases = pl.when(False).then(0.0)
    i_precip_cases = pl.when(False).then(0.0)
    i_suelo_cases = pl.when(False).then(0.0)
 
    for crop_id, params in CROP_AGRONOMIC_SETTINGS.items():
 
        # --- I_temp: temperatura de crecimiento (calor), trapezoidal sobre tmax ---
        # Penaliza tanto el frío insuficiente para crecer como el calor excesivo.
        t_trap = build_trapezoidal_expr("tmax", params["t_range"])
        i_temp_cases = i_temp_cases.when(pl.col("crop_id") == crop_id).then(t_trap)
 
        # --- I_frio: frío invernal, a partir de tmin anual ---
        frio = build_chill_expr(crop_id)
        i_frio_cases = i_frio_cases.when(pl.col("crop_id") == crop_id).then(frio)
 
        # --- I_precip: precipitación, trapezoidal sobre ppt ---
        p_trap = build_trapezoidal_expr("ppt", params["p_range"])
        i_precip_cases = i_precip_cases.when(pl.col("crop_id") == crop_id).then(p_trap)
 
        # --- I_suelo: promedio de pH, profundidad y agua disponible ---
        ph_trap = build_trapezoidal_expr("ph1to1h2o_r", params["ph_range"])
        depth_trap = build_trapezoidal_expr("profundidad_efectiva_cm", params["depth_range"])
        awc_norm = (
            pl.when(pl.col("awc_r") >= params["awc_opt"]).then(1.0)
            .otherwise(pl.col("awc_r") / params["awc_opt"])
        )
        suelo_avg = (ph_trap + depth_trap + awc_norm) / 3.0
        i_suelo_cases = i_suelo_cases.when(pl.col("crop_id") == crop_id).then(suelo_avg)
 
    df_with_indices = df.with_columns([
        i_temp_cases.otherwise(0.0).alias("I_temp"),
        i_frio_cases.otherwise(0.0).alias("I_frio"),
        i_precip_cases.otherwise(0.0).alias("I_precip"),
        i_suelo_cases.otherwise(0.0).alias("I_suelo"),
    ])
 
    df_final = df_with_indices.with_columns([
        (
            pl.col("I_temp")   * PESOS["temp"] +
            pl.col("I_frio")   * PESOS["frio"] +
            pl.col("I_precip") * PESOS["precip"] +
            pl.col("I_suelo")  * PESOS["suelo"]
        ).alias("IAI")
    ])
 
    return df_final