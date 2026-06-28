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

# Funciones de estimación del idice de idoneidad agroclimatica

def build_trapezoidal_expr(col_name: str, limits: tuple) -> pl.Expr:
    """
    Evalúa una función de pertenencia trapezoidal EcoCrop de la FAO.
    Retorna 1.0 en rango óptimo, decaimiento lineal en estrés y 0.0 en inviabilidad.
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

def calculate_iai(df):
    # 1. Inicialización limpia de las condiciones lógicas
    gdd_cases = pl.when(False).then(0.0)
    i_temp_cases = pl.when(False).then(0.0)
    i_precip_cases = pl.when(False).then(0.0)
    i_suelo_cases = pl.when(False).then(0.0)

    # Columnas climáticas del dataset
    t_max_col = "tmax"
    t_min_col = "tmin"
    p_total_col = "ppt"

    # Duración del período de crecimiento (proxy, en días)
    DIAS_TEMPORADA = 200
    UMBRAL_GDD = 2500  # GDD de referencia para normalizar a [0,1]

    # 2. Bucle de asignación agronómica
    for crop_id, params in CROP_AGRONOMIC_SETTINGS.items():

        # --- Grados-Día de Crecimiento (I_GDD) ---
        # Adaptación FAO a resolución anual: se asume una temperatura media
        # representativa constante, aproximada como (Tmax + Tmin) / 2,
        # multiplicada por la duración de la temporada de crecimiento.
        t_media = (pl.col(t_max_col) + pl.col(t_min_col)) / 2.0

        gdd_calc = (
            pl.when((t_media - params["t_base"]) > 0)
            .then((t_media - params["t_base"]) * DIAS_TEMPORADA)
            .otherwise(0.0)
        )
        i_gdd_norm = (
            pl.when(gdd_calc > UMBRAL_GDD).then(1.0)
            .otherwise(gdd_calc / UMBRAL_GDD)
        )
        gdd_cases = gdd_cases.when(pl.col("crop_id") == crop_id).then(i_gdd_norm)

        # --- Temperatura Estacional (I_temp) | función trapezoidal sobre tmax ---
        t_trap = build_trapezoidal_expr(t_max_col, params["t_range"])
        i_temp_cases = i_temp_cases.when(pl.col("crop_id") == crop_id).then(t_trap)

        # --- Precipitación (I_precip) | función trapezoidal sobre ppt ---
        p_trap = build_trapezoidal_expr(p_total_col, params["p_range"])
        i_precip_cases = i_precip_cases.when(pl.col("crop_id") == crop_id).then(p_trap)

        # --- Suelo (I_suelo) | promedio de pH, profundidad y AWC ---
        ph_trap = build_trapezoidal_expr("ph1to1h2o_r", params["ph_range"])
        depth_trap = build_trapezoidal_expr("profundidad_efectiva_cm", params["depth_range"])
        awc_norm = (
            pl.when(pl.col("awc_r") >= params["awc_opt"]).then(1.0)
            .otherwise(pl.col("awc_r") / params["awc_opt"])
        )
        suelo_avg = (ph_trap + depth_trap + awc_norm) / 3.0
        i_suelo_cases = i_suelo_cases.when(pl.col("crop_id") == crop_id).then(suelo_avg)

    # 3. Consolidación de sub-índices
    df_with_indices = df.with_columns([
        gdd_cases.otherwise(0.0).alias("I_GDD"),
        i_temp_cases.otherwise(0.0).alias("I_temp"),
        i_precip_cases.otherwise(0.0).alias("I_precip"),
        i_suelo_cases.otherwise(0.0).alias("I_suelo")
    ])

    # 4. Ecuación final del IAI (suma ponderada)
    df_final = df_with_indices.with_columns([
        (
            (pl.col("I_GDD")    * 0.30) +
            (pl.col("I_temp")   * 0.25) +
            (pl.col("I_precip") * 0.25) +
            (pl.col("I_suelo")  * 0.20)
        ).alias("IAI")
    ])

    return df_final