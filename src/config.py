import os
# --- RUTAS DE SISTEMA ---
# Usamos r"" para que Windows no confunda las barras \ con comandos especiales
BASE_PATH = r"C:\Users\lesli\Downloads"

# --- CONFIGURACIÓN MONGODB ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "tfm_agriculture"
COLLECTION_NAME = "cdl_skeleton"

# --- GEOGRAFÍA: CALIFORNIA BOUNDING BOX ---
# Límites aproximados para filtrar el dataset Nacional
# Bounding Coordinates de California (CDL USDA Metadata)
# West:  -124.5876  East:  -114.1885
# North:   41.9743  South:   32.5028
CA_BOUNDS = {
    "lat": (32.5, 42.0),
    "lon": (-124.5, -114.1)
}


# --- PARÁMETROS DE PROCESAMIENTO ---
# 55,556 puntos * 18 años (2008-2025) ≈ 1,000,000 registros
SAMPLES_PER_YEAR = 55556 

# Rango de años para el procesamiento (range en Python no incluye el último número)
YEARS_TO_PROCESS = range(2008, 2025)