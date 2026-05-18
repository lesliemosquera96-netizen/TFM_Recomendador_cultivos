import xarray as xr
import pandas as pd
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm
import os
import time
import netCDF4 as nc
import numpy as np
import requests
from datetime import datetime, timedelta
from src.config import MONGO_URI, DB_NAME, COLLECTION_NAME, BASE_PATH, CA_BOUNDS

# --- CONFIGURACIÓN ---
BASE_URL = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com/NEX-GDDP-CMIP6"
MODEL = "MPI-ESM1-2-HR"
SCENARIOS = ["ssp245", "ssp585"]
YEARS = [2030, 2040]
# Variables equivalentes a tu histórico (incluye hurs para calcular VPD después)
VARIABLES = ["pr", "tas", "tasmax", "tasmin", "hurs"]
ENSEMBLE = "r1i1p1f1"

BATCH_SIZE = 10_000

def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Los archivos NASA NEX-GDDP-CMIP6 pueden llamar a las coordenadas
    'latitude'/'longitude' o 'lat'/'lon' dependiendo del modelo/versión.
    Esta función las estandariza siempre a 'lat' y 'lon'.
    """
    rename_map = {}
    for candidate in ("latitude", "Latitude", "LATITUDE"):
        if candidate in ds.coords and "lat" not in ds.coords:
            rename_map[candidate] = "lat"
    for candidate in ("longitude", "Longitude", "LONGITUDE"):
        if candidate in ds.coords and "lon" not in ds.coords:
            rename_map[candidate] = "lon"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds

class NASADataDownloader:
    def __init__(self, base_url, model, download_path):
        self.base_url = base_url
        self.model = model
        self.download_path = os.path.join(download_path, "NASA_data")

        
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def download_all(self):
        print("="*60)
        print(f"Iniciando Extracción NASA NEX-GDDP-CMIP6 | Modelo: {self.model}")
        print("="*60)

        for scenario in SCENARIOS:
            for var in VARIABLES:
                for year in YEARS:
                    self._process_download(scenario, var, year)

    def _process_download(self, scenario, var, year):
        # Intentamos primero con la versión v2.0 detectada en tus capturas
        versions = ["_v2.0", ""] # Prioridad a v2.0, luego la estándar
        success = False

        for v in versions:
            filename = f"{var}_day_{self.model}_{scenario}_{ENSEMBLE}_gn_{year}{v}.nc"
            url = f"{self.base_url}/{self.model}/{scenario}/{ENSEMBLE}/{var}/{filename}"
            
            # Crear subcarpeta local organizada
            local_dir = os.path.join(self.download_path, self.model, scenario, var)
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, filename)

            if os.path.exists(local_path):
                print(f"[SKIP] Ya existe: {filename}")
                return

            # Intentar descarga
            if self._request_file(url, local_path, filename):
                success = True
                break # Si v2.0 funcionó, no buscamos la versión normal
        
        if not success:
            print(f"[ERROR] No se pudo encontrar {var} para {year} en {scenario}")

    def _request_file(self, url, dest, name):
        try:
            # stream=True para no saturar la RAM con archivos grandes
            with requests.get(url, stream=True, timeout=30) as r:
                if r.status_code == 404:
                    return False
                
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(dest, 'wb') as f, tqdm(
                    desc=name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
                return True
        except Exception as e:
            if os.path.exists(dest): os.remove(dest) # Limpiar archivo corrupto
            print(f"\n Error de conexión: {e}")
            return False

class NASAProcessor:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.data_path = os.path.join(BASE_PATH, "NASA_data")
        self.nasa_lat = CA_BOUNDS["lat"]
        self.nasa_lon = (CA_BOUNDS["lon"][0] % 360, CA_BOUNDS["lon"][1] % 360)

    def _normalize_coords(self, ds: xr.Dataset) -> xr.Dataset:
        rename_map = {}
        for candidate in ("latitude", "Latitude", "LATITUDE"):
            if candidate in ds.coords and "lat" not in ds.coords:
                rename_map[candidate] = "lat"
        for candidate in ("longitude", "Longitude", "LONGITUDE"):
            if candidate in ds.coords and "lon" not in ds.coords:
                rename_map[candidate] = "lon"
        if rename_map:
            ds = ds.rename(rename_map)
        return ds

    def _load_and_prepare(self, filepath: str) -> xr.Dataset:
        ds = xr.open_dataset(filepath)
        ds = self._normalize_coords(ds)

        missing = [c for c in ("lat", "lon") if c not in ds.coords]
        if missing:
            raise KeyError(
                f"Coordenadas {missing} no encontradas en {filepath}. "
                f"Disponibles: {list(ds.coords)}"
            )
        if float(ds.lon.min()) < 0:
            ds = ds.assign_coords(lon=(ds.lon % 360))
        ds = ds.sortby("lat").sortby("lon")
        return ds

    def get_pending_points(self, target_year):
        print(f"[INFO] Filtrando puntos pendientes para {target_year}...")

        existing = self.collection.find(
            {"year": target_year, "es_proyeccion": True},
            {"location.coordinates": 1, "_id": 0},
        )
        blacklist = {
            tuple(c["location"]["coordinates"]) for c in existing if "location" in c
        }

        base_cursor = self.collection.find(
            {"year": 2025, "location": {"$exists": True}},
            {
                "_id": 0,
                "location": 1, "crop_id": 1, "crop_name": 1, "is_target": 1,
                "awc_r": 1, "ph1to1h2o_r": 1, "cec7_r": 1, "ecec_r": 1,
                "om_r": 1, "ec_r": 1, "ksat_r": 1, "dbthirdbar_r": 1,
                "sandtotal_r": 1, "silttotal_r": 1, "claytotal_r": 1,
                "pbray1_r": 1, "ptotal_r": 1, "sumbases_r": 1,
                "profundidad_efectiva_cm": 1, "processed_soil": 1,
            },
        )

        pending = [
            p for p in base_cursor
            if tuple(p["location"]["coordinates"]) not in blacklist
        ]
        print(f"[INFO] {len(pending)} puntos pendientes ({len(blacklist)} ya existentes omitidos).")
        return pending 

    def _extract_batch(self, ds_dict: dict, lats: np.ndarray, lons_adj: np.ndarray) -> dict:
        lat_idx = xr.DataArray(lats,     dims="points")
        lon_idx = xr.DataArray(lons_adj, dims="points")

        results = {}
        for var, ds in ds_dict.items():
            data = ds[var].sel(lat=lat_idx, lon=lon_idx, method="nearest")

            if var == "pr":
                vals = (data * 86400).sum(dim="time").values
            elif var in ("tas", "tasmax", "tasmin"):
                vals = (data - 273.15).mean(dim="time").values
            else:
                vals = data.mean(dim="time").values

            results[var] = vals
        return results

    def run_projections(self, model, future_years):
        VAR_RENAME = {
        "tas":    "tmean",
        "tasmax": "tmax",
        "tasmin": "tmin",
        "hurs":   "hurs",
        "pr":     "pr",
        }
        SCENARIO_SUFFIX = {
            "ssp245": "245",
            "ssp585": "585",
        }
        variables = list(VAR_RENAME.keys())

        for year in future_years:

            # --- Estado actual en MongoDB ---
            already_done = self.collection.count_documents({
                "year": year,
                "es_proyeccion": True,
            })
            print(f"\n[INFO] {year}: {already_done} documentos ya en MongoDB.")

            base_points = self.get_pending_points(year)
            if not base_points:
                print(f"[INFO] {year} completo. Nada que procesar.")
                continue

            n        = len(base_points)
            all_lats = np.array([p["location"]["coordinates"][1] for p in base_points])
            all_lons = np.array([p["location"]["coordinates"][0] % 360 for p in base_points])

            # Acumula los valores de ambos escenarios antes de escribir
            # estructura: {suffix: {var_renamed: np.ndarray(n)}}
            scenario_matrices = {}

            for scenario, suffix in SCENARIO_SUFFIX.items():

                # --- Carga de datasets ---
                print(f"\n[INFO] Cargando datasets ({year} | {scenario})...")
                ds_dict = {}
                for var in variables:
                    var_dir = os.path.join(self.data_path, model, scenario, var)
                    files   = [f for f in os.listdir(var_dir) if str(year) in f]
                    if not files:
                        print(f"  [WARN] No encontrado: {var} {year} {scenario}")
                        continue
                    try:
                        ds = self._load_and_prepare(os.path.join(var_dir, files[0]))
                        ds_dict[var] = ds.sel(
                            lat=slice(self.nasa_lat[0], self.nasa_lat[1]),
                            lon=slice(self.nasa_lon[0], self.nasa_lon[1]),
                        ).load()
                        ds.close()
                        print(f"  [OK] {var} — lat x lon: {ds_dict[var].dims['lat']} x {ds_dict[var].dims['lon']}")
                    except Exception as e:
                        print(f"  [ERROR] Cargando {var}: {e}")

                if not ds_dict:
                    print(f"  [ERROR] Sin variables. Saltando {scenario}.")
                    continue

                # --- Extraccion vectorizada ---
                clima_matrix = {var: np.empty(n, dtype=np.float32) for var in ds_dict}
                n_batches    = (n + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"[INFO] Extrayendo {n} puntos en {n_batches} lotes ({scenario})...")

                for start in tqdm(range(0, n, BATCH_SIZE), desc=f"{year}-{scenario}"):
                    end           = min(start + BATCH_SIZE, n)
                    batch_results = self._extract_batch(ds_dict, all_lats[start:end], all_lons[start:end])
                    for var, vals in batch_results.items():
                        clima_matrix[var][start:end] = vals

                # Renombrar variables y guardar con sufijo de escenario
                scenario_matrices[suffix] = {
                    VAR_RENAME[var]: clima_matrix[var] for var in clima_matrix
                }

            if not scenario_matrices:
                print(f"[ERROR] Sin datos de ningun escenario para {year}. Saltando.")
                continue

            # --- Escritura en MongoDB: un documento por ubicacion/año ---
            print("[INFO] Escribiendo en MongoDB...")
            updates = []
            errors  = 0

            for i, point in enumerate(tqdm(base_points, desc="MongoDB bulk")):
                try:
                    coords = point["location"]["coordinates"]

                    # Base del documento: suelo + identificacion (heredado de 2025)
                    new_doc = point.copy()
                    new_doc["year"]            = year
                    new_doc["es_proyeccion"]   = True
                    new_doc["referencia_base"] = 2025

                    # Añadir campos climaticos aplanados por escenario
                    # resultado: tmax_245, tmax_585, tmean_245, tmean_585, etc.
                    for suffix, renamed_vars in scenario_matrices.items():
                        for var_name, arr in renamed_vars.items():
                            new_doc[f"{var_name}_{suffix}"] = float(arr[i])

                    updates.append(UpdateOne(
                        {"location.coordinates": coords, "year": year, "es_proyeccion": True},
                        {"$set": new_doc},
                        upsert=True,
                    ))

                    if len(updates) >= 1000:
                        self.collection.bulk_write(updates)
                        updates = []

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"\n  [WARN] Punto {i}: {e}")

            if updates:
                self.collection.bulk_write(updates)

            print(f"[OK] Finalizado: {year} ({errors} errores).")