"""
================================================================================
 CONSTRUCCIÓN DE LA BASE COMPLETA DE CALIFORNIA PARA LA APP
================================================================================
Genera un único Parquet con suelo + clima presente (2025) + clima futuro
(2030 y 2040, escenarios SSP2-4.5 y SSP5-8.5) para TODA California, no solo
las zonas de cultivo.

La malla base se toma de la rejilla de PRISM (4 km), que cubre todo el estado.
Reutiliza la lógica de extracción de tus scripts existentes:
  - Suelo: cruce espacial con gSSURGO (de soil_extractor.SoilEnricher)
  - Clima presente: muestreo de rasters PRISM (de weather_extractor)
  - Clima futuro: muestreo vectorizado de NetCDF CMIP6 (de f_weather_extractor)

No usa MongoDB: extrae directamente a un DataFrame y guarda en Parquet.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform as reproject_coords
import xarray as xr
from tqdm import tqdm

import numpy, geopandas, rasterio, pyogrio
print("NumPy:", numpy.__version__)
print("Todo importa OK")


# Ajusta esta ruta a la raíz de tus datos locales
BASE_PATH = r"C:\Users\lesli\Downloads"   # donde están CLIMATE_PRISM, NASA_data, soil_data
SALIDA_PARQUET = r"C:\Users\lesli\Documents\TFM\Data\base_california_app.parquet"

# Límites de California (mismos que tu config)
CA_BOUNDS = {
    "lat": (32.5, 42.0),
    "lon": (-124.5, -114.1),
}

# ==============================================================================
#  PASO 1 — Generar la malla base desde un raster de PRISM
# ==============================================================================
def generar_malla_prism(base_path, anio_ref=2025, elemento_ref="tmax"):
    """
    Toma un raster de PRISM y devuelve las coordenadas (lon, lat) de todas
    sus celdas dentro de California. Esta es la malla base de toda California.
    """
    climate_dir = os.path.join(base_path, "CLIMATE_PRISM")
    folder = os.path.join(climate_dir, f"{anio_ref}_{elemento_ref}")
    bil = None
    for f in os.listdir(folder):
        if f.endswith(".bil"):
            bil = os.path.join(folder, f)
            break
    if bil is None:
        raise FileNotFoundError(f"No se encontró .bil en {folder}")

    print(f"[MALLA] Generando malla base desde {bil}")
    with rasterio.open(bil) as src:
        # Construir la rejilla de centros de celda
        n_rows, n_cols = src.height, src.width
        # Submuestreo opcional: 1 = todas las celdas (4km). 2 = cada 8km, etc.
        step = 1
        rows = np.arange(0, n_rows, step)
        cols = np.arange(0, n_cols, step)
        col_grid, row_grid = np.meshgrid(cols, rows)
        xs, ys = rasterio.transform.xy(src.transform,
                                       row_grid.ravel(), col_grid.ravel())
        xs = np.array(xs)
        ys = np.array(ys)

        # Reproyectar a WGS84 (lon/lat) si el raster está en otra proyección
        if src.crs and src.crs.to_epsg() != 4326:
            lon, lat = reproject_coords(src.crs, "EPSG:4326", xs, ys)
            lon = np.array(lon)
            lat = np.array(lat)
        else:
            lon, lat = xs, ys

    # Filtrar a los límites de California
    mask = (
        (lon >= CA_BOUNDS["lon"][0]) & (lon <= CA_BOUNDS["lon"][1]) &
        (lat >= CA_BOUNDS["lat"][0]) & (lat <= CA_BOUNDS["lat"][1])
    )
    malla = pd.DataFrame({"lon": lon[mask], "lat": lat[mask]})
    print(f"[MALLA] {len(malla):,} puntos dentro de California")
    return malla


# ==============================================================================
#  PASO 2 — Extraer clima presente (PRISM) para la malla
# ==============================================================================
# Añadir vpd a los elementos de PRISM
ELEMENTOS_PRISM = ["tmin", "tmax", "tmean", "ppt", "vpdmax", "vpdmin"]

def get_raster_path(base_path, year, element):
    folder = os.path.join(base_path, "CLIMATE_PRISM", f"{year}_{element}")
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.endswith(".bil"):
                return os.path.join(folder, f)
    return None

def extraer_clima_presente(malla, base_path, year=2025):
    """Muestrea los rasters de PRISM en cada punto de la malla (vectorizado)."""
    print(f"\n[PRISM] Extrayendo clima {year}...")
    df = malla.copy()
    for element in ELEMENTOS_PRISM:
        path = get_raster_path(base_path, year, element)
        if path is None:
            print(f"  [WARN] No se encontró raster {element} {year}")
            df[element] = np.nan
            continue
        with rasterio.open(path) as src:
            # Reproyectar todos los puntos de una vez
            xs, ys = reproject_coords("EPSG:4326", src.crs,
                                      malla["lon"].tolist(), malla["lat"].tolist())
            coords = list(zip(xs, ys))
            vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
            vals[vals < -9000] = np.nan  # PRISM usa -9999 como nulo
        df[element] = vals
        print(f"  [OK] {element}: {np.isnan(vals).sum():,} nulos de {len(vals):,}")
    return df


# ==============================================================================
#  PASO 3 — Extraer suelo (gSSURGO) para la malla
#  Reutiliza la lógica de SoilEnricher._get_soil_data_layer
# ==============================================================================
SOIL_ELEMENTS = [
    'awc_r', 'ph1to1h2o_r', 'cec7_r', 'ecec_r', 'om_r', 'ec_r', 'ksat_r',
    'dbthirdbar_r', 'sandtotal_r', 'silttotal_r', 'claytotal_r',
    'pbray1_r', 'ptotal_r', 'sumbases_r',
]

def cargar_capa_suelo(gdb_path):
    """Carga y prepara las capas de gSSURGO (idéntico a tu SoilEnricher)."""
    print("\n[SUELO] Cargando capas de la GDB...")
    mupoly = gpd.read_file(gdb_path, layer="MUPOLYGON", engine="pyogrio")
    mupoly.columns = mupoly.columns.str.lower()
    mupoly = mupoly[['mukey', 'geometry']]
    mupoly["mukey"] = mupoly["mukey"].astype(str)

    comp = gpd.read_file(gdb_path, layer="component", engine="pyogrio")
    comp.columns = comp.columns.str.lower()

    horiz = gpd.read_file(gdb_path, layer="chorizon", engine="pyogrio")
    horiz.columns = horiz.columns.str.lower()

    columnas_fisicas = SOIL_ELEMENTS + ['cokey', 'hzdept_r', 'hzdepb_r']
    horiz_top = horiz[horiz['hzdept_r'] == 0][columnas_fisicas]
    comp_major = comp[comp['majcompflag'].str.lower() == 'yes'][['mukey', 'cokey']].copy()
    comp_major["mukey"] = comp_major["mukey"].astype(str)
    attr_merged = comp_major.merge(horiz_top, on='cokey', how='inner')
    return mupoly.to_crs("EPSG:4326"), attr_merged

def extraer_suelo(malla, gdb_path, chunk_size=50000):
    """Cruce espacial de la malla con gSSURGO, reproyectando a metros."""
    map_polys, soil_data = cargar_capa_suelo(gdb_path)

    # Reproyectar los polígonos a California Albers (metros) UNA vez
    CRS_METROS = "EPSG:3310"  # California Albers, en metros
    map_polys_m = map_polys.to_crs(CRS_METROS)

    resultados = []
    n = len(malla)
    for start in tqdm(range(0, n, chunk_size), desc="[SUELO] Chunks"):
        end = min(start + chunk_size, n)
        chunk = malla.iloc[start:end].copy().reset_index(drop=True)
        chunk["_punto_id"] = range(len(chunk))

        gdf_points = gpd.GeoDataFrame(
            chunk,
            geometry=gpd.points_from_xy(chunk["lon"], chunk["lat"]),
            crs="EPSG:4326",
        ).to_crs(CRS_METROS)   # reproyectar los puntos a metros también

        # Ahora sjoin_nearest mide en METROS (max_distance en metros)
        mapped = gpd.sjoin_nearest(gdf_points, map_polys_m, how="left",
                                   max_distance=5000)  # 5000 m = 5 km
        if 'mukey' not in mapped.columns and 'mukey_left' in mapped.columns:
            mapped = mapped.rename(columns={'mukey_left': 'mukey'})
        mapped["mukey"] = mapped["mukey"].astype(str)

        mapped = mapped.drop_duplicates(subset="_punto_id", keep="first")
        merged = mapped.merge(soil_data, on='mukey', how='left')
        merged = merged.drop_duplicates(subset="_punto_id", keep="first")
        merged = merged.sort_values("_punto_id").reset_index(drop=True)

        cols = SOIL_ELEMENTS + ['hzdepb_r']
        sub = merged[cols].copy()
        sub = sub.rename(columns={'hzdepb_r': 'profundidad_efectiva_cm'})
        sub["lon"] = chunk["lon"].values
        sub["lat"] = chunk["lat"].values
        resultados.append(sub)

    df_suelo = pd.concat(resultados, ignore_index=True)
    return df_suelo

# ==============================================================================
#  PASO 4 — Extraer clima futuro (CMIP6) para la malla
#  Reutiliza la lógica vectorizada de f_weather_extractor
# ==============================================================================
MODEL = "MPI-ESM1-2-HR"
SCENARIOS = {"ssp245": "245", "ssp585": "585"}
FUTURE_YEARS = [2030, 2040]
VAR_RENAME = {"tas": "tmean", "tasmax": "tmax", "tasmin": "tmin", "pr": "pr"}
FUT_VARS = list(VAR_RENAME.keys())

def _load_nc(filepath):
    ds = xr.open_dataset(filepath)
    # normalizar coords
    rename = {}
    for c in ("latitude", "Latitude"):
        if c in ds.coords and "lat" not in ds.coords:
            rename[c] = "lat"
    for c in ("longitude", "Longitude"):
        if c in ds.coords and "lon" not in ds.coords:
            rename[c] = "lon"
    if rename:
        ds = ds.rename(rename)
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360))
    return ds.sortby("lat").sortby("lon")

def extraer_clima_futuro(malla, base_path):
    """Extrae el clima futuro de los NetCDF CMIP6 para cada punto (vectorizado)."""
    nasa_path = os.path.join(base_path, "NASA_data")
    lats = xr.DataArray(malla["lat"].values, dims="points")
    lons = xr.DataArray(malla["lon"].values % 360, dims="points")

    df = malla.copy()
    for year in FUTURE_YEARS:
        for scenario, suffix in SCENARIOS.items():
            for var in FUT_VARS:
                var_dir = os.path.join(nasa_path, MODEL, scenario, var)
                if not os.path.isdir(var_dir):
                    print(f"  [WARN] No existe {var_dir}")
                    continue
                files = [f for f in os.listdir(var_dir) if str(year) in f]
                if not files:
                    continue
                ds = _load_nc(os.path.join(var_dir, files[0]))
                data = ds[var].sel(lat=lats, lon=lons, method="nearest")

                if var == "pr":
                    vals = (data * 86400).sum(dim="time").values
                else:
                    vals = (data - 273.15).mean(dim="time").values

                col = f"{VAR_RENAME[var]}_{suffix}_{year}"
                df[col] = vals.astype(np.float32)
                ds.close()
            print(f"  [OK] Futuro {year} {scenario} extraído")
    return df


# ==============================================================================
#  PASO 5 — Rellenar faltantes con vecino más cercano (con umbral de distancia)
# ==============================================================================
def rellenar_vecino_cercano(df, columnas, umbral_grados=0.1):
    """
    Rellena nulos de cada columna con el valor del vecino más cercano que
    sí tenga dato, siempre que esté dentro del umbral de distancia (en grados).
    Marca como no fiables los puntos cuyo vecino está más lejos del umbral.
    """
    from scipy.spatial import cKDTree

    df = df.copy()
    coords = df[["lon", "lat"]].to_numpy()

    for col in columnas:
        nulos = df[col].isna()
        if not nulos.any():
            continue
        validos = ~nulos
        if validos.sum() == 0:
            continue
        tree = cKDTree(coords[validos])
        dist, idx = tree.query(coords[nulos])
        valores_validos = df.loc[validos, col].to_numpy()
        rellenos = valores_validos[idx]
        # Solo rellenar si el vecino está dentro del umbral
        rellenos[dist > umbral_grados] = np.nan
        df.loc[nulos, col] = rellenos
    return df


# ==============================================================================
#  ORQUESTACIÓN
# ==============================================================================
def main():
    # 1. Malla base de California
    malla = generar_malla_prism(BASE_PATH, anio_ref=2025, elemento_ref="tmax")

    # 2. Clima presente
    df = extraer_clima_presente(malla, BASE_PATH, year=2025)
    # Calcular vpdmean presente (para el clustering)
    df["vpdmean"] = (df["vpdmax"] + df["vpdmin"]) / 2

    # 3. Suelo
    gdb_path = os.path.join(BASE_PATH, "soil_data", "gSSURGO_CA.gdb")
    df_suelo = extraer_suelo(malla, gdb_path)
    # Unir suelo por coordenadas (mismo orden de filas que la malla)
    for col in SOIL_ELEMENTS + ["profundidad_efectiva_cm"]:
        df[col] = df_suelo[col].values

    # 4. Clima futuro
    df_fut = extraer_clima_futuro(malla, BASE_PATH)
    cols_fut = [c for c in df_fut.columns if c not in ("lon", "lat")]
    for col in cols_fut:
        df[col] = df_fut[col].values

    # 5. Rellenar faltantes con vecino cercano (clima y suelo)
    cols_rellenar = ELEMENTOS_PRISM + SOIL_ELEMENTS + ["profundidad_efectiva_cm"] + cols_fut
    df = rellenar_vecino_cercano(df, cols_rellenar, umbral_grados=0.1)

    # Marcar puntos sin datos suficientes (los que siguen con nulos en variables clave)
    vars_clave = ["tmax", "tmin", "ppt", "ph1to1h2o_r", "profundidad_efectiva_cm"]
    df["datos_completos"] = ~df[vars_clave].isna().any(axis=1)

    # 6. Guardar
    df.to_parquet(SALIDA_PARQUET, index=False)
    print(f"\n[OK] Base guardada en {SALIDA_PARQUET}")
    print(f"     Puntos totales: {len(df):,}")
    print(f"     Con datos completos: {df['datos_completos'].sum():,}")
    print(f"     Sin datos suficientes: {(~df['datos_completos']).sum():,}")


if __name__ == "__main__":
    main()