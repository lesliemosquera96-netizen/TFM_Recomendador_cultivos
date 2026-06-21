import os
import sys
import geopandas as gpd
import polars as pl
import fiona

# Ruta absoluta a tu base de datos gSSURGO descargada
GDB_PATH = r"C:\Users\lesli\Downloads\soil_data\gSSURGO_CA.gdb"

print("Analizando la estructura de la Geodatabase local de California...")

if not os.path.exists(GDB_PATH):
    print(f" Error: No se encontró el archivo en la ruta especificada:\n{GDB_PATH}")
    sys.exit()

# 1. Listar todas las capas (tables) disponibles dentro del .gdb
try:
    layers = fiona.listlayers(GDB_PATH)
    print(f"\n Conexión exitosa. Se encontraron {len(layers)} tablas dentro del .gdb.")
    print("\n Primeras 15 tablas detectadas en gSSURGO (para identificar las de suelos):")
    for layer in layers[:15]:
        print(f" - {layer}")
except Exception as e:
    print(f" Error al listar las capas del .gdb: {e}")
    sys.exit()

# 2. Inspeccionar la tabla principal de horizontes de suelo 
target_layer = "chorizon" if "chorizon" in layers else layers[0]

print(f"\n Leyendo una muestra de la tabla objetivo: '{target_layer}'...")
try:
    # Leemos solo las primeras 5000 filas para no saturar la RAM, ya que estas bases de datos son gigantescas
    gdf_sample = gpd.read_file(GDB_PATH, layer=target_layer, rows=5000)
    
    # Convertimos a Polars para mantener tu estándar de alta velocidad
    df_soil = pl.DataFrame(gdf_sample.drop(columns='geometry', errors='ignore'))
    
    print(f" Muestra de '{target_layer}' cargada correctamente en memoria.")
    print(f"Dimensiones de la muestra: {df_soil.shape[0]} filas, {df_soil.shape[1]} columnas.")
    
    # 3. Identificar qué variables de interés existen realmente en el archivo original
    variables_interes = ["claytotal_r", "sandtotal_r", "silttotal_r", "om_r", "ph1to1h2o_r", "awc_r", "hzdept_r", "hzdepb_r"]
    existing_vars = [v for v in variables_interes if v in df_soil.columns]
    
    print("AUDITORÍA DE NULOS EN EL ARCHIVO ORIGINAL (.gdb)")
    if existing_vars:
        # Calcular nulos locales en el gdb original
        null_analysis = df_soil.select([
            pl.col(v).null_count().alias(f"{v}_nulos") for v in existing_vars
        ])
        print(null_analysis)
        
        print("\n Primeras líneas de los datos de suelo puros:")
        with pl.Config(tbl_cols=-1):
            print(df_soil.select(existing_vars).head(10))
    else:
        print("No se encontraron las columnas estándar en esta capa. Las columnas disponibles son:")
        print(df_soil.columns[:20])

except Exception as e:
    print(f" No se pudo leer la capa '{target_layer}': {e}")