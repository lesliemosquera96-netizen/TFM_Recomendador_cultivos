import os
from pymongo import MongoClient
from src.config import BASE_PATH, MONGO_URI, DB_NAME, COLLECTION_NAME, CA_BOUNDS, SAMPLES_PER_YEAR, YEARS_TO_PROCESS
from src.extractors.cdl_extractor import CDLExtractor
from src.extractors.weather_extractor import ClimateExtractor
from src.extractors.soil_extractor import SoilDownloader, SoilEnricher
from src.extractors.f_weather_extractor import NASADataDownloader, NASAProcessor, BASE_URL, MODEL


def chunk_years(list_of_years, chunk_size):
    for i in range(0, len(list_of_years), chunk_size):
        yield list_of_years[i:i + chunk_size]


def run_batch_pipeline():

    # 1. Conexion a MongoDB
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db  = client[DB_NAME]
        col = db[COLLECTION_NAME]
        print(f"Conexion exitosa a MongoDB: '{MONGO_URI}'")
        print(f"Base de datos: '{DB_NAME}' | Coleccion: '{COLLECTION_NAME}'")
        col.create_index([("location", "2dsphere")])
        col.create_index([("location.coordinates", 1)])
        col.create_index([("location.coordinates", 1), ("year", 1), ("scenario", 1)])
        col.create_index([("year", 1), ("scenario", 1), ("es_proyeccion", 1)])
        col.create_index([("processed_soil", 1)])
        print("Indices verificados.")
    except Exception as e:
        print(f"ERROR: No se pudo conectar a MongoDB: {e}")
        return

    # 2. Cultivos (CDL - USDA)
    all_years = list(range(2008, 2025))
    bloques   = list(chunk_years(all_years, 3))
    print(f"Total de años a procesar: {len(all_years)}")
    print(f"Dividido en {len(bloques)} bloques de 3 años.\n")

    for idx, bloque in enumerate(bloques):
        print(f"--- INICIANDO BLOQUE {idx + 1}/{len(bloques)}: {bloque} ---")

        for year in bloque:
            print(f"\n[AÑO {year}]")

            if col.count_documents({"year": year}) > 0:
                print(f"El año {year} ya existe en MongoDB. Saltando...")
                continue

            etl = CDLExtractor(BASE_PATH, year)
            try:
                if os.path.exists(etl.tif_path):
                    print(f"Datos del año {year} ya descargados.")
                else:
                    print(f"Iniciando descarga y extraccion...")
                etl.extract()

                print(f"Iniciando seleccion de puntos...")
                data_points = etl.selection(CA_BOUNDS, n_points=SAMPLES_PER_YEAR)

                if data_points:
                    etl.load(data_points, col)
                else:
                    print(f"No se generaron puntos para {year}.")
            except Exception as e:
                print(f"Error en el año {year}: {e}")
                continue

        print(f"\nBloque {idx + 1} finalizado.")

    print("Pipeline CDL completado: datos 2008-2025 en MongoDB.")

    # 3. Datos climaticos historicos (PRISM)
    
    print("\n[DATOS CLIMATICOS] Verificando archivos PRISM...")
    try:
        cli = ClimateExtractor(BASE_PATH)
        cli.download_range(min(YEARS_TO_PROCESS), max(YEARS_TO_PROCESS))
        cli.update_climate_in_mongo()
        print("Datos climaticos historicos cargados.")
    except Exception as e:
        print(f"Error en datos climaticos historicos: {e}")

    # 4. Datos climaticos futuros (NASA NEX-GDDP-CMIP6)
    
    print("\n[NASA] Iniciando descarga de datos climaticos futuros...")
    try:
        downloader = NASADataDownloader(
            base_url=BASE_URL,
            model=MODEL,
            download_path=BASE_PATH,
        )
        downloader.download_all()
        print("Fase de descarga NASA completada.")
    except Exception as e:
        print(f"Error durante la descarga NASA: {e}")
        return

    print("\n[NASA] Procesando e integrando en MongoDB...")
    try:
        projector = NASAProcessor()
        projector.run_projections(
        model=MODEL,
        future_years=[2030, 2040],  
        )
        print("Datos climaticos futuros integrados en MongoDB.")
    except Exception as e:
        print(f"Error durante la integracion NASA en MongoDB: {e}")
        raise

    # 5. Datos de suelos (gSSURGO)

    print("\n[SUELOS] Verificando base de datos gSSURGO_CA...")
    soil_ready = False
    try:
        soil_dl    = SoilDownloader()
        soil_ready = soil_dl.run_pipeline()
    except Exception as e:
        print(f"Error en descarga de suelos: {e}")

    if soil_ready:
        print("[SUELOS] Enriqueciendo MongoDB con datos edaficos...")
        try:
            enricher = SoilEnricher()
            enricher.update_soil_in_mongo()
            print("Datos de suelo cargados.")
        except Exception as e:
            print(f"Error en carga de datos de suelo: {e}")
    else:
        print("[SUELOS] Descarga no completada. Enriquecimiento omitido.")


if __name__ == "__main__":
    run_batch_pipeline()