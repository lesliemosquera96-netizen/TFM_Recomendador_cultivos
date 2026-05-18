import os
import re
import time
import zipfile
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import geopandas as gpd
import pandas as pd
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm
from src.config import MONGO_URI, DB_NAME, COLLECTION_NAME, BASE_PATH
from concurrent.futures import ThreadPoolExecutor, as_completed


# Definir descarga Clase extracción de datos (webscrappig)
class SoilDownloader:
    def __init__(self):
        self.download_path = os.path.join(BASE_PATH, "soil_data")
        self.debug_path    = os.path.join(BASE_PATH, "debug_screenshots")
        self.box_url       = "https://nrcs.app.box.com/v/gateway/"
        self.zip_name      = "gSSURGO_CA.zip"
        self.gdb_folder    = "gSSURGO_CA.gdb"

        for path in [self.download_path, self.debug_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    # PIPELINE, primero se define una función que verifica si la base de datos se encuentra en local o no
    # si no se encuentra procede a descargar los datos, en caso contrario se salta la carga 

    def run_pipeline(self):
        gdb_full_path = os.path.join(self.download_path, self.gdb_folder)
        zip_full_path = os.path.join(self.download_path, self.zip_name)

        if os.path.exists(gdb_full_path):
            print("[OK] GDB ya existe. Saltando descarga.")
            return True

        if os.path.exists(zip_full_path):
            print("[OK] ZIP encontrado localmente. Descomprimiendo...")
            self._extract_and_clean(zip_full_path)
            return True

        print("[INFO] No se encontraron datos locales. Iniciando web scraping...")
        self._scrape_and_download()
        return True


    # Definir función de web scrping con selenium

    def _scrape_and_download(self):
        """
        Web scraping completo:
          1. Navega por Box hasta encontrar gSSURGO_CA.zip  (igual que el original)
          2. Extrae el file_id del page_source sin hacer clic en el archivo
          3. Navega directamente a la página individual del archivo
          4. Hace clic en el botón de descarga
          5. Espera a que Chrome termine la descarga
        """
        driver = self._build_driver()
        if driver is None:
            return

        try:
            wait = WebDriverWait(driver, 30)

            # Navegar hasta la carpeta
            print("[SELENIUM] Accediendo a Box...")
            driver.get(self.box_url)
            self._wait_and_screenshot(driver, 6, "01_raiz")

            print("[SELENIUM] Entrando en 'soils'...")
            wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "soils"))).click()
            self._wait_and_screenshot(driver, 4, "02_soils")

            print("[SELENIUM] Entrando en 'gSSURGO by State'...")
            wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "gSSURGO by State"))).click()
            self._wait_and_screenshot(driver, 6, "03_gssurgo_by_state")

            # Localizar el archivo 
            found_page = self._find_file_across_pages(driver, wait)
            if not found_page:
                print("[ERROR] No se encontró gSSURGO_CA.zip en ninguna página.")
                return

            # Extraer file_id del page_source
            # El archivo ya es visible — su ID está en el HTML de la página
            file_id = self._extract_file_id_from_source(driver)
            if not file_id:
                print("[ERROR] No se pudo extraer el file_id del HTML.")
                return

            # Navegar directamente a la página del archivo 
            # Esto evita el clic en la lista que descargaba la carpeta entera
            file_url = f"https://nrcs.app.box.com/v/gateway/file/{file_id}"
            print(f"[SELENIUM] Navegando a la página del archivo: {file_url}")
            driver.get(file_url)
            self._wait_and_screenshot(driver, 5, "06_pagina_archivo")

            # Clic en el botón de descarga 
            print("[SELENIUM] Buscando botón de descarga...")
            if not self._click_download_button(driver):
                print("[ERROR] No se encontró el botón de descarga.")
                return

            # Esperar a que termine la descarga 
            self._wait_for_download_completion()

            # Descomprimir
            zip_path = os.path.join(self.download_path, self.zip_name)
            self._extract_and_clean(zip_path)

        except Exception as e:
            print(f"[ERROR] Fallo inesperado: {e}")
            self._screenshot(driver, "99_error")
        finally:
            print("[SELENIUM] Cerrando navegador.")
            driver.quit()


    # Buscar pagina por pagina el archivo de suelos para california gSSURGO_CA 

    def _find_file_across_pages(self, driver, wait):
        """
        Recorre páginas buscando gSSURGO_CA.zip.
        Retorna True cuando el archivo es visible, False si no se encuentra.
        No hace clic en el archivo — solo confirma que está en pantalla.
        """
        file_xpaths = [
            "//div[contains(@class,'item-name')]//span[contains(text(),'gSSURGO_CA.zip')]",
            "//span[contains(@class,'filename') and contains(text(),'gSSURGO_CA.zip')]",
            "//div[contains(@class,'bdl-GridView-row')]//span[contains(text(),'gSSURGO_CA.zip')]",
            "//*[contains(text(),'gSSURGO_CA.zip')]",
        ]

        for num_pag in range(1, 6):
            print(f"[SELENIUM] Analizando página {num_pag}...")
            self._scroll_page(driver)
            self._screenshot(driver, f"04_pagina_{num_pag}")

            for xpath in file_xpaths:
                try:
                    WebDriverWait(driver, 8).until(
                        EC.presence_of_element_located((By.XPATH, xpath))
                    )
                    print(f"[OK] Archivo visible en página {num_pag}.")
                    self._screenshot(driver, f"05_archivo_encontrado_pag{num_pag}")
                    return True  # ← solo confirmamos visibilidad, sin clic
                except Exception:
                    continue

            print(f"[INFO] No encontrado en página {num_pag}. Cambiando de página...")
            if not self._click_next_page(driver):
                print("[INFO] No hay más páginas.")
                break

        return False

    # Una vez he identificado el archivo, extraer file_id DEL HTML para descargar

    def _extract_file_id_from_source(self, driver):
        """
        Lee el page_source cuando gSSURGO_CA.zip ya es visible y extrae
        el file_id buscando el número ID más cercano al nombre del archivo.
        """
        source = driver.page_source

        # Buscar el fragmento de HTML alrededor del nombre del archivo
        idx = source.find("gSSURGO_CA")
        if idx == -1:
            print("[WARN] gSSURGO_CA no encontrado en page_source.")
            return None

        # Tomar 1000 caracteres alrededor del nombre para buscar el ID
        fragment = source[max(0, idx - 1000): idx + 1000]

        # Patrones más comunes donde Box coloca el file_id
        patterns = [
            r'/file/(\d{10,})',          # /file/2053602487091
            r'data-fileid="(\d{10,})"',  # data-fileid="..."
            r'data-item-id="(\d{10,})"', # data-item-id="..."
            r'"file_id"\s*:\s*"(\d+)"',  # JSON: "file_id": "..."
            r'"id"\s*:\s*"(\d{10,})"',   # JSON: "id": "..."
        ]

        for pattern in patterns:
            match = re.search(pattern, fragment)
            if match:
                file_id = match.group(1)
                print(f"[OK] file_id='{file_id}' extraído con patrón '{pattern}'")
                return file_id

        # Si no encontró nada cerca del nombre, buscar en todo el source
        print("[INFO] No encontrado cerca del nombre. Buscando en todo el source...")
        match = re.search(r'/file/(\d{10,})', source)
        if match:
            file_id = match.group(1)
            print(f"[OK] file_id='{file_id}' encontrado en el source completo.")
            return file_id

        print("[ERROR] No se pudo extraer el file_id del page_source.")
        return None


    #  CLIC EN BOTÓN DE DESCARGA

    def _click_download_button(self, driver):
        """
        Hace clic en el botón de descarga de la página individual del archivo.
        Retorna True si lo encontró y pulsó, False si no.
        """
        selectors = [
            "[data-testid='download-button']",
            "button[aria-label='Download']",
            "button[aria-label='Descargar']",
            "[title='Download']",
            "[title='Descargar']",
            "button.bcpr-btn--download",
        ]

        for selector in selectors:
            try:
                btn = WebDriverWait(driver, 8).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                print(f"[OK] Botón de descarga encontrado: {selector}")
                self._screenshot(driver, "07_boton_descarga")
                driver.execute_script("arguments[0].click();", btn)
                print("[OK] Clic en botón de descarga realizado.")
                return True
            except Exception:
                continue

        # Debug: listar botones si ningún selector funcionó
        print("[DEBUG] Botones disponibles en la página:")
        for btn in driver.find_elements(By.TAG_NAME, "button"):
            aria  = btn.get_attribute("aria-label") or ""
            title = btn.get_attribute("title") or ""
            text  = btn.text.strip()
            if aria or title or text:
                print(f"  aria='{aria}' | title='{title}' | text='{text}'")

        return False

    # MONITOREO DE DESCARGA

    def _wait_for_download_completion(self):
        """
        Espera a que Chrome termine de descargar gSSURGO_CA.zip.
        Muestra el tamaño descargado en tiempo real.
        Timeout máximo: 2 horas.
        """
        print("[DOWNLOAD] Monitoreando descarga...")
        timeout = 0

        while timeout < 7200:
            files       = os.listdir(self.download_path)
            downloading = [f for f in files if f.endswith(".crdownload") or f.endswith(".tmp")]
            zip_done    = self.zip_name in files

            if zip_done and not downloading:
                size_mb = os.path.getsize(
                    os.path.join(self.download_path, self.zip_name)
                ) / (1024 * 1024)
                print(f"\n[OK] Descarga completada: {self.zip_name} ({size_mb:.1f} MB)")
                return

            if downloading:
                tmp_file = os.path.join(self.download_path, downloading[0])
                try:
                    size_mb = os.path.getsize(tmp_file) / (1024 * 1024)
                    print(f"\r  Descargando... {size_mb:.1f} MB", end="", flush=True)
                except Exception:
                    pass

            time.sleep(10)
            timeout += 10

        print("\n[WARN] Timeout de descarga alcanzado (2h).")

    # DESCOMPRESIÓN
    

    def _extract_and_clean(self, zip_path):
        """Descomprime el ZIP mostrando progreso y elimina el ZIP al terminar."""
        print(f"[EXTRACT] Descomprimiendo en {self.download_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.namelist()
            total   = len(members)
            for i, member in enumerate(members, 1):
                zip_ref.extract(member, self.download_path)
                print(f"\r  Extrayendo {i}/{total}: {member[:60]}", end="", flush=True)
        print("\n[OK] Descompresión completada.")
        os.remove(zip_path)
        print("[OK] ZIP eliminado. GDB lista para usar.")

    # CONFIGURACIÓN DEL DRIVER

    def _build_driver(self):
        """
        Chrome headless con directorio de descarga apuntando a soil_data.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        prefs = {
            "download.default_directory": self.download_path,
            "download.prompt_for_download": False,
            "directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        try:
            driver_path = os.path.join(os.getcwd(), "chromedriver.exe")
            if not os.path.exists(driver_path):
                print(f"[ERROR] chromedriver.exe no encontrado en {driver_path}")
                return None

            service = ChromeService(executable_path=driver_path)
            driver  = webdriver.Chrome(service=service, options=chrome_options)

            # Habilitar descarga en modo headless
            driver.execute_cdp_cmd("Page.setDownloadBehavior", {
                "behavior": "allow",
                "downloadPath": self.download_path,
            })
            print("[OK] Navegador iniciado correctamente.")
            return driver

        except Exception as e:
            print(f"[ERROR] No se pudo iniciar Chrome: {e}")
            return None

    # HELPERS DE NAVEGACIÓN Y DEBUG

    def _scroll_page(self, driver):
        """Scroll gradual para forzar renderizado de listas virtualizadas."""
        for i in range(1, 6):
            driver.execute_script(f"window.scrollTo(0, {i * 400});")
            time.sleep(0.6)
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.5)

    def _click_next_page(self, driver):
        """Hace clic en el botón Siguiente de la paginación de Box."""
        selectors = [
            "[aria-label='Next Page']",
            "[aria-label='Página siguiente']",
            "[aria-label='Next']",
            "button.bdl-Pagination-navBtn--next",
        ]
        for selector in selectors:
            try:
                btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", btn)
                print(f"[OK] Avanzado a siguiente página: {selector}")
                time.sleep(6)
                return True
            except Exception:
                continue

        print("[DEBUG] Botones disponibles:")
        for btn in driver.find_elements(By.TAG_NAME, "button"):
            aria = btn.get_attribute("aria-label") or ""
            text = btn.text.strip()
            if aria or text:
                print(f"  aria='{aria}' | text='{text}'")

        return False

    def _screenshot(self, driver, name):
        path = os.path.join(self.debug_path, f"{name}.png")
        try:
            driver.save_screenshot(path)
            print(f"  [Screenshot: {path}]")
        except Exception:
            pass

    def _wait_and_screenshot(self, driver, seconds, name):
        time.sleep(seconds)
        self._screenshot(driver, name)



# Definir clase para cargar datos a mongodb
class SoilEnricher:
    def __init__(self):
        self.download_path = os.path.join(BASE_PATH, "soil_data")
        self.gdb_path = os.path.join(self.download_path, "gSSURGO_CA.gdb")
        self.soil_elements = [
            'awc_r', 'ph1to1h2o_r', 'cec7_r', 'ecec_r', 'om_r',
            'ec_r', 'ksat_r', 'dbthirdbar_r', 'sandtotal_r',
            'silttotal_r', 'claytotal_r', 'pbray1_r', 'ptotal_r', 'sumbases_r'
        ]

    def _get_soil_data_layer(self):
        print("Cargando y normalizando capas de la GDB de Suelos...")
        mupoly = gpd.read_file(self.gdb_path, layer="MUPOLYGON", engine="pyogrio")
        mupoly.columns = mupoly.columns.str.lower()
        mupoly = mupoly[['mukey', 'geometry']]
        mupoly["mukey"] = mupoly["mukey"].astype(str)
        comp = gpd.read_file(self.gdb_path, layer="component", engine="pyogrio")
        comp.columns = comp.columns.str.lower()
        horiz = gpd.read_file(self.gdb_path, layer="chorizon", engine="pyogrio")
        horiz.columns = horiz.columns.str.lower()
        columnas_fisicas = self.soil_elements + ['cokey', 'hzdept_r', 'hzdepb_r']
        horiz_top = horiz[horiz['hzdept_r'] == 0][columnas_fisicas]
        comp_major = comp[comp['majcompflag'].str.lower() == 'yes'][['mukey', 'cokey']].copy()
        comp_major["mukey"] = comp_major["mukey"].astype(str)
        attr_merged = comp_major.merge(horiz_top, on='cokey', how='inner')
        return mupoly.to_crs("EPSG:4326"), attr_merged

    # función que procesa UN solo chunk (será llamada en paralelo)
    def _process_chunk(self, chunk_index, chunk, map_polys, soil_data):
        """
        Procesa un bloque de coordenadas: cruza espacialmente con SSURGO
        y devuelve la lista de UpdateOne listos para bulk_write en MongoDB.
        """
        chunk_num = chunk_index + 1
        print(f"\n  [Hilo] Iniciando chunk {chunk_num} ({len(chunk)} puntos)...")

        # Convertir coordenadas a GeoDataFrame
        coords_list = [
            {'lon': loc['_id'][0], 'lat': loc['_id'][1], 'orig_id': loc['_id']}
            for loc in chunk
        ]
        df_coords = pd.DataFrame(coords_list)
        gdf_points = gpd.GeoDataFrame(
            df_coords,
            geometry=gpd.points_from_xy(df_coords.lon, df_coords.lat),
            crs="EPSG:4326"
        )

        # Cruce espacial
        points_mapped = gpd.sjoin(gdf_points, map_polys, how="left", predicate="within")

        if 'mukey' not in points_mapped.columns and 'mukey_left' in points_mapped.columns:
            points_mapped = points_mapped.rename(columns={'mukey_left': 'mukey'})

        points_mapped["mukey"] = points_mapped["mukey"].astype(str)
        final_chunk_data = points_mapped.merge(soil_data, on='mukey', how='left')

        # Construir lista de updates
        updates = []
        for row in final_chunk_data.itertuples():
            payload = {}
            for element in self.soil_elements:
                val = getattr(row, element, None)
                payload[element] = float(val) if pd.notnull(val) else None
            depth_val = getattr(row, 'hzdepb_r', None)
            payload["profundidad_efectiva_cm"] = float(depth_val) if pd.notnull(depth_val) else None
            payload["processed_soil"] = True
            updates.append(UpdateOne({"location.coordinates": row.orig_id}, {"$set": payload}))

        print(f"  [Hilo] Chunk {chunk_num} listo: {len(updates)} updates preparados.")
        return chunk_num, updates  # ← devuelve los datos, no escribe en Mongo directamente

    def update_soil_in_mongo(self, chunk_size=25000, max_workers=4):  # ← NUEVO parámetro max_workers
        """
        Versión paralela: procesa múltiples chunks simultáneamente con hilos.
        
        Parámetros:
          chunk_size  : cantidad de coordenadas por bloque (igual que antes: 25000)
          max_workers : número de hilos en paralelo (recomendado: 4)
        """
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        col = db[COLLECTION_NAME]

        # 1. Obtener coordenadas únicas sin datos de suelo
        query_missing = {"processed_soil": {"$ne": True}}
        print("Buscando ubicaciones únicas sin datos de suelo...")
        pipeline = [
            {"$match": query_missing},
            {"$group": {"_id": "$location.coordinates"}}
        ]
        unique_locations = list(col.aggregate(pipeline))

        if not unique_locations:
            print("Todos los puntos ya tienen datos de suelo.")
            return

        total_puntos = len(unique_locations)
        print(f"Total de ubicaciones únicas a procesar: {total_puntos}")

        # Cargar datos espaciales de SSURGO (una sola vez, compartido entre hilos)
        map_polys, soil_data = self._get_soil_data_layer()

        # Dividir en chunks
        chunks = [
            unique_locations[i: i + chunk_size]
            for i in range(0, total_puntos, chunk_size)
        ]
        total_chunks = len(chunks)
        print(f"Total de chunks: {total_chunks} | Hilos paralelos: {max_workers}\n")

        # Procesamiento paralelo con ThreadPoolExecutor
        # Cada hilo procesa un chunk de forma independiente
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            # Lanzar todos los chunks al pool de hilos
            futures = {
                executor.submit(self._process_chunk, idx, chunk, map_polys, soil_data): idx
                for idx, chunk in enumerate(chunks)
            }

            # Recoger resultados conforme terminan (el que termina primero se guarda primero)
            completed = 0
            for future in tqdm(as_completed(futures), total=total_chunks, desc="Chunks completados"):
                chunk_num, updates = future.result()
                completed += 1

                # Escribir en MongoDB desde el hilo principal (evita conflictos de conexión)
                if updates:
                    col.bulk_write(updates)
                    print(f"  [MongoDB] Chunk {chunk_num} guardado ({len(updates)} docs). "
                          f"Progreso: {completed}/{total_chunks}")

        print("\nEnriquecimiento edáfico finalizado.")
        client.close()


