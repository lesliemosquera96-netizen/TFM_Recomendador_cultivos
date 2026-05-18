import os
import requests
import zipfile
import time
import rasterio
from rasterio.warp import transform as reproject_coords
from pymongo import MongoClient, UpdateOne
from tqdm import tqdm # Para ver una barra de progreso
from src.config import MONGO_URI, DB_NAME, COLLECTION_NAME, BASE_PATH

class ClimateExtractor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.climate_dir = os.path.join(self.base_path, "CLIMATE_PRISM")
        # Las 5 variables que definimos
        self.elements = ['tmin', 'tmax', 'tmean', 'ppt', 'vpdmax', 'vpdmin']
        
        if not os.path.exists(self.climate_dir):
            os.makedirs(self.climate_dir)

    def download_range(self, start_year, end_year):
        """Descarga automática para un rango de años"""
        for year in range(start_year, end_year + 1):
            self.download_annual_data(year)

    def download_annual_data(self, year):
        print(f"\n--- Procesando Año {year} ---")
        
        for element in self.elements:
            # 1. RESTRICCIÓN: Verificamos si la carpeta ya existe y tiene contenido
            target_folder = os.path.join(self.climate_dir, f"{year}_{element}")
            
            if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
                print(f"{element} {year}: Ya existe localmente. Saltando descarga.")
                continue

            # 2. Si no existe, procedemos con la descarga
            url = f"https://services.nacse.org/prism/data/get/us/4km/{element}/{year}?format=bil"
            
            try:
                print(f"Descargando {element} {year} desde PRISM...")
                response = requests.get(url, stream=True, timeout=60)
                
                if response.status_code == 200:
                    zip_path = os.path.join(self.climate_dir, f"temp_{year}_{element}.zip")
                    
                    with open(zip_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Crear carpeta y descomprimir
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                        
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(target_folder)
                    
                    os.remove(zip_path) # Borrar el zip para ahorrar espacio
                    print(f" Guardado con éxito en: {target_folder}")
                    
                    # Pausa de seguridad para no saturar el servidor de PRISM
                    time.sleep(2) 
                else:
                    print(f"Error {response.status_code} para {element} {year}. (Quizás el año aún no está disponible)")

            except Exception as e:
                print(f"Falló {element} {year}: {e}")

    def get_raster_path(self, year, element):
        """Método útil para el futuro: devuelve la ruta al archivo .bil"""
        folder = os.path.join(self.climate_dir, f"{year}_{element}")
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".bil"):
                    return os.path.join(folder, file)
        return None
    
    def get_value_at_point(self, lat, lon, year, element):
        """Extrae el valor del raster para una coordenada específica"""
        path = self.get_raster_path(year, element)
        if not path: return None
        
        try:
            with rasterio.open(path) as src:
                # Reproyectar de WGS84 a la proyección del mapa (NAD83)
                xs, ys = reproject_coords('EPSG:4326', src.crs, [lon], [lat])
                for val in src.sample([(xs[0], ys[0])]):
                    v = float(val[0])
                    # PRISM usa -9999 para valores nulos
                    return v if v > -9000 else None
        except:
            return None

    def update_climate_in_mongo(self):
        """Lee de Mongo y actualiza cada punto con su clima"""
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        col = db[COLLECTION_NAME]

        # Buscamos puntos que NO tengan 'tmin' (indicador de que falta el clima)
        query = {"tmin": {"$exists": False}}
        total = col.count_documents(query)
        
        if total == 0:
            print("Todos los puntos tienen clima.")
            return

        years = col.distinct("year", query)
        for year in years:
            print(f"\n Procesando clima año {year}...")
            cursor = col.find({"year": year, "tmin": {"$exists": False}})
            
            updates = []
            # Usamos el total del año para la barra de progreso
            total_year = col.count_documents({"year": year, "tmin": {"$exists": False}})
            
            for doc in tqdm(cursor, total=total_year):
                lon, lat = doc['location']['coordinates']
                
                # Creamos un diccionario con las nuevas variables
                # Esto es lo que pedías: usar el vector de elementos
                payload = {}
                for element in self.elements:
                    payload[element] = self.get_value_at_point(lat, lon, year, element)
                
                payload["processed_climate"] = True

                # Usamos UpdateOne para procesar por lotes (más rápido)
                updates.append(UpdateOne({"_id": doc["_id"]}, {"$set": payload}))

                # Enviamos a Mongo cada 1000 registros
                if len(updates) >= 1000:
                    col.bulk_write(updates)
                    updates = []

            if updates:
                col.bulk_write(updates)
