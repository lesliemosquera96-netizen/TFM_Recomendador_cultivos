import os
import requests
from zipfile import ZipFile
import rasterio
import numpy as np
from rasterio.warp import transform as reproject_coords
from src.utils import TARGET_CROPS, EXCLUDED_IDS, CROP_DICT

##Descarga los datos de USDA
class CDLExtractor:
    def __init__(self, base_path, year):
        self.base_path = base_path
        self.year = year
        self.zip_url = f"https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/{year}_30m_cdls.zip"
        self.folder_path = os.path.join(self.base_path, f"{year}_30m_cdls")
        self.tif_path = os.path.join(self.folder_path, f"{year}_30m_cdls.tif")

    def extract(self):
        """Paso 1: Extract - Baja el archivo y lo descomprime"""
        if not os.path.exists(self.tif_path): #evita descargar el archivo si ya existe
            print(f"Descargando CDL {self.year}...")
            r = requests.get(self.zip_url)
            zip_file_path = self.tif_path.replace(".tif", ".zip")
            
            os.makedirs(self.folder_path, exist_ok=True)
            with open(zip_file_path, 'wb') as f:
                f.write(r.content)
            
            print(f"Descomprimiendo...")
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.folder_path)
            os.remove(zip_file_path) # Limpiamos el zip para ahorrar espacio
        return self.tif_path
    
    def selection(self, bounds, n_points=55000):
        print(f"Filtrando California {self.year} (Prioridad: {len(TARGET_CROPS)} cultivos)")
        
        priority_samples = []
        secondary_samples = []
        
        with rasterio.open(self.tif_path) as src:
            # Metadata y Sistema de Referencia del Mapa (CRS)
            self.crs = src.crs
            
            attempts = 0
            max_attempts = 15 
            
            while (len(priority_samples) + len(secondary_samples)) < n_points and attempts < max_attempts:
                # 1. Generamos coordenadas en Lat/Lon (WGS84)
                raw_lons = np.random.uniform(bounds['lon'][0], bounds['lon'][1], size=n_points * 3)
                raw_lats = np.random.uniform(bounds['lat'][0], bounds['lat'][1], size=n_points * 3)
                
                # 2. ¡CRUCIAL!: Traducir Lat/Lon al sistema del mapa (Reproyección)
                # Esto convierte grados a los metros que entiende el archivo .tif
                proj_lons, proj_lats = reproject_coords('EPSG:4326', self.crs, raw_lons, raw_lats)
                
                # Creamos las parejas proyectadas para el muestreo
                coords_to_sample = list(zip(proj_lons, proj_lats))
                
                # 3. Muestrear usando las coordenadas proyectadas
                # Pero guardamos las coordenadas ORIGINALES (raw_lon/lat) en el JSON
                for i, val in enumerate(src.sample(coords_to_sample)):
                    crop_id = int(val[0])
                    lon_original = raw_lons[i]
                    lat_original = raw_lats[i]
                    
                    if crop_id in TARGET_CROPS:
                        priority_samples.append(self._create_doc(crop_id, lon_original, lat_original, is_priority=True))
                    elif crop_id not in EXCLUDED_IDS:
                        secondary_samples.append(self._create_doc(crop_id, lon_original, lat_original, is_priority=False))
                    
                    if (len(priority_samples) + len(secondary_samples)) >= n_points:
                        break
                
                attempts += 1
                print(f"   Intento {attempts}: Llevamos {len(priority_samples) + len(secondary_samples)} puntos...")

        # Selección final
        final_data = priority_samples + secondary_samples[:(n_points - len(priority_samples))]
        print(f"Resultado Final: {len(priority_samples)} prioridad + {len(final_data)-len(priority_samples)} secundarios.")
        return final_data
    
    def _create_doc(self, crop_id, lon, lat, is_priority):
        """Función auxiliar para formatear el documento de Mongo"""
        name = TARGET_CROPS.get(crop_id) or CROP_DICT.get(crop_id, "Other Agriculture")
        return {
            "year": self.year,
            "crop_id": crop_id,
            "crop_name": name,
            "is_target": is_priority,
            "location": {"type": "Point", "coordinates": [lon, lat]},
            "processed": False
        }

    #def load(self, data, collection):
        #if data:
           # collection.insert_many(data)
            #print(f"{len(data)} documentos cargados con éxito.")
    
    def check_if_loaded(self, collection):
        """Verifica si ya existen datos para el año actual en MongoDB."""
        count = collection.count_documents({"year": self.year})
        return count > 0

    def load(self, data, collection):
        """Carga los datos solo si no existen para evitar duplicados."""
        if not data:
            print(f"No hay datos generados para el año {self.year}.")
            return

        print(f"Verificando estado de la base de datos para el año {self.year}...")
        
        if self.check_if_loaded(collection):
            print(f"El año {self.year} ya tiene datos en MongoDB. Saltando carga para evitar duplicados.")
        else:
            print(f"Cargando {len(data)} documentos nuevos para el año {self.year}...")
            # Insertamos por lotes para evitar errores de timeout si la conexión es inestable
            batch_size = 5000
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                collection.insert_many(batch)
            print(f"¡Carga de {self.year} completada con éxito")