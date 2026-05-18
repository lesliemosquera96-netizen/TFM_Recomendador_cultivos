from pymongo import MongoClient
from pprint import pprint
from src.config import MONGO_URI, DB_NAME, COLLECTION_NAME

client = MongoClient(MONGO_URI)
col = client[DB_NAME][COLLECTION_NAME]

# Un documento por año, ordenados
years = sorted(col.distinct("year"))
print(f"Anos encontrados en la coleccion: {years}\n")

for year in years:
    doc = col.find_one({"year": year})
    if doc:
        doc.pop("_id", None)  # quitar _id para que sea mas legible
        print(f"AÑO: {year}")
        pprint(doc)
        print()


client.close()