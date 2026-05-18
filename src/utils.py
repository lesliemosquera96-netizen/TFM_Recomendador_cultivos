# Cultivos de alto valor en California (Prioridad)
TARGET_CROPS = {
    75: "Almonds", 69: "Grapes", 204: "Pistachios", 76: "Walnuts", 
    54: "Tomatoes", 3: "Rice", 2: "Cotton", 227: "Lettuce", 
    221: "Strawberries", 212: "Oranges", 
    231: "Dbl Crop Lettuce/Cantaloupe", 232: "Dbl Crop Lettuce/Cotton", 
    233: "Dbl Crop Lettuce/Barley", 238: "Dbl Crop WinWht/Cotton", 
    239: "Dbl Crop Soybeans/Cotton"
}

# IDs a excluir (No agrícolas / Ruido)
EXCLUDED_IDS = {0, 111, 121, 122, 123, 124, 131, 141, 142, 143, 152, 176, 190, 195}

## Diccionario de todos los cultivos

CROP_DICT = {
    1:  "Corn",           2:  "Cotton",         3:  "Rice",
    4:  "Sorghum",        5:  "Soybeans",        6:  "Sunflower",
    10: "Peanuts",        11: "Tobacco",         12: "Sweet Corn",
    13: "Pop or Orn Corn",14: "Mint",
    21: "Barley",         22: "Durum Wheat",     23: "Spring Wheat",
    24: "Winter Wheat",   25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",            28: "Oats",            29: "Millet",
    30: "Speltz",         31: "Canola",          32: "Flaxseed",
    33: "Safflower",      34: "Rape Seed",       35: "Mustard",
    36: "Alfalfa",        37: "Other Hay/Non Alfalfa", 38: "Camelina",
    39: "Buckwheat",      41: "Sugarbeets",     42: "Dry Beans",        
    43: "Potatoes",       44: "Other Crops",    45: "Sugarcane",    
    46: "Sweet Potatoes", 47: "Misc Vegs & Fruits",
    48: "Watermelons",    49: "Onions",           50: "Cucumbers",
    51: "Chick Peas",     52: "Lentils",          53: "Peas",
    54: "Tomatoes",       55: "Caneberries",      56: "Hops",
    57: "Herbs",          58: "Clover/Wildflowers", 59: "Sod/ Grass Seed",
    60: "Switchgrass",    61: "Fallow/ Idle Cropland", 62: "Pasture/Grass",
    63: "Forest",         64: "Shrubland",        65:"Barren",
    66: "Cherries",       67: "Peaches",          68: "Apples",
    69: "Grapes",         70: "Christmas Trees",  71: "Other Tree Crops",
    72: "Citrus",         74: "Pecans",           75: "Almonds",
    76: "Walnuts",        77: "Pears",
    # ── Coberturas no agrícolas ───────────────────────────────────────────────
    111: "Open Water",    121: "Developed/Open Space", 122: "Developed/Low Intensity",
    123: "Developed/Med Intensity", 124: "Developed/High Intensity", 131: "Barren",
    141: "Deciduous Forest", 142: "Evergreen Forest", 143: "Mixed Forest",
    152: "Shrubland",        176: "Grassland/Pasture",190: "Woody Wetlands",
    195: "Herbaceous Wetlands",
    #-
    204: "Pistachios",    205: "Triticale",       206: "Carrots",
    207: "Asparagus",     208: "Garlic",          209: "Cantaloupes",
    210: "Prunes",        211: "Olives",          212: "Oranges",
    213: "Honeydew Melons",214: "Broccoli",       215: "Avocados",
    216: "Peppers",       217: "Pomegranates",    218: "Nectarines",
    219: "Greens",        220: "Plums",           221: "Strawberries",
    222: "Squash",        223: "Apricots",        224: "Vetch",
    225: "Dbl Crop WinWht/Corn", 226: "Dbl Crop Oats/Corn",
    227: "Lettuce",       228: "Dbl Crop Triticale/Corn",
    229: "Pumpkins",      230: "Dbl Crop Lettuce/Durum Wht",
    231:"Dbl Crop Lettuce/Cantaloupe", 232: "Dbl Crop Lettuce/Cotton",
    233: "Dbl Crop Lettuce/Barley",    234: "Dbl Crop Durum Wht/Sorghum",
    235: "Dbl Crop Barley/Sorghum",    236: "Dbl Crop WinWht/Sorghum",
    237: "Dbl Crop Barley/Corn",       238: "Dbl Crop WinWht/Cotton",
    239: "Dbl Crop Soybeans/Cotton",   240: "Dbl Crop Soybeans/Oats",
    241: "Dbl Crop Corn/Soybeans",     242: "Blueberrues",     243: "Cabbage",        244: "Cauliflower",
    245: "Celery",     246: "Radishes",  247: "Turnips",
    248: "Eggplants",   249: "Gourds",          250: "Cranberries",
    254: "Dbl Crop Barley/Soybeans",
}

# Diccionario de metadatos para variables de suelos y clima
# Útil para: Leyendas de gráficos, títulos de mapas y documentación del TFM

SOIL_VARIABLES_DICT = {
    "awc_r": {
        "name": "Capacidad de Agua Disponible",
        "unit": "cm/cm",
        "description": "Cantidad de agua que el suelo puede almacenar para las plantas."
    },
    "claytotal_r": {
        "name": "Contenido de Arcilla",
        "unit": "%",
        "description": "Porcentaje de partículas de arcilla (influye en la retención de nutrientes)."
    },
    "sandtotal_r": {
        "name": "Contenido de Arena",
        "unit": "%",
        "description": "Porcentaje de partículas de arena (influye en el drenaje)."
    },
    "silttotal_r": {
        "name": "Contenido de Limo",
        "unit": "%",
        "description": "Porcentaje de partículas de limo (fertilidad y textura)."
    },
    "om_r": {
        "name": "Materia Orgánica",
        "unit": "%",
        "description": "Indicador clave de la salud y fertilidad del suelo."
    },
    "ph1to1h2o_r": {
        "name": "pH del Suelo",
        "unit": "pH",
        "description": "Nivel de acidez o alcalinidad (disponibilidad de nutrientes)."
    },
    "cec7_r": {
        "name": "Capacidad de Intercambio Catiónico",
        "unit": "meq/100g",
        "description": "Capacidad del suelo para retener e intercambiar cationes (nutrientes)."
    },
    "dbthirdbar_r": {
        "name": "Densidad Aparente",
        "unit": "g/cm³",
        "description": "Masa por unidad de volumen; indica compactación del suelo."
    },
    "ksat_r": {
        "name": "Conductividad Hidráulica Saturada",
        "unit": "μm/s",
        "description": "Velocidad del movimiento del agua en suelo saturado."
    },
    "ec_r": {
        "name": "Conductividad Eléctrica",
        "unit": "dS/m",
        "description": "Medida de la salinidad del suelo."
    },
    "profundidad_efectiva_cm": {
        "name": "Profundidad Efectiva",
        "unit": "cm",
        "description": "Profundidad máxima que pueden alcanzar las raíces sin restricciones."
    }
}
HIST_CLIM_VARS = {
    "ppt":  {
        "name":"Precipitacion Anual",   
        "unit":"mm",
        "description": "Suma total de lluvia acumulada en el año."
    },
    "tmean": {
        "name": "Temperatura Media",
        "unit": "°C",
        "description": "Promedio anual de la temperatura media diaria"
    },
    "tmax": { 
        "name" :"Temperatura Maxima Media",
        "unit": "°C",
        "description": "Promedio anual de la temperatura máxima"
    },
    "tmin":  {
        "name":"Temperatura Minima Media", 
        "unit": "°C",
        "description": "Promedio anual de la temperatura mínima"
    },
    "vpdmax": {
        "name":"VPD Maximo",
        "unit":"hPa",
        "description": "déficit máximo de presión de vapor"
    },
    "vpdmin": {
        "name":"VPD Minimo", 
        "unit":"hPa",
        "description": "déficit mínimo de presión de vapor"
    }
}


FUT_CLIM_VARS = {
    "pr": {
        "name": "Precipitación Anual Total",
        "unit": "mm",
        "description": "Suma total de lluvia acumulada en el año."
    },
    "tmean": {
        "name": "Temperatura Media Anual",
        "unit": "°C",
        "description": "Promedio diario de la temperatura del aire."
    },
    "tmax": {
        "name": "Temperatura Máxima Media",
        "unit": "°C",
        "description": "Promedio de las temperaturas máximas diarias."
    },
    "tmin": {
        "name": "Temperatura Mínima Media",
        "unit": "°C",
        "description": "Promedio de las temperaturas mínimas diarias."
    },
    "hurs": {
        "name": "Humedad Relativa",
        "unit": "%",
        "description": "Promedio de humedad relativa en la superficie."
    }
}