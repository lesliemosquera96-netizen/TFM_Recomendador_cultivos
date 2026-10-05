"""
================================================================================
 APP — RECOMENDADOR DE CULTIVOS PARA CALIFORNIA
================================================================================
Aplicación Streamlit para agricultores. Predice el Índice de Idoneidad Agrícola
(IAI) de 12 cultivos y muestra cómo cambia con el clima futuro (2030 y 2040).

Pestañas:
  1. Mi lote             — ubicación + cultivos aptos + gráficos + evolución clima
  2. Mapas por cultivo   — mapas del IAI filtrables por cultivo y año
  3. Introduce tus datos — el productor ajusta clima actual y suelo de su parcela;
                           el clima futuro se toma de la base (NASA CMIP6)
  4. Acerca de           — metodología, autoría y fuentes del proyecto

Cambios principales respecto a la versión anterior:
  · El usuario ya no introduce clima futuro: se calcula con el método delta a
    partir de las proyecciones de la base para su ubicación.
  · Solo se dan resultados para coordenadas dentro de California.
  · Guía paso a paso, ayudas en cada campo y lectura sencilla de resultados.
  · Nuevo diseño visual.
================================================================================
"""

import os
import base64
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px

from predictor import (
    PredictorCultivos, VARS_CLUSTER, FINAL_CROPS,
    CROP_DICT_EN, CROP_DICT_ES, ETIQUETAS_CLUSTER,
)

# ─── Rutas ───
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RUTA_BASE = os.path.join(PROJECT_ROOT, "Data", "base_california_app.parquet")
RUTA_IAI_MAPAS = os.path.join(PROJECT_ROOT, "Data", "iai_mapas_california.parquet")
LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

# ─── Paleta ───
COLOR_FONDO = "#F6F4EC"      # crema suave
COLOR_PRIMARIO = "#0B5D45"   # verde bosque
COLOR_ACENTO = "#2E9E6B"     # verde hoja
COLOR_LIMA = "#9BC53D"       # verde brote
COLOR_SOL = "#F2B33D"        # amarillo trigo
COLOR_TIERRA = "#D9653B"     # terracota
COLOR_TEXTO = "#2F3E35"
COLOR_SUAVE = "#6B7F72"
COLOR_BORDE = "#E3E8DF"
COLOR_MOD = "#2E9E6B"        # escenario moderado (SSP2-4.5)
COLOR_SEV = "#D9653B"        # escenario severo (SSP5-8.5)
ESCALA_IAI = [[0.0, "#D9653B"], [0.5, "#F2C66D"], [1.0, "#0B5D45"]]

# ─── Categorías de idoneidad (ajustar si el TFM usa otros umbrales) ───
CATEGORIAS_IAI = [
    (0.70, "Muy apto", "#0B5D45"),
    (0.50, "Apto", "#2E9E6B"),
    (0.30, "Poco apto", "#E0A129"),
    (0.00, "No recomendado", "#D9653B"),
]

# Ancho completo: las versiones recientes de Streamlit usan width="stretch"
_VERSION_ST = tuple(int(p) for p in st.__version__.split(".")[:2])
ANCHO = {"width": "stretch"} if _VERSION_ST >= (1, 50) else {"use_container_width": True}

# Distancia máxima (grados) al punto con datos antes de avisar (~15 km)
DIST_AVISO = 0.15


# ══════════════════════════════════════════════════════════════════
#  LÍMITE DE CALIFORNIA
# ══════════════════════════════════════════════════════════════════
# Polígono simplificado (lon, lat). La costa incluye un pequeño margen hacia
# el mar para no excluir parcelas costeras; los puntos en el océano se
# descartan después porque no tienen datos cercanos en la base.
CALIFORNIA_POLY = np.array([
    (-124.45, 42.00), (-120.00, 42.00), (-120.00, 39.00), (-114.63, 35.00),
    (-114.57, 34.83), (-114.43, 34.60), (-114.35, 34.45), (-114.13, 34.27),
    (-114.43, 34.08), (-114.53, 33.93), (-114.50, 33.60), (-114.52, 33.03),
    (-114.47, 32.84), (-114.72, 32.72), (-117.12, 32.53), (-117.30, 32.53),
    (-117.40, 33.10), (-118.00, 33.55), (-118.55, 33.70), (-118.60, 34.00),
    (-119.30, 34.10), (-120.10, 34.35), (-120.75, 34.45), (-120.80, 35.20),
    (-121.40, 35.60), (-122.05, 36.30), (-122.10, 36.95), (-122.55, 37.20),
    (-122.65, 37.80), (-123.10, 37.95), (-123.85, 38.90), (-123.95, 39.80),
    (-124.55, 40.40), (-124.25, 41.00), (-124.35, 41.80), (-124.45, 42.00),
])


def en_california(lon, lat):
    """Devuelve True si el punto (lon, lat) cae dentro de California.

    Se usa el algoritmo de ray casting sobre el polígono simplificado,
    así no hace falta instalar shapely ni geopandas.
    """
    if lon is None or lat is None:
        return False
    x, y = float(lon), float(lat)
    # Descarte rápido con la caja envolvente
    if not (-124.6 <= x <= -114.1 and 32.5 <= y <= 42.0):
        return False
    xs, ys = CALIFORNIA_POLY[:, 0], CALIFORNIA_POLY[:, 1]
    dentro = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > y) != (ys[j] > y):
            x_cruce = (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]
            if x < x_cruce:
                dentro = not dentro
        j = i
    return dentro


# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN Y ESTILOS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Recomendador de cultivos · California",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {{
        font-family: 'Nunito', 'Source Sans Pro', sans-serif !important;
    }}
    .stApp {{
        background-color: {COLOR_FONDO};
        background-image:
            radial-gradient(circle at 0% 0%, rgba(155,197,61,0.10) 0, transparent 35%),
            radial-gradient(circle at 100% 100%, rgba(242,179,61,0.10) 0, transparent 35%);
        background-attachment: fixed;
    }}
    .block-container {{ padding-top: 2.2rem; max-width: 1250px; }}
    section[data-testid="stSidebar"] {{ display: none; }}
    div[data-testid="collapsedControl"] {{ display: none; }}
    p, li, label {{ color: {COLOR_TEXTO}; }}

    /* ── Hero ── */
    .hero {{
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, {COLOR_PRIMARIO} 0%, #1C7F57 55%, {COLOR_LIMA} 120%);
        border-radius: 24px; padding: 30px 34px 26px 34px; margin-bottom: 22px;
        color: white; box-shadow: 0 12px 30px rgba(11,93,69,0.22);
    }}
    .hero::before, .hero::after {{
        content: ""; position: absolute; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }}
    .hero::before {{ width: 320px; height: 320px; right: -80px; top: -140px; }}
    .hero::after  {{ width: 200px; height: 200px; right: 140px; bottom: -120px;
                     background: rgba(242,179,61,0.18); }}
    .hero-fila {{ display: flex; align-items: center; gap: 18px; position: relative; z-index: 1; }}
    .hero-titulo {{ font-size: 34px; font-weight: 800; line-height: 1.15; color: white; }}
    .hero-desc {{ font-size: 16px; opacity: 0.92; margin-top: 6px; max-width: 640px; color: white; }}
    .hero-pasos {{ display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap;
                   position: relative; z-index: 1; }}
    .hero-paso {{
        background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22);
        backdrop-filter: blur(4px); border-radius: 14px; padding: 10px 14px;
        font-size: 14px; color: white; display: flex; align-items: center; gap: 10px;
    }}
    .hero-num {{
        background: {COLOR_SOL}; color: {COLOR_PRIMARIO}; font-weight: 800;
        width: 26px; height: 26px; border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center; font-size: 14px;
    }}

    /* ── Pestañas tipo píldora (selectores válidos en versiones nuevas y antiguas) ── */
    .stTabs [role="tablist"] {{
        gap: 6px; background: white; padding: 6px; border-radius: 16px;
        border: 1px solid {COLOR_BORDE}; box-shadow: 0 2px 10px rgba(11,93,69,0.05);
        width: fit-content; max-width: 100%;
    }}
    .stTabs [role="tab"] {{
        border-radius: 12px; padding: 10px 22px; height: auto; cursor: pointer;
        background: transparent; transition: background .15s ease;
    }}
    .stTabs [role="tab"] p {{ color: {COLOR_PRIMARIO} !important; font-weight: 700; font-size: 15px; }}
    .stTabs [role="tab"]:hover {{ background: #EEF5EC; }}
    .stTabs [role="tab"][aria-selected="true"] {{
        background: linear-gradient(120deg, {COLOR_PRIMARIO}, {COLOR_ACENTO});
        box-shadow: 0 4px 12px rgba(11,93,69,0.25);
    }}
    .stTabs [role="tab"][aria-selected="true"] p {{ color: white !important; }}
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"],
    .stTabs .react-aria-SelectionIndicator {{ display: none !important; }}
    .stTabs [role="tabpanel"] {{ padding-top: 18px; }}

    /* ── Pasos guiados ── */
    .paso {{ display: flex; align-items: flex-start; gap: 12px; margin: 18px 0 10px 0; }}
    .paso-num {{
        flex-shrink: 0; width: 34px; height: 34px; border-radius: 50%;
        background: linear-gradient(135deg, {COLOR_PRIMARIO}, {COLOR_ACENTO});
        color: white; font-weight: 800; font-size: 16px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px rgba(11,93,69,0.25);
    }}
    .paso-titulo {{ font-size: 19px; font-weight: 800; color: {COLOR_PRIMARIO}; line-height: 1.3; }}
    .paso-texto {{ font-size: 14px; color: {COLOR_SUAVE}; margin-top: 2px; }}

    /* ── Cajas de ayuda ── */
    .consejo {{
        background: #FFF8E6; border-left: 5px solid {COLOR_SOL}; border-radius: 12px;
        padding: 12px 16px; font-size: 14px; color: #5B4A1C; margin: 8px 0 14px 0;
    }}
    .consejo b {{ color: #4A3A10; }}
    .info {{
        background: #EAF5EE; border-left: 5px solid {COLOR_ACENTO}; border-radius: 12px;
        padding: 12px 16px; font-size: 14px; color: #1F4D3A; margin: 8px 0 14px 0;
    }}
    .fuera {{
        background: #FDEEE7; border: 1px solid #F3C3AE; border-left: 6px solid {COLOR_TIERRA};
        border-radius: 14px; padding: 18px 20px; color: #6B2F17; margin: 6px 0 14px 0;
    }}
    .fuera-titulo {{ font-weight: 800; font-size: 18px; margin-bottom: 6px; color: #8A3615; }}

    /* ── Tarjetas ── */
    .bloque {{
        background: white; border-radius: 18px; padding: 18px 20px;
        border: 1px solid {COLOR_BORDE}; margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(11,93,69,0.06);
    }}
    .etiqueta-mini {{ font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
                      text-transform: uppercase; color: {COLOR_SUAVE}; }}
    .zona-nombre {{ font-size: 22px; font-weight: 800; color: {COLOR_PRIMARIO}; margin-top: 2px; }}
    .cond-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }}
    .cond {{ background: #F4F8F2; border-radius: 12px; padding: 10px 12px; }}
    .cond-val {{ font-size: 18px; font-weight: 800; color: {COLOR_PRIMARIO}; }}
    .cond-lab {{ font-size: 12px; color: {COLOR_SUAVE}; }}

    .card-top {{
        background: white; border-radius: 20px; padding: 20px 16px 16px 16px;
        border: 1px solid {COLOR_BORDE}; text-align: center; position: relative;
        box-shadow: 0 6px 18px rgba(11,93,69,0.08); transition: transform .15s ease;
    }}
    .card-top:hover {{ transform: translateY(-3px); }}
    .card-rank {{
        position: absolute; top: -14px; left: 50%; transform: translateX(-50%);
        color: white; font-weight: 800; font-size: 13px; padding: 4px 14px; border-radius: 20px;
    }}
    .card-cultivo {{ font-size: 22px; font-weight: 800; color: {COLOR_PRIMARIO}; margin: 8px 0 2px 0; }}
    .card-valor {{ font-size: 36px; font-weight: 800; line-height: 1.1; }}
    .badge {{ display: inline-block; padding: 3px 12px; border-radius: 20px; font-size: 12px;
              font-weight: 800; color: white; margin-top: 4px; }}
    .barra-fondo {{ background: #EEF2EC; height: 8px; border-radius: 6px; margin-top: 12px; overflow: hidden; }}
    .barra {{ height: 8px; border-radius: 6px; }}

    .resiliente {{
        background: linear-gradient(120deg, #FFF6DD, #FDEBD2); border: 1px solid #F3D9A4;
        border-radius: 18px; padding: 16px 20px; margin: 18px 0 6px 0;
        display: flex; gap: 16px; align-items: center;
    }}
    .resiliente-ico {{
        flex-shrink: 0; width: 46px; height: 46px; border-radius: 14px; background: {COLOR_SOL};
        display: flex; align-items: center; justify-content: center;
    }}

    .seccion {{ color: {COLOR_PRIMARIO}; font-weight: 800; font-size: 20px; margin: 22px 0 2px 0; }}
    .seccion-sub {{ color: {COLOR_SUAVE}; font-size: 14px; margin-bottom: 8px; }}
    .leyenda {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 10px 0; }}
    h4, h5 {{ color: {COLOR_PRIMARIO} !important; font-weight: 800 !important; }}

    /* ── Widgets ── */
    div[data-testid="stMetric"] {{
        background: white; border: 1px solid {COLOR_BORDE}; border-radius: 16px;
        padding: 14px 18px; box-shadow: 0 4px 14px rgba(11,93,69,0.06);
    }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_PRIMARIO}; font-weight: 800; }}
    .stButton > button {{
        border-radius: 12px; font-weight: 700; padding: 0.55rem 1.2rem;
        border: 1px solid {COLOR_BORDE};
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(120deg, {COLOR_PRIMARIO}, {COLOR_ACENTO});
        border: none; box-shadow: 0 6px 16px rgba(11,93,69,0.25);
    }}
    .stButton > button[kind="primary"] p {{ color: white !important; font-weight: 800; }}
    .stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); }}
    div[data-baseweb="input"], div[data-baseweb="select"] > div {{ border-radius: 10px !important; }}
    div[data-testid="stExpander"] {{
        background: white; border-radius: 14px; border: 1px solid {COLOR_BORDE};
    }}
    iframe {{ border-radius: 16px; }}

    /* ── Pie ── */
    .pie {{ text-align: center; color: {COLOR_SUAVE}; font-size: 12px;
            margin-top: 30px; padding-top: 14px; border-top: 1px solid {COLOR_BORDE}; }}

    @media (max-width: 700px) {{
        .hero-titulo {{ font-size: 26px; }}
        .cond-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CARGA (en caché)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_motor():
    return PredictorCultivos()

@st.cache_data
def cargar_base():
    return pl.read_parquet(RUTA_BASE).filter(pl.col("datos_completos") == True)

@st.cache_data
def cargar_iai_mapas():
    return pl.read_parquet(RUTA_IAI_MAPAS)

@st.cache_resource
def construir_kdtree(_base):
    return cKDTree(_base.select(["lon", "lat"]).to_numpy())

@st.cache_data(show_spinner=False, ttl=3600)
def geocodificar(direccion):
    """Busca una dirección en California. Devuelve (lon, lat) o None."""
    from geopy.geocoders import Nominatim
    geo = Nominatim(user_agent="recomendador_cultivos_tfm")
    loc = geo.geocode(f"{direccion}, California, USA", timeout=10, country_codes="us")
    return (loc.longitude, loc.latitude) if loc else None

motor = cargar_motor()
base = cargar_base()
iai_mapas = cargar_iai_mapas()
tree = construir_kdtree(base)


# ══════════════════════════════════════════════════════════════════
#  FUNCIONES AUXILIARES
# ══════════════════════════════════════════════════════════════════
def _limpiar(html):
    """Quita sangrías y líneas vacías del HTML.

    Markdown interpreta las líneas con 4 espacios como bloque de código y las
    líneas vacías cortan el bloque HTML, así que se eliminan ambas.
    """
    return "\n".join(l.strip() for l in html.splitlines() if l.strip())


def logo_b64():
    """Devuelve el logo en base64 si existe junto a app.py; si no, None."""
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def punto_mas_cercano(lon, lat):
    dist, idx = tree.query([lon, lat])
    return base.row(int(idx), named=True), dist


def categoria_iai(v):
    """Traduce el IAI a una etiqueta sencilla y su color."""
    for umbral, nombre, color in CATEGORIAS_IAI:
        if v >= umbral:
            return nombre, color
    return CATEGORIAS_IAI[-1][1], CATEGORIAS_IAI[-1][2]


def color_iai(v):
    """IAI [0,1] -> color RGB (terracota -> amarillo -> verde)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return [200, 200, 200]
    v = min(max(float(v), 0.0), 1.0)
    bajo, medio, alto = (217, 101, 59), (242, 198, 109), (11, 93, 69)
    if v < 0.5:
        a, b, t = bajo, medio, v / 0.5
    else:
        a, b, t = medio, alto, (v - 0.5) / 0.5
    return [int(a[k] + (b[k] - a[k]) * t) for k in range(3)]


def clima_futuro_desde_base(fila, tmax, tmin, ppt):
    """Calcula el clima futuro de la parcela con el método delta.

    El productor solo conoce su clima actual. Por eso se toma el cambio que
    proyecta la base (NASA NEX-GDDP-CMIP6) en el punto más cercano y se aplica
    sobre los valores introducidos: las temperaturas se desplazan en grados
    (diferencia) y la lluvia se escala en proporción (cociente).
    """
    futuro = {}
    for ssp in ("245", "585"):
        for anio in ("2030", "2040"):
            d_tmax = fila[f"tmax_{ssp}_{anio}"] - fila["tmax"]
            d_tmin = fila[f"tmin_{ssp}_{anio}"] - fila["tmin"]
            ratio = fila[f"pr_{ssp}_{anio}"] / fila["ppt"] if fila["ppt"] else 1.0
            futuro[f"tmax_{ssp}_{anio}"] = tmax + d_tmax
            futuro[f"tmin_{ssp}_{anio}"] = tmin + d_tmin
            futuro[f"pr_{ssp}_{anio}"] = ppt * ratio
    return futuro


def _estilo(fig, alto=320):
    """Aplica un estilo limpio y consistente a las figuras de Plotly."""
    fig.update_layout(
        height=alto,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito, sans-serif", size=13, color=COLOR_TEXTO),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title=""),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Nunito"),
    )
    fig.update_xaxes(showgrid=False, linecolor=COLOR_BORDE)
    fig.update_yaxes(showgrid=True, gridcolor="#E9EEE6", zeroline=False)
    return fig


_contador_graficos = {"n": 0}


def _grafico(fig):
    """Dibuja una figura con una clave única (los mismos gráficos pueden
    aparecer en 'Mi lote' y en 'Introduce tus datos')."""
    _contador_graficos["n"] += 1
    st.plotly_chart(fig, key=f"grafico_{_contador_graficos['n']}",
                    config={"displayModeBar": False}, **ANCHO)


# ─── Piezas de interfaz reutilizables ───
def paso(numero, titulo, texto=""):
    st.markdown(_limpiar(f"""
    <div class='paso'>
      <div class='paso-num'>{numero}</div>
      <div><div class='paso-titulo'>{titulo}</div>
           {'<div class="paso-texto">' + texto + '</div>' if texto else ''}</div>
    </div>"""), unsafe_allow_html=True)


def consejo(texto):
    st.markdown(f"<div class='consejo'><b>Consejo:</b> {texto}</div>", unsafe_allow_html=True)


def info(texto):
    st.markdown(f"<div class='info'>{texto}</div>", unsafe_allow_html=True)


def aviso_fuera_california(lon, lat):
    st.markdown(_limpiar(f"""
    <div class='fuera'>
      <div class='fuera-titulo'>Esta ubicación está fuera de California</div>
      El punto seleccionado ({lon:.4f}, {lat:.4f}) no pertenece al estado de California.
      Los modelos solo se entrenaron con datos californianos, así que no podemos darte
      una recomendación fiable para este lugar.<br><br>
      <b>Qué puedes hacer:</b> busca una ciudad de California (por ejemplo, <i>Fresno</i>
      o <i>Salinas</i>) o haz clic dentro de la línea discontinua del mapa.
    </div>"""), unsafe_allow_html=True)


def seccion(titulo, subtitulo=""):
    st.markdown(f"<div class='seccion'>{titulo}</div>"
                + (f"<div class='seccion-sub'>{subtitulo}</div>" if subtitulo else ""),
                unsafe_allow_html=True)


ICONO_HOJA = ("<svg width='26' height='26' viewBox='0 0 24 24' fill='none' stroke='#0B5D45' "
              "stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'>"
              "<path d='M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z'/>"
              "<path d='M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12'/></svg>")


# ══════════════════════════════════════════════════════════════════
#  CABECERA
# ══════════════════════════════════════════════════════════════════
_b64 = logo_b64()
_logo_img = (f"<img src='data:image/png;base64,{_b64}' "
             f"style='height:70px; background:white; border-radius:16px; padding:6px;'>"
             if _b64 else "")
st.markdown(_limpiar(f"""
<div class="hero">
  <div class="hero-fila">
    {_logo_img}
    <div>
      <div class="hero-titulo">Recomendador de cultivos</div>
      <div class="hero-desc">Descubre qué cultivos se adaptan mejor a tu parcela en
        California, hoy y ante el cambio climático.</div>
    </div>
  </div>
  <div class="hero-pasos">
    <div class="hero-paso"><span class="hero-num">1</span>Ubica tu parcela en el mapa</div>
    <div class="hero-paso"><span class="hero-num">2</span>Mira los cultivos más aptos</div>
    <div class="hero-paso"><span class="hero-num">3</span>Revisa cómo cambiarán en 2030 y 2040</div>
  </div>
</div>
"""), unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Mi lote", "Mapas por cultivo", "Introduce tus datos", "Acerca de"])


# ══════════════════════════════════════════════════════════════════
#  BLOQUES DE RESULTADO (reutilizables en Mi lote y Datos manuales)
# ══════════════════════════════════════════════════════════════════
def render_leyenda():
    chips = "".join(
        f"<span class='badge' style='background:{c};'>{n} · {'≥ ' + format(u, '.1f') if u > 0 else '< 0.3'}</span>"
        for u, n, c in CATEGORIAS_IAI)
    st.markdown(
        "<div class='info'><b>Cómo leer los resultados.</b> Cada cultivo recibe un "
        "<b>Índice de Idoneidad Agrícola (IAI)</b> entre 0 y 1. Cuanto más cerca de 1, "
        "mejor se adaptan el clima y el suelo de tu parcela a ese cultivo."
        f"<div class='leyenda' style='margin-top:8px;'>{chips}</div></div>",
        unsafe_allow_html=True)


def render_zona(resultado):
    st.markdown(_limpiar(f"""
    <div class='bloque'>
      <div class='etiqueta-mini'>Tu zona agroclimática</div>
      <div class='zona-nombre'>{resultado['etiqueta']}</div>
      <div style='font-size:13px; color:{COLOR_SUAVE}; margin-top:4px;'>
        California se dividió en cinco zonas con clima y suelo parecidos.
        Cada zona tiene su propio modelo de predicción.</div>
    </div>"""), unsafe_allow_html=True)


def render_condiciones(fila, titulo="Condiciones de tu parcela"):
    st.markdown(_limpiar(f"""
    <div class='bloque'>
      <div class='etiqueta-mini'>{titulo}</div>
      <div class='cond-grid'>
        <div class='cond'><div class='cond-val'>{fila['tmax']:.0f} °C</div><div class='cond-lab'>Temp. máxima</div></div>
        <div class='cond'><div class='cond-val'>{fila['tmin']:.0f} °C</div><div class='cond-lab'>Temp. mínima</div></div>
        <div class='cond'><div class='cond-val'>{fila['ppt']:.0f} mm</div><div class='cond-lab'>Lluvia anual</div></div>
        <div class='cond'><div class='cond-val'>{fila['ph1to1h2o_r']:.1f}</div><div class='cond-lab'>pH del suelo</div></div>
        <div class='cond'><div class='cond-val'>{fila['profundidad_efectiva_cm']:.0f} cm</div><div class='cond-lab'>Profundidad</div></div>
        <div class='cond'><div class='cond-val'>{fila['claytotal_r']:.0f} %</div><div class='cond-lab'>Arcilla</div></div>
      </div>
    </div>"""), unsafe_allow_html=True)


def render_top3(resultado):
    iai = resultado["iai"]
    ranking_hoy = motor.ranking(iai["2025"], top=3)
    seccion("Los 3 cultivos más aptos para tu parcela hoy",
            "Estos son los cultivos que mejor encajan con tus condiciones actuales.")
    colores_rank = [COLOR_PRIMARIO, COLOR_ACENTO, COLOR_LIMA]
    st.write("")
    cols = st.columns(3)
    for i, (nombre, val) in enumerate(ranking_hoy):
        cat, color_cat = categoria_iai(val)
        with cols[i]:
            st.markdown(_limpiar(f"""
            <div class='card-top' style='border-top: 5px solid {colores_rank[i]};'>
              <div class='card-rank' style='background:{colores_rank[i]};'>Opción {i+1}</div>
              <div class='card-cultivo'>{nombre}</div>
              <div class='card-valor' style='color:{color_cat};'>{val:.2f}</div>
              <span class='badge' style='background:{color_cat};'>{cat}</span>
              <div class='barra-fondo'><div class='barra'
                   style='width:{val*100:.0f}%; background:{color_cat};'></div></div>
            </div>
            """), unsafe_allow_html=True)
    return ranking_hoy


def render_resiliente(resultado):
    """Destaca el cultivo que mejor aguanta el escenario severo en 2040."""
    iai = resultado["iai"]
    mejor = max(FINAL_CROPS, key=lambda c: iai["2040-585"][c])
    v_hoy, v_fut = iai["2025"][mejor], iai["2040-585"][mejor]
    st.markdown(_limpiar(f"""
    <div class='resiliente'>
      <div class='resiliente-ico'>{ICONO_HOJA}</div>
      <div>
        <div class='etiqueta-mini' style='color:#8A6A1C;'>Pensando en el futuro</div>
        <div style='font-size:17px; font-weight:800; color:#5B4A1C;'>
          {CROP_DICT_ES[mejor]} es el cultivo que mejor resiste el escenario más exigente</div>
        <div style='font-size:14px; color:#6B5A2C;'>Idoneidad hoy: <b>{v_hoy:.2f}</b> ·
          en 2040 con cambio climático severo: <b>{v_fut:.2f}</b>.
          Es una buena opción si planificas inversiones a largo plazo (frutales, viñedos).</div>
      </div>
    </div>"""), unsafe_allow_html=True)


def render_barras(resultado):
    iai = resultado["iai"]
    df_barras = pd.DataFrame({
        "Cultivo": [CROP_DICT_ES[c] for c in FINAL_CROPS],
        "Idoneidad": [iai["2025"][c] for c in FINAL_CROPS],
    }).sort_values("Idoneidad", ascending=True)

    seccion("Idoneidad de los 12 cultivos hoy",
            "Pasa el ratón sobre una barra para ver el valor exacto.")
    fig = px.bar(
        df_barras, x="Idoneidad", y="Cultivo", orientation="h",
        color="Idoneidad", color_continuous_scale=ESCALA_IAI, range_color=[0, 1],
        text=df_barras["Idoneidad"].map(lambda v: f"{v:.2f}"),
    )
    fig.update_traces(textposition="outside", textfont_size=12, marker_line_width=0,
                      cliponaxis=False, hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    for x0, x1, color in [(0.7, 1.05, "rgba(11,93,69,0.05)"), (0.5, 0.7, "rgba(46,158,107,0.04)")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below")
    fig.add_vline(x=0.7, line_dash="dot", line_color=COLOR_PRIMARIO, opacity=0.5,
                  annotation_text="Muy apto", annotation_position="top",
                  annotation_font_color=COLOR_PRIMARIO)
    fig.update_xaxes(range=[0, 1.08], title="Índice de Idoneidad Agrícola (IAI)")
    fig.update_yaxes(title="")
    fig.update_coloraxes(showscale=False)
    _estilo(fig, alto=450)
    _grafico(fig)


def _frase_cambio(v0, v1):
    if v0 <= 0:
        return "sin cambios relevantes"
    cambio = (v1 - v0) / v0 * 100
    if abs(cambio) < 3:
        return "se mantendría prácticamente igual"
    verbo = "subiría" if cambio > 0 else "bajaría"
    return f"{verbo} un {abs(cambio):.0f}%"


def render_evolucion(resultado, ranking_hoy):
    iai = resultado["iai"]
    nombre_top = ranking_hoy[0][0]
    top_id = [k for k, v in CROP_DICT_ES.items() if v == nombre_top][0]
    anios = [2025, 2030, 2040]
    mod = [iai["2025"][top_id], iai["2030-245"][top_id], iai["2040-245"][top_id]]
    sev = [iai["2025"][top_id], iai["2030-585"][top_id], iai["2040-585"][top_id]]

    seccion(f"¿Seguirá siendo buena idea cultivar {nombre_top.lower()}?",
            "Evolución de la idoneidad de tu mejor cultivo en dos escenarios de cambio climático.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anios, y=mod, name="Moderado (SSP2-4.5)",
                             mode="lines+markers", line=dict(color=COLOR_MOD, width=4),
                             marker=dict(size=11), fill="tozeroy",
                             fillcolor="rgba(46,158,107,0.08)"))
    fig.add_trace(go.Scatter(x=anios, y=sev, name="Severo (SSP5-8.5)",
                             mode="lines+markers", line=dict(color=COLOR_SEV, width=4, dash="dot"),
                             marker=dict(size=11)))
    fig.update_xaxes(tickmode="array", tickvals=anios)
    fig.update_yaxes(range=[0, 1], title="Idoneidad (IAI)")
    _estilo(fig, alto=320)
    _grafico(fig)

    info(f"Hacia 2040, la idoneidad de <b>{nombre_top}</b> {_frase_cambio(mod[0], mod[2])} "
         f"en el escenario moderado y {_frase_cambio(sev[0], sev[2])} en el escenario severo.")


def render_clima(resultado):
    clima = resultado["clima"]
    anios = [2025, 2030, 2040]
    seccion("Cómo cambiará el clima en tu zona",
            "Proyecciones de la NASA (NEX-GDDP-CMIP6). El escenario moderado supone que las "
            "emisiones se estabilizan; el severo, que siguen creciendo.")
    cc1, cc2 = st.columns(2)

    def _linea(var, titulo_y, caption):
        st.caption(caption)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anios, y=[clima["2025"][var], clima["2030-245"][var], clima["2040-245"][var]],
            name="Moderado", mode="lines+markers", line=dict(color=COLOR_MOD, width=4),
            marker=dict(size=10)))
        fig.add_trace(go.Scatter(
            x=anios, y=[clima["2025"][var], clima["2030-585"][var], clima["2040-585"][var]],
            name="Severo", mode="lines+markers", line=dict(color=COLOR_SEV, width=4, dash="dot"),
            marker=dict(size=10)))
        fig.update_xaxes(tickmode="array", tickvals=anios)
        fig.update_yaxes(title=titulo_y)
        _estilo(fig, alto=290)
        _grafico(fig)

    with cc1:
        _linea("tmax", "°C", "Temperatura máxima media (°C)")
    with cc2:
        _linea("ppt", "mm", "Lluvia anual (mm)")


def render_tabla(resultado):
    iai = resultado["iai"]
    with st.expander("Ver los 12 cultivos en detalle (todos los años y escenarios)"):
        filas = []
        for cid in FINAL_CROPS:
            filas.append({
                "Cultivo": CROP_DICT_ES[cid],
                "Hoy 2025": round(iai["2025"][cid], 2),
                "Valoración": categoria_iai(iai["2025"][cid])[0],
                "2030 mod.": round(iai["2030-245"][cid], 2),
                "2040 mod.": round(iai["2040-245"][cid], 2),
                "2030 sev.": round(iai["2030-585"][cid], 2),
                "2040 sev.": round(iai["2040-585"][cid], 2),
            })
        df_tabla = pd.DataFrame(filas).sort_values("Hoy 2025", ascending=False)
        st.dataframe(
            df_tabla, hide_index=True, **ANCHO,
            column_config={
                "Hoy 2025": st.column_config.ProgressColumn(
                    "Hoy 2025", min_value=0, max_value=1, format="%.2f"),
            })
        st.caption("mod. = escenario moderado (SSP2-4.5) · sev. = escenario severo (SSP5-8.5)")


def render_resultados(resultado, con_zona=True):
    """Bloque completo de resultados."""
    if "error" in resultado:
        st.warning(resultado["error"])
        return
    if con_zona:
        render_zona(resultado)
    render_leyenda()
    ranking_hoy = render_top3(resultado)
    render_resiliente(resultado)
    render_barras(resultado)
    render_evolucion(resultado, ranking_hoy)
    render_clima(resultado)
    render_tabla(resultado)


# ══════════════════════════════════════════════════════════════════
#  ESTADO DE LA UBICACIÓN (compartido entre pestañas)
# ══════════════════════════════════════════════════════════════════
ss = st.session_state
if "sel_lon" not in ss:
    ss.sel_lon, ss.sel_lat = -119.8, 36.7      # Fresno, en pleno Valle Central
    ss.num_lon, ss.num_lat = -119.8, 36.7


def fijar_ubicacion(lon, lat):
    """Guarda el punto y deja pendiente la actualización de las casillas.

    Streamlit no permite cambiar el valor de un widget después de dibujarlo,
    por eso las casillas se actualizan al inicio de la siguiente ejecución.
    """
    ss.sel_lon, ss.sel_lat = lon, lat
    ss._pend_coords = (round(lon, 4), round(lat, 4))


def _sync_num():
    ss.sel_lon, ss.sel_lat = ss.num_lon, ss.num_lat


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — MI LOTE
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_izq, col_der = st.columns([1.15, 1], gap="large")

    with col_izq:
        paso(1, "Ubica tu parcela",
             "Elige la forma que te resulte más cómoda: escribe una ciudad, "
             "introduce coordenadas o haz clic en el mapa.")

        direccion = st.text_input(
            "Buscar dirección o ciudad de California",
            placeholder="Ejemplo: Fresno, Salinas, Bakersfield...",
            help="Escribe el nombre y pulsa Enter. Solo se buscan lugares de California.")

        # Solo se geocodifica cuando cambia el texto; así un clic en el mapa
        # no se sobrescribe en la siguiente ejecución.
        if direccion and direccion != ss.get("_ult_dir"):
            ss._ult_dir = direccion
            try:
                coords = geocodificar(direccion)
                if coords:
                    fijar_ubicacion(*coords)
                    ss._msg_dir = ("ok", f"Ubicación encontrada: {coords[1]:.4f}, {coords[0]:.4f}")
                else:
                    ss._msg_dir = ("warn", "No encontramos esa dirección. Prueba con el nombre "
                                           "de la ciudad o haz clic en el mapa.")
            except Exception:
                ss._msg_dir = ("warn", "El buscador no está disponible ahora mismo. "
                                       "Usa las coordenadas o el mapa.")
        if not direccion:
            ss._ult_dir, ss._msg_dir = None, None
        if ss.get("_msg_dir"):
            tipo, texto = ss._msg_dir
            (st.success if tipo == "ok" else st.warning)(texto)

        # Aplicar coordenadas pendientes antes de dibujar las casillas
        if "_pend_coords" in ss:
            ss.num_lon, ss.num_lat = ss._pend_coords
            del ss._pend_coords

        c1, c2 = st.columns(2)
        c1.number_input("Latitud", key="num_lat", format="%.4f", step=0.01,
                        on_change=_sync_num,
                        help="En California va aproximadamente de 32.5 (sur) a 42.0 (norte).")
        c2.number_input("Longitud", key="num_lon", format="%.4f", step=0.01,
                        on_change=_sync_num,
                        help="En California es negativa: entre -124.4 (costa) y -114.1 (este).")

        lon_sel, lat_sel = ss.sel_lon, ss.sel_lat
        dentro = en_california(lon_sel, lat_sel)

        m = folium.Map(location=[lat_sel, lon_sel], zoom_start=6,
                       tiles="OpenStreetMap", control_scale=True)
        folium.Polygon(
            locations=[(lat, lon) for lon, lat in CALIFORNIA_POLY],
            color=COLOR_PRIMARIO, weight=2, dash_array="6 6",
            fill=True, fill_color=COLOR_LIMA, fill_opacity=0.06,
            tooltip="Zona disponible: California",
        ).add_to(m)
        folium.Marker(
            [lat_sel, lon_sel], tooltip="Tu parcela",
            icon=folium.Icon(color="green" if dentro else "red", icon="leaf", prefix="fa"),
        ).add_to(m)
        map_data = st_folium(m, height=360, width=None, key="mapa_lote",
                             returned_objects=["last_clicked"])

        if map_data and map_data.get("last_clicked"):
            clat = map_data["last_clicked"]["lat"]
            clon = map_data["last_clicked"]["lng"]
            if (round(clon, 4) != round(ss.sel_lon, 4) or
                    round(clat, 4) != round(ss.sel_lat, 4)):
                fijar_ubicacion(clon, clat)
                st.rerun()

        consejo("la línea discontinua marca California, que es la zona donde funciona el "
                "recomendador. Haz clic dentro de ella para mover tu parcela.")

    with col_der:
        paso(2, "Tu parcela de un vistazo",
             "Datos de clima y suelo del punto más cercano de nuestra base.")

        resultado = None
        if not dentro:
            aviso_fuera_california(lon_sel, lat_sel)
        else:
            fila, dist = punto_mas_cercano(lon_sel, lat_sel)
            resultado = motor.predecir_punto(fila)
            if "error" in resultado:
                st.warning(resultado["error"])
            else:
                render_zona(resultado)
            render_condiciones(fila)
            if dist > DIST_AVISO:
                st.warning(
                    f"Tu parcela está a unos {dist*111:.0f} km del punto más cercano con datos "
                    f"(zonas de montaña, desierto o tierras federales). El resultado es solo "
                    f"orientativo. Si conoces las condiciones de tu parcela, usa la pestaña "
                    f"**Introduce tus datos**.")
            else:
                consejo("si tienes un análisis de suelo propio, ve a la pestaña "
                        "<b>Introduce tus datos</b> para afinar la recomendación.")

    if resultado is not None and "error" not in resultado:
        st.divider()
        paso(3, "Tus resultados",
             "Primero verás los cultivos más aptos y después cómo podrían cambiar con el clima.")
        render_resultados(resultado, con_zona=False)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — MAPAS POR CULTIVO
# ══════════════════════════════════════════════════════════════════
with tab2:
    paso(1, "Explora un cultivo en toda California",
         "Elige un cultivo y un año para ver dónde se adapta mejor. "
         "Pasa el ratón sobre el mapa para ver el valor de cada punto.")

    fc1, fc2 = st.columns(2)
    cultivo_sel = fc1.selectbox("Cultivo", sorted(CROP_DICT_ES.values()))
    escenarios_disp = ["2025", "2030 · SSP2-4.5", "2040 · SSP2-4.5",
                       "2030 · SSP5-8.5", "2040 · SSP5-8.5"]
    nombres_esc = {
        "2025": "Hoy (2025)",
        "2030 · SSP2-4.5": "2030 · escenario moderado",
        "2040 · SSP2-4.5": "2040 · escenario moderado",
        "2030 · SSP5-8.5": "2030 · escenario severo",
        "2040 · SSP5-8.5": "2040 · escenario severo",
    }
    escenario_sel = fc2.selectbox("Año y escenario", escenarios_disp,
                                  format_func=lambda x: nombres_esc.get(x, x),
                                  help="Moderado (SSP2-4.5): las emisiones se estabilizan. "
                                       "Severo (SSP5-8.5): las emisiones siguen creciendo.")

    datos_mapa = iai_mapas.filter(
        (pl.col("cultivo") == cultivo_sel) & (pl.col("escenario") == escenario_sel)
    ).select(["lon", "lat", "iai"]).to_pandas()

    if len(datos_mapa) == 0:
        st.warning("No hay datos para esta combinación de cultivo y escenario.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Idoneidad media", f"{datos_mapa['iai'].mean():.2f}")
        e2.metric("Zonas muy aptas (IAI > 0.7)", f"{(datos_mapa['iai'] > 0.7).mean()*100:.0f}%")
        e3.metric("Puntos evaluados", f"{len(datos_mapa):,}".replace(",", "."))

        colores = datos_mapa["iai"].apply(color_iai)
        datos_mapa["r"] = colores.apply(lambda c: c[0])
        datos_mapa["g"] = colores.apply(lambda c: c[1])
        datos_mapa["b"] = colores.apply(lambda c: c[2])
        datos_mapa["iai_txt"] = datos_mapa["iai"].map(lambda v: f"{v:.2f}")
        datos_mapa["categoria"] = datos_mapa["iai"].map(lambda v: categoria_iai(v)[0])

        capa = pdk.Layer(
            "ScatterplotLayer", data=datos_mapa,
            get_position=["lon", "lat"], get_fill_color=["r", "g", "b", 190],
            get_radius=2500, pickable=True,
        )
        vista = pdk.ViewState(latitude=37.2, longitude=-119.5, zoom=5.2)
        st.pydeck_chart(pdk.Deck(
            layers=[capa], initial_view_state=vista, map_style="light",
            tooltip={"html": "<b>" + cultivo_sel + "</b><br>IAI: {iai_txt}<br>{categoria}",
                     "style": {"backgroundColor": "white", "color": COLOR_TEXTO,
                               "fontFamily": "Nunito", "borderRadius": "8px"}}))

        st.markdown(_limpiar(f"""
        <div style="display:flex; align-items:center; gap:10px; font-size:13px; color:{COLOR_SUAVE};
                    margin-top:8px;">
          <span>Baja idoneidad</span>
          <div style="flex:1; height:12px; border-radius:6px; max-width:340px;
                      background:linear-gradient(to right,#D9653B,#F2C66D,#0B5D45);"></div>
          <span>Alta idoneidad</span>
        </div>
        """), unsafe_allow_html=True)

        consejo("compara el mismo cultivo en <b>Hoy (2025)</b> y en <b>2040 · escenario severo</b> "
                "para ver qué zonas podrían dejar de ser aptas.")


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 3 — INTRODUCE TUS DATOS
# ══════════════════════════════════════════════════════════════════
# Relación entre las casillas del formulario y las columnas de la base
CAMPOS_MANUALES = {
    "m_tmax": "tmax", "m_tmin": "tmin", "m_ppt": "ppt", "m_vpd": "vpdmean",
    "m_ph": "ph1to1h2o_r", "m_awc": "awc_r", "m_prof": "profundidad_efectiva_cm",
    "m_clay": "claytotal_r", "m_sand": "sandtotal_r", "m_silt": "silttotal_r",
    "m_db": "dbthirdbar_r",
}


def _rellenar_con_zona():
    """Rellena el formulario con los valores de la base para la ubicación indicada."""
    if not en_california(ss.m_lon, ss.m_lat):
        ss._msg_rell = "warn"
        return
    fila, _ = punto_mas_cercano(ss.m_lon, ss.m_lat)
    for clave, col in CAMPOS_MANUALES.items():
        ss[clave] = float(round(fila[col], 2))
    ss._msg_rell = "ok"
    ss.res_manual = None


def _usar_ubicacion_lote():
    ss.m_lon, ss.m_lat = round(ss.sel_lon, 4), round(ss.sel_lat, 4)
    _rellenar_con_zona()


# Valores iniciales: la ubicación de "Mi lote" y su clima/suelo
if "m_lon" not in ss:
    ss.m_lon, ss.m_lat = round(ss.sel_lon, 4), round(ss.sel_lat, 4)
    if en_california(ss.m_lon, ss.m_lat):
        _rellenar_con_zona()
    ss._msg_rell = None
for _k, _v in {"m_tmax": 24.0, "m_tmin": 10.0, "m_ppt": 330.0, "m_vpd": 15.0,
               "m_ph": 7.0, "m_awc": 0.15, "m_prof": 100.0, "m_clay": 25.0,
               "m_sand": 40.0, "m_silt": 35.0, "m_db": 1.4}.items():
    ss.setdefault(_k, _v)
ss.setdefault("res_manual", None)

with tab3:
    info("<b>¿Para qué sirve esta pestaña?</b> Si conoces mejor que nadie tu parcela "
         "(por ejemplo, tienes un análisis de suelo o una estación meteorológica), puedes "
         "ajustar aquí los datos. Solo necesitas el <b>clima actual</b> y el <b>suelo</b>: "
         "el clima futuro lo calculamos nosotros con las proyecciones de la NASA para tu zona.")

    # ── Paso 1: ubicación ──
    paso(1, "¿Dónde está tu parcela?",
         "La ubicación se usa para elegir el modelo de tu zona y las proyecciones de clima futuro.")
    u1, u2, u3 = st.columns([1, 1, 1.2])
    u1.number_input("Latitud", key="m_lat", format="%.4f", step=0.01,
                    on_change=_rellenar_con_zona)
    u2.number_input("Longitud", key="m_lon", format="%.4f", step=0.01,
                    on_change=_rellenar_con_zona)
    with u3:
        st.write("")
        st.button("Usar la ubicación de 'Mi lote'", on_click=_usar_ubicacion_lote, **ANCHO)

    ubic_ok = en_california(ss.m_lon, ss.m_lat)
    if not ubic_ok:
        aviso_fuera_california(ss.m_lon, ss.m_lat)
    else:
        # ── Paso 2: datos de la parcela ──
        paso(2, "Revisa y ajusta los datos de tu parcela",
             "Hemos rellenado el formulario con los valores típicos de tu zona. "
             "Cambia solo los que conozcas; el resto puedes dejarlos como están.")
        st.button("Volver a cargar los valores típicos de mi zona", on_click=_rellenar_con_zona)
        if ss.get("_msg_rell") == "ok":
            st.success("Valores de tu zona cargados. Ahora ajusta los que conozcas.")
            ss._msg_rell = None

        st.markdown("##### Clima actual")
        m1, m2, m3 = st.columns(3)
        m1.number_input("Temperatura máxima media (°C)", key="m_tmax", step=0.5,
                        help="Media anual de las temperaturas máximas diarias. "
                             "En el Valle Central suele estar entre 22 y 27 °C.")
        m2.number_input("Temperatura mínima media (°C)", key="m_tmin", step=0.5,
                        help="Media anual de las temperaturas mínimas diarias.")
        m3.number_input("Lluvia anual (mm)", key="m_ppt", step=10.0, min_value=0.0,
                        help="Precipitación total de un año normal. "
                             "Va desde menos de 100 mm en el desierto a más de 1.500 mm en el norte.")

        st.markdown("##### Suelo")
        s1, s2, s3, s4 = st.columns(4)
        s1.number_input("pH", key="m_ph", step=0.1, min_value=3.0, max_value=11.0,
                        help="Acidez del suelo. 7 es neutro; la mayoría de cultivos "
                             "prefieren valores entre 6 y 7.5.")
        s2.number_input("Arcilla (%)", key="m_clay", step=1.0, min_value=0.0, max_value=100.0,
                        help="Lo encontrarás en el análisis de textura del suelo.")
        s3.number_input("Arena (%)", key="m_sand", step=1.0, min_value=0.0, max_value=100.0)
        s4.number_input("Limo (%)", key="m_silt", step=1.0, min_value=0.0, max_value=100.0)

        suma_textura = ss.m_clay + ss.m_sand + ss.m_silt
        if abs(suma_textura - 100) > 2:
            st.warning(f"Arcilla + arena + limo suman {suma_textura:.0f}%. "
                       f"Lo normal es que sumen 100%; revisa los valores.")

        with st.expander("Datos avanzados (opcional, si tienes un análisis de suelo completo)"):
            st.caption("Si no conoces estos valores, déjalos como están: son los de tu zona.")
            a1, a2 = st.columns(2)
            a1.number_input("Profundidad efectiva (cm)", key="m_prof", step=5.0, min_value=0.0,
                            help="Profundidad que pueden explorar las raíces antes de "
                                 "encontrar roca o una capa dura.")
            a2.number_input("Capacidad de agua disponible (AWC, cm/cm)", key="m_awc",
                            step=0.01, min_value=0.0, max_value=1.0, format="%.2f",
                            help="Agua que el suelo puede retener para las plantas. "
                                 "Suelos arenosos ≈ 0.05; francos ≈ 0.15–0.20.")
            a3, a4 = st.columns(2)
            a3.number_input("Densidad aparente (g/cm³)", key="m_db", step=0.05, format="%.2f",
                            help="Indica la compactación. Valores típicos: 1.1–1.6.")
            a4.number_input("Déficit de presión de vapor medio (VPD)", key="m_vpd", step=0.5,
                            help="Mide lo seco que está el aire. Si no lo conoces, "
                                 "deja el valor de tu zona.")

        # ── Paso 3: calcular ──
        paso(3, "Calcula tu recomendación",
             "El clima de 2030 y 2040 se estima automáticamente; no necesitas introducirlo.")

        if st.button("Calcular recomendación", type="primary"):
            fila_zona, dist_zona = punto_mas_cercano(ss.m_lon, ss.m_lat)
            punto_manual = {col: ss[clave] for clave, col in CAMPOS_MANUALES.items()}
            punto_manual.update(
                clima_futuro_desde_base(fila_zona, ss.m_tmax, ss.m_tmin, ss.m_ppt))
            # Se conservan lon/lat por si el predictor los utiliza
            punto_manual.update({"lon": ss.m_lon, "lat": ss.m_lat})
            ss.res_manual = {
                "resultado": motor.predecir_punto(punto_manual),
                "dist": dist_zona,
                "delta_t": fila_zona["tmax_245_2040"] - fila_zona["tmax"],
            }

        if ss.res_manual:
            st.divider()
            if ss.res_manual["dist"] > DIST_AVISO:
                st.warning("Tu ubicación está lejos de los puntos con datos; las proyecciones "
                           "de clima futuro son aproximadas.")
            info(f"Para tu zona, las proyecciones indican un cambio de "
                 f"<b>{ss.res_manual['delta_t']:+.1f} °C</b> en la temperatura máxima hacia 2040 "
                 f"(escenario moderado). Ese cambio se ha sumado a los datos que has introducido.")
            render_resultados(ss.res_manual["resultado"])


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 4 — ACERCA DE
# ══════════════════════════════════════════════════════════════════
with tab4:
    a1, a2 = st.columns([1.6, 1], gap="large")

    with a1:
        st.markdown("#### Sobre el sistema de recomendación")
        st.markdown("""
        Esta aplicación forma parte de un Trabajo de Fin de Máster en Ciencia de Datos.
        El objetivo es ayudar a decidir qué cultivos son más adecuados para una parcela
        en California, tanto hoy como bajo distintos escenarios de cambio climático.

        **Cómo funciona**

        El sistema se basa en un **Índice de Idoneidad Agrícola (IAI)**, un valor entre
        0 y 1 que resume qué tan apto es un lugar para un cultivo. El índice combina
        variables de clima (temperatura, lluvia) y de suelo (pH, textura, profundidad,
        capacidad de retención de agua).

        Para estimar el IAI se entrenaron modelos de aprendizaje automático (LightGBM)
        especializados por región. California se dividió en cinco zonas biofísicas
        mediante un análisis de agrupamiento, y cada zona tiene su propio modelo.

        **Los escenarios de futuro**

        Las proyecciones de clima provienen del modelo NASA NEX-GDDP-CMIP6, para dos
        escenarios de emisiones: uno moderado (SSP2-4.5) y uno severo (SSP5-8.5),
        en los horizontes 2030 y 2040.

        Cuando el productor introduce sus propios datos, el clima futuro se obtiene con
        el **método delta**: se calcula el cambio que proyecta el modelo en su zona
        (grados de más en temperatura y porcentaje de cambio en lluvia) y se aplica
        sobre el clima actual que ha indicado. De esta forma el productor no necesita
        conocer el clima futuro.

        **Limitaciones**

        La aplicación solo ofrece resultados dentro de California. Los datos de suelo
        cubren principalmente las zonas agrícolas; en áreas de montaña, desierto o
        tierras federales puede no haber información cercana. Los resultados son
        orientativos y de carácter académico, y no sustituyen el criterio técnico
        agronómico.
        """)

    with a2:
        st.markdown(_limpiar(f"""
        <div class='bloque'>
          <div class='etiqueta-mini'>Ficha del proyecto</div>
          <div style='font-size:14px; color:{COLOR_TEXTO}; line-height:1.9; margin-top:8px;'>
            <b>Autora</b><br>Leslie Estefany Mosquera<br><br>
            <b>Tutor</b><br>José Lloreda Sánchez<br><br>
            <b>Institución</b><br>La Salle – Universitat Ramon Llull<br>
            Máster en Ciencia de Datos
          </div>
        </div>
        <div class='bloque'>
          <div class='etiqueta-mini'>Fuentes de datos</div>
          <div style='font-size:14px; color:{COLOR_TEXTO}; line-height:1.9; margin-top:8px;'>
            <b>Clima actual:</b> PRISM (4 km)<br>
            <b>Clima futuro:</b> NASA NEX-GDDP-CMIP6<br>
            <b>Suelo:</b> gSSURGO (USDA)
          </div>
        </div>
        """), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PIE
# ══════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='pie'>Recomendador de cultivos · TFM Máster en Ciencia de Datos · "
    "Leslie Estefany Mosquera · Tutor: José Lloreda Sánchez · "
    "La Salle – Universitat Ramon Llull · Resultados orientativos de carácter académico</div>",
    unsafe_allow_html=True)
