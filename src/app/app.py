"""
================================================================================
 APP — RECOMENDADOR DE CULTIVOS PARA CALIFORNIA
================================================================================
Aplicación Streamlit para agricultores. Predice el Índice de Idoneidad Agrícola
(IAI) de 12 cultivos y muestra cómo cambia con el clima futuro (2030 y 2040).

Pestañas:
  1. Mi lote          — ubicación + cultivos aptos + gráficos + evolución clima
  2. Mapas por cultivo — coropléticos del IAI filtrables por cultivo y año
  3. Datos manuales    — el productor introduce sus propios datos
  4. Acerca de         — metodología del proyecto
================================================================================
"""

import os
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree
import pydeck as pdk
import folium
from streamlit_folium import st_folium

from predictor import (
    PredictorCultivos, VARS_CLUSTER, FINAL_CROPS,
    CROP_DICT_EN, CROP_DICT_ES, ETIQUETAS_CLUSTER,
)

# ─── Rutas ───
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RUTA_BASE = os.path.join(PROJECT_ROOT, "Data", "base_california_app.parquet")
RUTA_IAI_MAPAS = os.path.join(PROJECT_ROOT, "Data", "iai_mapas_california.parquet")

# ─── Paleta ───
COLOR_FONDO = "#F4F9F4"
COLOR_PRIMARIO = "#0F6E56"
COLOR_ACENTO = "#1D9E75"

st.set_page_config(page_title="Recomendador de cultivos", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_FONDO}; }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; background-color: #E4EFE4; padding: 4px; border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px; padding: 8px 20px; color: {COLOR_PRIMARIO}; font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: white; color: {COLOR_PRIMARIO}; font-weight: 600;
    }}
    .bloque {{ background: white; border-radius: 12px; padding: 16px;
               border: 1px solid #DDE8DD; margin-bottom: 12px; }}
    .titulo-zona {{ color: {COLOR_PRIMARIO}; font-weight: 600; font-size: 18px; }}
    h4 {{ color: {COLOR_PRIMARIO} !important; }}
    section[data-testid="stSidebar"] {{ background-color: #E9F3E9; }}
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

motor = cargar_motor()
base = cargar_base()
iai_mapas = cargar_iai_mapas()
tree = construir_kdtree(base)


def punto_mas_cercano(lon, lat):
    dist, idx = tree.query([lon, lat])
    return base.row(idx, named=True), dist


def color_iai(v):
    """IAI [0,1] -> color RGB (rojo->amarillo->verde)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return [200, 200, 200]  # gris para valores faltantes
    if v < 0.5:
        t = v / 0.5
        return [int(216+(250-216)*t), int(90+(199-90)*t), int(48+(117-48)*t)]
    t = (v - 0.5) / 0.5
    return [int(250+(15-250)*t), int(199+(110-199)*t), int(86+(86-117)*t)]
# ══════════════════════════════════════════════════════════════════
#  BARRA LATERAL
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div style='font-size:18px; font-weight:600; color:{COLOR_PRIMARIO};'>"
                f"Sobre este proyecto</div>", unsafe_allow_html=True)
    st.markdown("""
    Sistema de recomendación de cultivos para California basado en un
    **Índice de Idoneidad Agrícola (IAI)** que combina clima y suelo.

    El sistema estima la idoneidad de 12 cultivos en el presente y bajo
    dos escenarios de cambio climático (2030 y 2040).

    ---
    **TFM · Máster en Ciencia de Datos**
    La Salle – Universitat Ramon Llull

    **Datos:** PRISM (clima), NASA NEX-GDDP-CMIP6 (proyecciones),
    gSSURGO (suelo).
    """)
    st.caption("Los resultados son orientativos y de carácter académico.")


# ══════════════════════════════════════════════════════════════════
#  CABECERA
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="border-left:4px solid {COLOR_ACENTO}; padding-left:14px; margin-bottom:16px;">
    <div style="font-size:24px; font-weight:600; color:{COLOR_PRIMARIO};">Recomendador de cultivos</div>
    <div style="font-size:14px; color:#5F7A5F;">California · idoneidad agrícola presente y futura</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Mi lote", "Mapas por cultivo", "Datos manuales", "Acerca de"])


# ══════════════════════════════════════════════════════════════════
#  RESULTADOS DE UN PUNTO
# ══════════════════════════════════════════════════════════════════
def mostrar_resultados(resultado):
    if "error" in resultado:
        st.warning(resultado["error"])
        return

    iai = resultado["iai"]
    clima = resultado["clima"]

    st.markdown(f"<div class='bloque'><span class='titulo-zona'>Tu zona: "
                f"{resultado['etiqueta']}</span></div>", unsafe_allow_html=True)

    # Top 3 hoy
    ranking_hoy = motor.ranking(iai["2025"], top=3)
    st.markdown("#### Los 3 cultivos más aptos hoy")
    cols = st.columns(3)
    for i, (nombre, val) in enumerate(ranking_hoy):
        with cols[i]:
            st.markdown(f"""
            <div class='bloque' style='text-align:center; border-top:3px solid {COLOR_ACENTO};'>
              <div style='font-size:13px; color:#5F7A5F; font-weight:500;'>Opción {i+1}</div>
              <div style='font-size:19px; font-weight:600; color:{COLOR_PRIMARIO}; margin:4px 0;'>{nombre}</div>
              <div style='font-size:26px; font-weight:600; color:{COLOR_ACENTO};'>{val:.2f}</div>
              <div style='font-size:12px; color:#5F7A5F;'>idoneidad</div>
            </div>
            """, unsafe_allow_html=True)

    # Gráfico de barras: los 12 cultivos hoy
    st.markdown("#### Idoneidad de todos los cultivos hoy")
    df_barras = pd.DataFrame({
        "Cultivo": [CROP_DICT_ES[c] for c in FINAL_CROPS],
        "Idoneidad": [iai["2025"][c] for c in FINAL_CROPS],
    }).sort_values("Idoneidad", ascending=True)
    st.bar_chart(df_barras.set_index("Cultivo"), horizontal=True, color=COLOR_ACENTO)

    # Evolución de la idoneidad del cultivo top
    st.markdown("#### Cómo cambia la idoneidad de tu mejor cultivo")
    top_id = [k for k, v in CROP_DICT_ES.items() if v == ranking_hoy[0][0]][0]
    df_pivot = pd.DataFrame({
        "Año": [2025, 2030, 2040],
        "Moderado (SSP2-4.5)": [iai["2025"][top_id], iai["2030-245"][top_id], iai["2040-245"][top_id]],
        "Severo (SSP5-8.5)": [iai["2025"][top_id], iai["2030-585"][top_id], iai["2040-585"][top_id]],
    }).set_index("Año")
    st.line_chart(df_pivot)

    # Evolución del clima
    st.markdown("#### Cómo cambia el clima en tu zona")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.caption("Temperatura máxima (°C)")
        df_t = pd.DataFrame({
            "Año": [2025, 2030, 2040],
            "Moderado": [clima["2025"]["tmax"], clima["2030-245"]["tmax"], clima["2040-245"]["tmax"]],
            "Severo": [clima["2025"]["tmax"], clima["2030-585"]["tmax"], clima["2040-585"]["tmax"]],
        }).set_index("Año")
        st.line_chart(df_t)
    with cc2:
        st.caption("Lluvia anual (mm)")
        df_p = pd.DataFrame({
            "Año": [2025, 2030, 2040],
            "Moderado": [clima["2025"]["ppt"], clima["2030-245"]["ppt"], clima["2040-245"]["ppt"]],
            "Severo": [clima["2025"]["ppt"], clima["2030-585"]["ppt"], clima["2040-585"]["ppt"]],
        }).set_index("Año")
        st.line_chart(df_p)

    # Tabla completa
    with st.expander("Ver los 12 cultivos en detalle (todos los años)"):
        filas = []
        for cid in FINAL_CROPS:
            filas.append({
                "Cultivo": CROP_DICT_ES[cid],
                "Hoy 2025": round(iai["2025"][cid], 2),
                "2030 mod.": round(iai["2030-245"][cid], 2),
                "2040 mod.": round(iai["2040-245"][cid], 2),
                "2030 sev.": round(iai["2030-585"][cid], 2),
                "2040 sev.": round(iai["2040-585"][cid], 2),
            })
        df_tabla = pd.DataFrame(filas).sort_values("Hoy 2025", ascending=False)
        st.dataframe(df_tabla, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — MI LOTE
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("Ubica tu parcela: escribe una dirección, introduce las coordenadas "
                "o toca un punto en el mapa.")

    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        direccion = st.text_input("Buscar dirección o ciudad", placeholder="Fresno, California")
        cc1, cc2 = st.columns(2)
        lon_in = cc1.number_input("Longitud", value=-119.8, format="%.4f")
        lat_in = cc2.number_input("Latitud", value=36.7, format="%.4f")

        lon_sel, lat_sel = lon_in, lat_in
        if direccion:
            try:
                from geopy.geocoders import Nominatim
                geo = Nominatim(user_agent="recomendador_cultivos_tfm")
                loc = geo.geocode(f"{direccion}, California, USA", timeout=10)
                if loc:
                    lon_sel, lat_sel = loc.longitude, loc.latitude
                    st.success(f"Ubicación: {lon_sel:.4f}, {lat_sel:.4f}")
                else:
                    st.warning("No se encontró la dirección. Usa las coordenadas.")
            except Exception as e:
                st.warning(f"No se pudo buscar la dirección: {e}")

        m = folium.Map(location=[lat_sel, lon_sel], zoom_start=7, tiles="CartoDB positron")
        folium.Marker([lat_sel, lon_sel], tooltip="Tu lote",
                      icon=folium.Icon(color="green", icon="leaf")).add_to(m)
        map_data = st_folium(m, height=280, width=None, key="mapa_lote")
        if map_data and map_data.get("last_clicked"):
            lat_sel = map_data["last_clicked"]["lat"]
            lon_sel = map_data["last_clicked"]["lng"]
            st.info(f"Punto seleccionado: {lon_sel:.4f}, {lat_sel:.4f}")

        fila, dist = punto_mas_cercano(lon_sel, lat_sel)
        st.markdown("###### Condiciones de tu parcela")
        st.markdown(
            f"<div style='font-size:13px; color:#3A5A3A;'>"
            f"Temp. máx: <b>{fila['tmax']:.0f}°C</b> · "
            f"Temp. mín: <b>{fila['tmin']:.0f}°C</b> · "
            f"Lluvia: <b>{fila['ppt']:.0f} mm</b><br>"
            f"pH: <b>{fila['ph1to1h2o_r']:.1f}</b> · "
            f"Prof.: <b>{fila['profundidad_efectiva_cm']:.0f} cm</b> · "
            f"Arcilla: <b>{fila['claytotal_r']:.0f}%</b></div>",
            unsafe_allow_html=True)

    with col_der:
        if dist > 0.15:
            st.warning(
                f"La ubicación está fuera de la zona con datos "
                f"(a {dist*111:.0f} km del punto más cercano con información). "
                f"La predicción es orientativa. Puedes usar 'Datos manuales' "
                f"si conoces las condiciones de tu parcela."
            )
        resultado = motor.predecir_punto(fila)
        mostrar_resultados(resultado)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — MAPAS POR CULTIVO
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("Explora la idoneidad de cada cultivo en toda California, "
                "hoy y en el futuro.")

    fc1, fc2 = st.columns(2)
    cultivo_sel = fc1.selectbox("Cultivo", sorted(CROP_DICT_ES.values()))
    escenarios_disp = ["2025", "2030 · SSP2-4.5", "2040 · SSP2-4.5",
                       "2030 · SSP5-8.5", "2040 · SSP5-8.5"]
    escenario_sel = fc2.selectbox("Año y escenario", escenarios_disp,
                                  format_func=lambda x: "Hoy (2025)" if x == "2025" else x)

    datos_mapa = iai_mapas.filter(
        (pl.col("cultivo") == cultivo_sel) & (pl.col("escenario") == escenario_sel)
    ).select(["lon", "lat", "iai"]).to_pandas()

    e1, e2, e3 = st.columns(3)
    e1.metric("Idoneidad media", f"{datos_mapa['iai'].mean():.2f}")
    e2.metric("Zonas muy aptas (>0.7)", f"{(datos_mapa['iai']>0.7).mean()*100:.0f}%")
    e3.metric("Puntos evaluados", f"{len(datos_mapa):,}")

    colores_rgb = datos_mapa["iai"].apply(color_iai)
    datos_mapa[["r", "g", "b"]] = pd.DataFrame(colores_rgb.tolist(), index=datos_mapa.index)

    capa = pdk.Layer(
        "ScatterplotLayer", data=datos_mapa,
        get_position=["lon", "lat"], get_fill_color=["r", "g", "b", 180],
        get_radius=2500, pickable=True,
    )
    vista = pdk.ViewState(latitude=37.2, longitude=-119.5, zoom=5.2)
    st.pydeck_chart(pdk.Deck(layers=[capa], initial_view_state=vista,
                             map_style="light", tooltip={"text": "IAI: {iai}"}))

    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#5F7A5F;">
      <span>Baja idoneidad</span>
      <div style="flex:1; height:12px; border-radius:6px; max-width:300px;
                  background:linear-gradient(to right,#D85A30,#FAC775,#0F6E56);"></div>
      <span>Alta idoneidad</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 3 — DATOS MANUALES
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("Introduce los datos de tu parcela para obtener una recomendación.")

    st.markdown("##### Clima actual")
    m1, m2c, m3 = st.columns(3)
    tmax_m = m1.number_input("Temp. máxima (°C)", value=24.0)
    tmin_m = m2c.number_input("Temp. mínima (°C)", value=10.0)
    ppt_m = m3.number_input("Lluvia anual (mm)", value=330.0)
    vpd_m = st.number_input("VPD medio de la zona", value=15.0,
                            help="Déficit de presión de vapor; si no lo conoces, deja el valor por defecto.")

    st.markdown("##### Suelo")
    s1, s2, s3 = st.columns(3)
    ph_m = s1.number_input("pH", value=7.0)
    awc_m = s2.number_input("Capacidad de agua (AWC)", value=0.15)
    prof_m = s3.number_input("Profundidad efectiva (cm)", value=100.0)
    s4, s5, s6 = st.columns(3)
    clay_m = s4.number_input("Arcilla (%)", value=25.0)
    sand_m = s5.number_input("Arena (%)", value=40.0)
    silt_m = s6.number_input("Limo (%)", value=35.0)
    db_m = st.number_input("Densidad aparente", value=1.4)

    st.markdown("##### Clima futuro estimado")
    st.caption("Introduce la temperatura y lluvia que esperas para 2030 y 2040.")
    fa, fb = st.columns(2)
    with fa:
        st.markdown("**2030**")
        tmax_30 = st.number_input("Temp. máx 2030", value=25.0)
        tmin_30 = st.number_input("Temp. mín 2030", value=11.0)
        ppt_30 = st.number_input("Lluvia 2030", value=340.0)
    with fb:
        st.markdown("**2040**")
        tmax_40 = st.number_input("Temp. máx 2040", value=27.0)
        tmin_40 = st.number_input("Temp. mín 2040", value=12.0)
        ppt_40 = st.number_input("Lluvia 2040", value=350.0)

    if st.button("Calcular recomendación", type="primary"):
        punto_manual = {
            "tmax": tmax_m, "tmin": tmin_m, "ppt": ppt_m, "vpdmean": vpd_m,
            "ph1to1h2o_r": ph_m, "awc_r": awc_m, "profundidad_efectiva_cm": prof_m,
            "claytotal_r": clay_m, "sandtotal_r": sand_m, "silttotal_r": silt_m,
            "dbthirdbar_r": db_m,
            "tmax_245_2030": tmax_30, "tmin_245_2030": tmin_30, "pr_245_2030": ppt_30,
            "tmax_245_2040": tmax_40, "tmin_245_2040": tmin_40, "pr_245_2040": ppt_40,
            "tmax_585_2030": tmax_30, "tmin_585_2030": tmin_30, "pr_585_2030": ppt_30,
            "tmax_585_2040": tmax_40, "tmin_585_2040": tmin_40, "pr_585_2040": ppt_40,
        }
        resultado = motor.predecir_punto(punto_manual)
        mostrar_resultados(resultado)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 4 — ACERCA DE
# ══════════════════════════════════════════════════════════════════
with tab4:
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

    **Fuentes de datos**

    - **Clima actual:** PRISM (resolución de 4 km).
    - **Clima futuro:** NASA NEX-GDDP-CMIP6.
    - **Suelo:** gSSURGO (USDA).

    **Limitaciones**

    Los datos de suelo cubren principalmente las zonas agrícolas de California;
    en áreas de montaña, desierto o tierras federales puede no haber información.
    Los resultados son orientativos y de carácter académico, no sustituyen el
    criterio técnico agronómico.
    """)
