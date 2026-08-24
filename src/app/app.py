"""
================================================================================
 APP — RECOMENDADOR DE CULTIVOS PARA CALIFORNIA
================================================================================
Aplicación Streamlit para agricultores. Tres pestañas:
  1. Mi lote        — el productor ubica su parcela y ve los cultivos más aptos
                      hoy y en el futuro, con dashboards de clima y aptitud.
  2. Mapas por cultivo — mapas coropléticos del IAI, filtrables por cultivo y año.
  3. Datos manuales — el productor introduce sus propios datos de suelo y clima.

Usa el motor predictor.py (en vivo) y el IAI precalculado para los mapas.
================================================================================
"""

import os
import numpy as np
import polars as pl
import pandas as pd
import streamlit as st
from scipy.spatial import cKDTree

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

# ─── Paleta (verde/teal, sin negro de fondo) ───
COLOR_FONDO = "#F4F9F4"
COLOR_PRIMARIO = "#0F6E56"
COLOR_ACENTO = "#1D9E75"

st.set_page_config(page_title="Recomendador de cultivos", layout="wide")

# ─── Estilos ───
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_FONDO}; }}

    /* Pestañas: fondo con color y texto oscuro legible */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background-color: #E4EFE4;
        padding: 4px;
        border-radius: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 20px;
        color: {COLOR_PRIMARIO};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: white;
        color: {COLOR_PRIMARIO};
        font-weight: 600;
    }}

    .bloque {{ background: white; border-radius: 12px; padding: 16px;
               border: 1px solid #DDE8DD; margin-bottom: 12px; }}
    .titulo-zona {{ color: {COLOR_PRIMARIO}; font-weight: 600; font-size: 18px; }}

    /* Encabezados de sección en verde */
    h4 {{ color: {COLOR_PRIMARIO} !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  CARGA DE DATOS Y MOTOR (en caché para no recargar)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def cargar_motor():
    return PredictorCultivos()

@st.cache_data
def cargar_base():
    base = pl.read_parquet(RUTA_BASE).filter(pl.col("datos_completos") == True)
    return base

@st.cache_data
def cargar_iai_mapas():
    return pl.read_parquet(RUTA_IAI_MAPAS)

@st.cache_resource
def construir_kdtree(_base):
    coords = _base.select(["lon", "lat"]).to_numpy()
    return cKDTree(coords)

motor = cargar_motor()
base = cargar_base()
iai_mapas = cargar_iai_mapas()
tree = construir_kdtree(base)


# ─── Utilidad: buscar el punto más cercano en la base ───
def punto_mas_cercano(lon, lat):
    dist, idx = tree.query([lon, lat])
    fila = base.row(idx, named=True)
    return fila, dist


# ══════════════════════════════════════════════════════════════════
#  CABECERA
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="border-left:4px solid {COLOR_ACENTO}; padding-left:14px; margin-bottom:16px;">
    <div style="font-size:24px; font-weight:600; color:{COLOR_PRIMARIO};">Recomendador de cultivos</div>
    <div style="font-size:14px; color:#5F7A5F;">California · idoneidad agrícola presente y futura</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Mi lote", "Mapas por cultivo", "Datos manuales"])


# ══════════════════════════════════════════════════════════════════
#  FUNCIÓN COMÚN: mostrar resultados de un punto
# ══════════════════════════════════════════════════════════════════
def mostrar_resultados(resultado, punto):
    if "error" in resultado:
        st.warning(resultado["error"])
        return

    # Zona / región
    st.markdown(f"<div class='bloque'><span class='titulo-zona'>Tu zona: "
                f"{resultado['etiqueta']}</span></div>", unsafe_allow_html=True)

    # Top 3 cultivos hoy
    ranking_hoy = motor.ranking(resultado["iai_presente"], top=3)
    st.markdown("#### Los 3 cultivos más aptos hoy")
    cols = st.columns(3)
    for i, (nombre, iai) in enumerate(ranking_hoy):
        with cols[i]:
            st.markdown(f"""
            <div class='bloque' style='text-align:center; border-top:3px solid {COLOR_ACENTO};'>
              <div style='font-size:13px; color:#5F7A5F; font-weight:500;'>Opción {i+1}</div>
              <div style='font-size:19px; font-weight:600; color:{COLOR_PRIMARIO}; margin:4px 0;'>{nombre}</div>
              <div style='font-size:26px; font-weight:600; color:{COLOR_ACENTO};'>{iai:.2f}</div>
              <div style='font-size:12px; color:#5F7A5F;'>idoneidad</div>
            </div>
            """, unsafe_allow_html=True)

    # Dashboard: cambio de aptitud del cultivo top
    st.markdown("#### Cómo cambia la aptitud con el clima futuro")
    top_crop_nombre = ranking_hoy[0][0]
    top_crop_id = [k for k, v in CROP_DICT_ES.items() if v == top_crop_nombre][0]
    iai_hoy = resultado["iai_presente"][top_crop_id]
    iai_mod = resultado["iai_futuro"]["245"][top_crop_id]
    iai_sev = resultado["iai_futuro"]["585"][top_crop_id]

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{top_crop_nombre} · hoy", f"{iai_hoy:.2f}")
    c2.metric(f"Moderado (SSP2-4.5)", f"{iai_mod:.2f}", f"{iai_mod-iai_hoy:+.2f}")
    c3.metric(f"Severo (SSP5-8.5)", f"{iai_sev:.2f}", f"{iai_sev-iai_hoy:+.2f}")

    # Dashboard: cambio de clima
    st.markdown("#### Cómo cambia el clima en tu zona")
    tmax_hoy = punto["tmax"]
    tmax_fut = np.mean([punto["tmax_585_2030"], punto["tmax_585_2040"]])
    tmin_hoy = punto["tmin"]
    tmin_fut = np.mean([punto["tmin_585_2030"], punto["tmin_585_2040"]])
    ppt_hoy = punto["ppt"]
    ppt_fut = np.mean([punto["pr_585_2030"], punto["pr_585_2040"]])

    d1, d2, d3 = st.columns(3)
    d1.metric("Temp. máxima", f"{tmax_hoy:.0f}°C", f"{tmax_fut-tmax_hoy:+.1f}°C")
    d2.metric("Temp. mínima", f"{tmin_hoy:.0f}°C", f"{tmin_fut-tmin_hoy:+.1f}°C")
    d3.metric("Lluvia anual", f"{ppt_hoy:.0f} mm", f"{ppt_fut-ppt_hoy:+.0f} mm")

    # Tabla completa (los 12 cultivos, presente y futuro)
    with st.expander("Ver los 12 cultivos en detalle"):
        filas = []
        for cid in FINAL_CROPS:
            filas.append({
                "Cultivo": CROP_DICT_ES[cid],
                "Hoy": round(resultado["iai_presente"][cid], 2),
                "Moderado": round(resultado["iai_futuro"]["245"][cid], 2),
                "Severo": round(resultado["iai_futuro"]["585"][cid], 2),
            })
        df_tabla = pd.DataFrame(filas).sort_values("Hoy", ascending=False)
        st.dataframe(df_tabla, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — MI LOTE
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("Ubica tu parcela en el mapa, escribe una dirección o introduce las coordenadas.")

    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        # Entrada por dirección
        direccion = st.text_input("Buscar dirección o ciudad", placeholder="Fresno, California")
        # Entrada por coordenadas
        cc1, cc2 = st.columns(2)
        lon_in = cc1.number_input("Longitud", value=-119.8, format="%.4f")
        lat_in = cc2.number_input("Latitud", value=36.7, format="%.4f")

        # Geocodificar dirección si se escribió
        lon_sel, lat_sel = lon_in, lat_in
        if direccion:
            try:
                from geopy.geocoders import Nominatim
                geo = Nominatim(user_agent="recomendador_cultivos_tfm")
                loc = geo.geocode(f"{direccion}, California, USA", timeout=10)
                if loc:
                    lon_sel, lat_sel = loc.longitude, loc.latitude
                    st.success(f"Ubicación encontrada: {lon_sel:.4f}, {lat_sel:.4f}")
                else:
                    st.warning("No se encontró la dirección. Usa las coordenadas.")
            except Exception as e:
                st.warning(f"No se pudo buscar la dirección: {e}")

        # Mapa con folium
        m = folium.Map(location=[lat_sel, lon_sel], zoom_start=7, tiles="CartoDB positron")
        folium.Marker([lat_sel, lon_sel], tooltip="Tu lote",
                      icon=folium.Icon(color="green", icon="leaf")).add_to(m)
        map_data = st_folium(m, height=300, width=None, key="mapa_lote")

        # Si el usuario hace clic en el mapa, actualizar coordenadas
        if map_data and map_data.get("last_clicked"):
            lat_sel = map_data["last_clicked"]["lat"]
            lon_sel = map_data["last_clicked"]["lng"]
            st.info(f"Punto seleccionado: {lon_sel:.4f}, {lat_sel:.4f}")

    with col_der:
        # Buscar el punto más cercano y predecir
        fila, dist = punto_mas_cercano(lon_sel, lat_sel)

        # Control de cobertura: si está lejos, avisar
        if dist > 0.15:  # ~15 km
            st.warning(
                f"⚠️ La ubicación seleccionada está fuera de la zona con datos "
                f"(a {dist*111:.0f} km del punto más cercano con información). "
                f"La predicción es orientativa. Considera usar la pestaña "
                f"'Datos manuales' si conoces las condiciones de tu parcela."
            )

        resultado = motor.predecir_punto(fila)
        mostrar_resultados(resultado, fila)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 2 — MAPAS POR CULTIVO (coropléticos)
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("Explora la idoneidad de cada cultivo en toda California, hoy y en el futuro.")

    fc1, fc2 = st.columns(2)
    cultivo_sel = fc1.selectbox("Cultivo", sorted(CROP_DICT_ES.values()))
    escenario_sel = fc2.selectbox("Escenario", ["presente", "SSP2-4.5", "SSP5-8.5"],
                                  format_func=lambda x: {
                                      "presente": "Hoy (2025)",
                                      "SSP2-4.5": "Futuro moderado (SSP2-4.5)",
                                      "SSP5-8.5": "Futuro severo (SSP5-8.5)",
                                  }[x])

    # Filtrar el IAI precalculado
    datos_mapa = iai_mapas.filter(
        (pl.col("cultivo") == cultivo_sel) & (pl.col("escenario") == escenario_sel)
    ).select(["lon", "lat", "iai"]).to_pandas()

    st.markdown(f"**{cultivo_sel}** — idoneidad media: "
                f"{datos_mapa['iai'].mean():.2f} · "
                f"{len(datos_mapa):,} puntos")

    # ── Color por IAI: de rojo (baja) a verde (alta) ──
    def iai_a_color(v):
        # v en [0,1]. Interpolar rojo -> amarillo -> verde.
        if v < 0.5:
            t = v / 0.5
            r = int(216 + (250-216)*t)
            g = int(90 + (199-90)*t)
            b = int(48 + (117-48)*t)
        else:
            t = (v - 0.5) / 0.5
            r = int(250 + (15-250)*t)
            g = int(199 + (110-199)*t)
            b = int(117 + (86-117)*t)
        return [r, g, b]

    datos_mapa[["r", "g", "b"]] = datos_mapa["iai"].apply(
        lambda v: pd.Series(iai_a_color(v)))

    # ── Mapa con pydeck (renderiza miles de puntos de forma nativa) ──
    import pydeck as pdk

    capa = pdk.Layer(
        "ScatterplotLayer",
        data=datos_mapa,
        get_position=["lon", "lat"],
        get_fill_color=["r", "g", "b", 180],
        get_radius=2500,          # radio en metros
        pickable=True,
    )

    vista = pdk.ViewState(latitude=37.2, longitude=-119.5, zoom=5.2)

    st.pydeck_chart(pdk.Deck(
        layers=[capa],
        initial_view_state=vista,
        map_style="light",
        tooltip={"text": "IAI: {iai}"},
    ))

    # Leyenda de color
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

    st.markdown("##### Clima presente")
    m1, m2c, m3 = st.columns(3)
    tmax_m = m1.number_input("Temp. máxima (°C)", value=24.0)
    tmin_m = m2c.number_input("Temp. mínima (°C)", value=10.0)
    ppt_m = m3.number_input("Lluvia anual (mm)", value=330.0)
    vpd_m = st.number_input("VPD medio (opcional, para la zona)", value=15.0)

    st.markdown("##### Suelo")
    s1, s2, s3 = st.columns(3)
    ph_m = s1.number_input("pH", value=7.0)
    awc_m = s2.number_input("Capacidad de agua (AWC)", value=0.15)
    prof_m = s3.number_input("Profundidad efectiva (cm)", value=100.0)
    s4, s5, s6 = st.columns(3)
    clay_m = s4.number_input("Arcilla (%)", value=25.0)
    sand_m = s5.number_input("Arena (%)", value=40.0)
    silt_m = s6.number_input("Limo (%)", value=35.0)
    db_m = st.number_input("Densidad aparente (dbthirdbar)", value=1.4)

    st.markdown("##### Clima futuro (severo, para estimar el cambio)")
    f1, f2, f3 = st.columns(3)
    tmax_f = f1.number_input("Temp. máxima futura (°C)", value=27.0)
    tmin_f = f2.number_input("Temp. mínima futura (°C)", value=12.0)
    ppt_f = f3.number_input("Lluvia anual futura (mm)", value=394.0)

    if st.button("Calcular recomendación", type="primary"):
        # Construir un punto con los datos manuales
        punto_manual = {
            "tmax": tmax_m, "tmin": tmin_m, "ppt": ppt_m, "vpdmean": vpd_m,
            "ph1to1h2o_r": ph_m, "awc_r": awc_m, "profundidad_efectiva_cm": prof_m,
            "claytotal_r": clay_m, "sandtotal_r": sand_m, "silttotal_r": silt_m,
            "dbthirdbar_r": db_m,
            # Futuro: usamos el mismo valor para 2030 y 2040 (el usuario da uno)
            "tmax_245_2030": tmax_f, "tmax_245_2040": tmax_f,
            "tmin_245_2030": tmin_f, "tmin_245_2040": tmin_f,
            "pr_245_2030": ppt_f, "pr_245_2040": ppt_f,
            "tmax_585_2030": tmax_f, "tmax_585_2040": tmax_f,
            "tmin_585_2030": tmin_f, "tmin_585_2040": tmin_f,
            "pr_585_2030": ppt_f, "pr_585_2040": ppt_f,
        }
        resultado = motor.predecir_punto(punto_manual)
        mostrar_resultados(resultado, punto_manual)
