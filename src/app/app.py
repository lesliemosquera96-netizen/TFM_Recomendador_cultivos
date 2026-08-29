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
  4. Acerca de         — metodología, autoría y fuentes del proyecto
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

# ─── Paleta ───
COLOR_FONDO = "#F4F9F4"
COLOR_PRIMARIO = "#0F6E56"
COLOR_ACENTO = "#1D9E75"
COLOR_MOD = "#1D9E75"   # escenario moderado (SSP2-4.5)
COLOR_SEV = "#E8833A"   # escenario severo (SSP5-8.5)
COLOR_TEXTO = "#3A5A3A"
ESCALA_IAI = [[0.0, "#D85A30"], [0.5, "#FAC775"], [1.0, "#0F6E56"]]

LOGO_PATH = os.path.join(BASE_DIR, "logo.png")

def logo_b64():
    """Devuelve el logo en base64 si existe junto a app.py; si no, None."""
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None

st.set_page_config(
    page_title="Recomendador de cultivos · California",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
#  ESTILOS
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_FONDO}; }}
    .block-container {{ padding-top: 1.4rem; max-width: 1250px; }}

    /* Ocultar la barra lateral por completo */
    section[data-testid="stSidebar"] {{ display: none; }}
    div[data-testid="collapsedControl"] {{ display: none; }}

    /* Cabecera */
    .cab-titulo {{ font-size: 28px; font-weight: 700; color: {COLOR_PRIMARIO}; line-height: 1.1; }}
    .cab-desc  {{ font-size: 15px; color: #5F7A5F; margin-top: 5px; }}

    /* Pestañas en verde oscuro, sin emojis */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px; background: transparent; border-bottom: 2px solid #DDE8DD;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0; padding: 10px 24px;
        color: {COLOR_PRIMARIO}; font-weight: 600; font-size: 15px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {COLOR_PRIMARIO}; color: white; font-weight: 700;
    }}

    /* Tarjetas */
    .bloque {{ background: white; border-radius: 14px; padding: 18px;
               border: 1px solid #DDE8DD; margin-bottom: 14px;
               box-shadow: 0 2px 8px rgba(15,110,86,0.05); }}
    .titulo-zona {{ color: {COLOR_PRIMARIO}; font-weight: 600; font-size: 18px; }}
    .card-top {{ background: white; border-radius: 14px; padding: 16px 12px;
                 border: 1px solid #DDE8DD; text-align: center;
                 border-top: 4px solid {COLOR_ACENTO};
                 box-shadow: 0 2px 8px rgba(15,110,86,0.06); }}
    .seccion {{ color: {COLOR_PRIMARIO}; font-weight: 600; font-size: 18px;
                margin: 8px 0 6px 0; }}
    h4 {{ color: {COLOR_PRIMARIO} !important; }}

    /* Métricas */
    div[data-testid="stMetric"] {{
        background: white; border: 1px solid #DDE8DD; border-radius: 12px;
        padding: 12px 16px; box-shadow: 0 2px 8px rgba(15,110,86,0.05);
    }}
    div[data-testid="stMetricValue"] {{ color: {COLOR_PRIMARIO}; }}

    /* Pie de página */
    .pie {{ text-align:center; color:#7A8F7A; font-size:12px;
            margin-top: 26px; padding-top: 12px; border-top: 1px solid #DDE8DD; }}
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


def _estilo(fig, alto=320):
    """Aplica un estilo limpio y consistente a las figuras de Plotly."""
    fig.update_layout(
        height=alto,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Source Sans Pro, sans-serif", size=13, color=COLOR_TEXTO),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, title=""),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#DDE8DD")
    fig.update_yaxes(showgrid=True, gridcolor="#EDF3ED", zeroline=False)
    return fig


# ══════════════════════════════════════════════════════════════════
#  CABECERA
# ══════════════════════════════════════════════════════════════════
_b64 = logo_b64()
_logo_img = (f"<img src='data:image/png;base64,{_b64}' style='height:74px;'>"
             if _b64 else "")
st.markdown(f"""
<div style="display:flex; align-items:center; gap:16px; margin:4px 0 20px 0;">
    {_logo_img}
    <div>
        <div class="cab-titulo">Recomendador de cultivos</div>
        <div class="cab-desc">Descubre qué cultivos se adaptan mejor a tu parcela en
            California, hoy y ante el cambio climático.</div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Mi lote", "Información por cultivo", "Introduce tus datos", "Acerca de"])


# ══════════════════════════════════════════════════════════════════
#  BLOQUES DE RESULTADO (reutilizables en Mi lote y Datos manuales)
# ══════════════════════════════════════════════════════════════════
def render_zona(resultado):
    st.markdown(f"<div class='bloque'><span class='titulo-zona'>Tu zona: "
                f"{resultado['etiqueta']}</span></div>", unsafe_allow_html=True)


def render_top3(resultado):
    iai = resultado["iai"]
    ranking_hoy = motor.ranking(iai["2025"], top=3)
    st.markdown("<div class='seccion'>Los 3 cultivos más aptos hoy</div>",
                unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (nombre, val) in enumerate(ranking_hoy):
        with cols[i]:
            st.markdown(f"""
            <div class='card-top'>
              <div style='font-size:13px; color:#5F7A5F; font-weight:500;'>Opción {i+1}</div>
              <div style='font-size:20px; font-weight:700; color:{COLOR_PRIMARIO}; margin:6px 0;'>{nombre}</div>
              <div style='font-size:30px; font-weight:700; color:{COLOR_ACENTO};'>{val:.2f}</div>
              <div style='font-size:12px; color:#5F7A5F;'>idoneidad (0–1)</div>
            </div>
            """, unsafe_allow_html=True)
    return ranking_hoy


def render_barras(resultado):
    iai = resultado["iai"]
    df_barras = pd.DataFrame({
        "Cultivo": [CROP_DICT_ES[c] for c in FINAL_CROPS],
        "Idoneidad": [iai["2025"][c] for c in FINAL_CROPS],
    }).sort_values("Idoneidad", ascending=True)

    st.markdown("<div class='seccion'>Idoneidad de todos los cultivos hoy</div>",
                unsafe_allow_html=True)
    fig = px.bar(
        df_barras, x="Idoneidad", y="Cultivo", orientation="h",
        color="Idoneidad", color_continuous_scale=ESCALA_IAI, range_color=[0, 1],
        text=df_barras["Idoneidad"].map(lambda v: f"{v:.2f}"),
    )
    fig.update_traces(textposition="outside", textfont_size=11,
                      cliponaxis=False, hovertemplate="%{y}: %{x:.2f}<extra></extra>")
    fig.update_xaxes(range=[0, 1.05], title="Índice de Idoneidad Agrícola (IAI)")
    fig.update_yaxes(title="")
    fig.update_coloraxes(showscale=False)
    _estilo(fig, alto=430)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_evolucion(resultado, ranking_hoy):
    iai = resultado["iai"]
    st.markdown("<div class='seccion'>Cómo cambia la idoneidad de tu mejor cultivo</div>",
                unsafe_allow_html=True)
    top_id = [k for k, v in CROP_DICT_ES.items() if v == ranking_hoy[0][0]][0]
    anios = [2025, 2030, 2040]
    mod = [iai["2025"][top_id], iai["2030-245"][top_id], iai["2040-245"][top_id]]
    sev = [iai["2025"][top_id], iai["2030-585"][top_id], iai["2040-585"][top_id]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anios, y=mod, name="Moderado (SSP2-4.5)",
                             mode="lines+markers", line=dict(color=COLOR_MOD, width=3),
                             marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=anios, y=sev, name="Severo (SSP5-8.5)",
                             mode="lines+markers", line=dict(color=COLOR_SEV, width=3),
                             marker=dict(size=9)))
    fig.update_xaxes(tickmode="array", tickvals=anios)
    fig.update_yaxes(range=[0, 1], title="Idoneidad (IAI)")
    _estilo(fig, alto=320)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_clima(resultado):
    clima = resultado["clima"]
    anios = [2025, 2030, 2040]
    st.markdown("<div class='seccion'>Cómo cambia el clima en tu zona</div>",
                unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)

    with cc1:
        st.caption("Temperatura máxima media (°C)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anios, y=[clima["2025"]["tmax"], clima["2030-245"]["tmax"], clima["2040-245"]["tmax"]],
            name="Moderado", mode="lines+markers", line=dict(color=COLOR_MOD, width=3)))
        fig.add_trace(go.Scatter(
            x=anios, y=[clima["2025"]["tmax"], clima["2030-585"]["tmax"], clima["2040-585"]["tmax"]],
            name="Severo", mode="lines+markers", line=dict(color=COLOR_SEV, width=3)))
        fig.update_xaxes(tickmode="array", tickvals=anios)
        fig.update_yaxes(title="°C")
        _estilo(fig, alto=290)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with cc2:
        st.caption("Lluvia anual (mm)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=anios, y=[clima["2025"]["ppt"], clima["2030-245"]["ppt"], clima["2040-245"]["ppt"]],
            name="Moderado", mode="lines+markers", line=dict(color=COLOR_MOD, width=3)))
        fig.add_trace(go.Scatter(
            x=anios, y=[clima["2025"]["ppt"], clima["2030-585"]["ppt"], clima["2040-585"]["ppt"]],
            name="Severo", mode="lines+markers", line=dict(color=COLOR_SEV, width=3)))
        fig.update_xaxes(tickmode="array", tickvals=anios)
        fig.update_yaxes(title="mm")
        _estilo(fig, alto=290)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_tabla(resultado):
    iai = resultado["iai"]
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


def render_resultados_completo(resultado):
    """Muestra todos los bloques en vertical (usado en Datos manuales)."""
    if "error" in resultado:
        st.warning(resultado["error"])
        return
    render_zona(resultado)
    ranking_hoy = render_top3(resultado)
    render_barras(resultado)
    render_evolucion(resultado, ranking_hoy)
    render_clima(resultado)
    render_tabla(resultado)


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 1 — MI LOTE
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("Ubica tu parcela: escribe una dirección, introduce las coordenadas "
                "o **haz clic en un punto del mapa**.")

    # Punto seleccionado como fuente única de verdad
    if "sel_lon" not in st.session_state:
        st.session_state.sel_lon = -119.8
        st.session_state.sel_lat = 36.7
        st.session_state.num_lon = -119.8
        st.session_state.num_lat = 36.7

    def _sync_num():
        st.session_state.sel_lon = st.session_state.num_lon
        st.session_state.sel_lat = st.session_state.num_lat

    col_izq, col_der = st.columns([1.15, 1])

    with col_izq:
        direccion = st.text_input("Buscar dirección o ciudad",
                                  placeholder="Fresno, California")
        if direccion:
            try:
                from geopy.geocoders import Nominatim
                geo = Nominatim(user_agent="recomendador_cultivos_tfm")
                loc = geo.geocode(f"{direccion}, California, USA", timeout=10)
                if loc:
                    st.session_state.sel_lon = loc.longitude
                    st.session_state.sel_lat = loc.latitude
                    st.session_state.num_lon = round(loc.longitude, 4)
                    st.session_state.num_lat = round(loc.latitude, 4)
                    st.success(f"Ubicación encontrada: {loc.longitude:.4f}, {loc.latitude:.4f}")
                else:
                    st.warning("No se encontró la dirección. Usa las coordenadas o el mapa.")
            except Exception as e:
                st.warning(f"No se pudo buscar la dirección: {e}")

        c1, c2 = st.columns(2)
        c1.number_input("Longitud", key="num_lon", format="%.4f", on_change=_sync_num)
        c2.number_input("Latitud", key="num_lat", format="%.4f", on_change=_sync_num)

        lon_sel = st.session_state.sel_lon
        lat_sel = st.session_state.sel_lat

        # Mapa: OpenStreetMap (no requiere clave) + marcador en el punto activo
        m = folium.Map(location=[lat_sel, lon_sel], zoom_start=7,
                       tiles="OpenStreetMap", control_scale=True)
        folium.Marker([lat_sel, lon_sel], tooltip="Tu lote",
                      icon=folium.Icon(color="green", icon="leaf", prefix="fa")).add_to(m)
        map_data = st_folium(m, height=330, width=None, key="mapa_lote",
                             returned_objects=["last_clicked"])

        if map_data and map_data.get("last_clicked"):
            clat = map_data["last_clicked"]["lat"]
            clon = map_data["last_clicked"]["lng"]
            if (round(clon, 4) != round(st.session_state.sel_lon, 4) or
                    round(clat, 4) != round(st.session_state.sel_lat, 4)):
                st.session_state.sel_lon = clon
                st.session_state.sel_lat = clat
                st.session_state.num_lon = round(clon, 4)
                st.session_state.num_lat = round(clat, 4)
                st.rerun()

        st.caption(f"Punto seleccionado: {lon_sel:.4f}, {lat_sel:.4f}")

    # Punto más cercano con datos + predicción
    fila, dist = punto_mas_cercano(lon_sel, lat_sel)
    resultado = motor.predecir_punto(fila)

    with col_der:
        if "error" in resultado:
            st.warning(resultado["error"])
        else:
            render_zona(resultado)

        st.markdown("<div class='bloque'>"
                    "<div style='font-weight:600; color:%s; margin-bottom:6px;'>"
                    "Condiciones de tu parcela</div>"
                    "<div style='font-size:13px; color:%s; line-height:1.7;'>"
                    "Temp. máx: <b>%.0f°C</b> · Temp. mín: <b>%.0f°C</b> · Lluvia: <b>%.0f mm</b><br>"
                    "pH: <b>%.1f</b> · Profundidad: <b>%.0f cm</b> · Arcilla: <b>%.0f%%</b>"
                    "</div></div>" % (
                        COLOR_PRIMARIO, COLOR_TEXTO,
                        fila['tmax'], fila['tmin'], fila['ppt'],
                        fila['ph1to1h2o_r'], fila['profundidad_efectiva_cm'], fila['claytotal_r']),
                    unsafe_allow_html=True)

        if dist > 0.15:
            st.warning(
                f"La ubicación está fuera de la zona con datos "
                f"(a {dist*111:.0f} km del punto más cercano con información). "
                f"La predicción es orientativa; puedes usar 'Datos manuales' "
                f"si conoces las condiciones de tu parcela."
            )

    # Resultados a lo ancho, bien distribuidos
    if "error" not in resultado:
        st.divider()
        ranking_hoy = render_top3(resultado)
        st.write("")
        render_barras(resultado)
        render_evolucion(resultado, ranking_hoy)
        render_clima(resultado)
        render_tabla(resultado)


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
    if len(datos_mapa):
        e1.metric("Idoneidad media", f"{datos_mapa['iai'].mean():.2f}")
        e2.metric("Zonas muy aptas (>0.7)", f"{(datos_mapa['iai']>0.7).mean()*100:.0f}%")
        e3.metric("Puntos evaluados", f"{len(datos_mapa):,}")

    if len(datos_mapa) == 0:
        st.warning("No hay datos para esta combinación de cultivo y escenario.")
        st.stop()

    colores = datos_mapa["iai"].apply(color_iai)
    datos_mapa["r"] = colores.apply(lambda c: c[0])
    datos_mapa["g"] = colores.apply(lambda c: c[1])
    datos_mapa["b"] = colores.apply(lambda c: c[2])

    capa = pdk.Layer(
        "ScatterplotLayer", data=datos_mapa,
        get_position=["lon", "lat"], get_fill_color=["r", "g", "b", 180],
        get_radius=2500, pickable=True,
    )
    vista = pdk.ViewState(latitude=37.2, longitude=-119.5, zoom=5.2)
    st.pydeck_chart(pdk.Deck(layers=[capa], initial_view_state=vista,
                             map_style="light", tooltip={"text": "IAI: {iai}"}))

    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#5F7A5F;
                margin-top:6px;">
      <span>Baja idoneidad</span>
      <div style="flex:1; height:12px; border-radius:6px; max-width:340px;
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
        st.divider()
        render_resultados_completo(motor.predecir_punto(punto_manual))


# ══════════════════════════════════════════════════════════════════
#  PESTAÑA 4 — ACERCA DE
# ══════════════════════════════════════════════════════════════════
with tab4:
    a1, a2 = st.columns([1.6, 1])

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

        **Limitaciones**

        Los datos de suelo cubren principalmente las zonas agrícolas de California;
        en áreas de montaña, desierto o tierras federales puede no haber información.
        Los resultados son orientativos y de carácter académico, y no sustituyen el
        criterio técnico agronómico.
        """)

    with a2:
        st.markdown(f"""
        <div class='bloque'>
          <div style='font-weight:600; color:{COLOR_PRIMARIO}; font-size:16px;'>Ficha del proyecto</div>
          <div style='font-size:14px; color:{COLOR_TEXTO}; line-height:1.9; margin-top:8px;'>
            <b>Autora</b><br>Leslie Estefany Mosquera<br><br>
            <b>Tutor</b><br>José Lloreda Sánchez<br><br>
            <b>Institución</b><br>La Salle – Universitat Ramon Llull<br>
            Máster en Ciencia de Datos
          </div>
        </div>
        <div class='bloque'>
          <div style='font-weight:600; color:{COLOR_PRIMARIO}; font-size:16px;'>Fuentes de datos</div>
          <div style='font-size:14px; color:{COLOR_TEXTO}; line-height:1.9; margin-top:8px;'>
            <b>Clima actual:</b> PRISM (4 km)<br>
            <b>Clima futuro:</b> NASA NEX-GDDP-CMIP6<br>
            <b>Suelo:</b> gSSURGO (USDA)
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PIE
# ══════════════════════════════════════════════════════════════════
st.markdown(
    "<div class='pie'>Recomendador de cultivos · TFM Máster en Ciencia de Datos · "
    "Leslie Estefany Mosquera · Tutor: José Lloreda Sánchez · "
    "La Salle – Universitat Ramon Llull · Resultados orientativos de carácter académico</div>",
    unsafe_allow_html=True)
