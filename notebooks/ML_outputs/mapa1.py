# -*- coding: utf-8 -*-
"""
Mapa coroplético de la producción agrícola de EE.UU. por estado (2024).
Escala de verdes: cuanto más verde, mayor producción.
Cada estado muestra su nombre y su valor. Los estados pequeños del
noreste se etiquetan por fuera con una flecha (leader line).

Requisitos:
    pip install plotly pandas kaleido
"""

import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# 1) DATOS  (valor 2024 en miles de millones de USD -> "value" numérico)
# ---------------------------------------------------------------------------
data = [
    # abrev, nombre,           valor_texto, valor_num (USD miles de millones)
    ("CA", "California",      "$67.4B", 67.4),
    ("IA", "Iowa",           "$40.5B", 40.5),
    ("TX", "Texas",          "$37.6B", 37.6),
    ("NE", "Nebraska",       "$34.2B", 34.2),
    ("KS", "Kansas",         "$26.3B", 26.3),
    ("IL", "Illinois",       "$24.4B", 24.4),
    ("MN", "Minnesota",      "$24.1B", 24.1),
    ("WI", "Wisconsin",      "$16.9B", 16.9),
    ("NC", "North Carolina", "$16.8B", 16.8),
    ("IN", "Indiana",        "$16.3B", 16.3),
    ("MO", "Missouri",       "$15.4B", 15.4),
    ("SD", "South Dakota",   "$15.1B", 15.1),
    ("OH", "Ohio",           "$14.6B", 14.6),
    ("GA", "Georgia",        "$14.2B", 14.2),
    ("WA", "Washington",     "$14.2B", 14.2),
    ("AR", "Arkansas",       "$13.5B", 13.5),
    ("ND", "North Dakota",   "$13.0B", 13.0),
    ("ID", "Idaho",          "$12.6B", 12.6),
    ("MI", "Michigan",       "$12.2B", 12.2),
    ("FL", "Florida",        "$11.2B", 11.2),
    ("CO", "Colorado",       "$10.8B", 10.8),
    ("OK", "Oklahoma",       "$10.7B", 10.7),
    ("PA", "Pennsylvania",   "$10.7B", 10.7),
    ("KY", "Kentucky",       "$9.2B",  9.2),
    ("AL", "Alabama",        "$9.0B",  9.0),
    ("NY", "New York",       "$8.2B",  8.2),
    ("OR", "Oregon",         "$8.1B",  8.1),
    ("MS", "Mississippi",    "$7.9B",  7.9),
    ("TN", "Tennessee",      "$6.1B",  6.1),
    ("MT", "Montana",        "$6.0B",  6.0),
    ("AZ", "Arizona",        "$5.6B",  5.6),
    ("VA", "Virginia",       "$5.6B",  5.6),
    ("LA", "Louisiana",      "$5.0B",  5.0),
    ("NM", "New Mexico",     "$4.7B",  4.7),
    ("SC", "South Carolina", "$4.2B",  4.2),
    ("MD", "Maryland",       "$3.6B",  3.6),
    ("UT", "Utah",           "$3.0B",  3.0),
    ("WY", "Wyoming",        "$2.4B",  2.4),
    ("DE", "Delaware",       "$2.3B",  2.3),
    ("NJ", "New Jersey",     "$2.0B",  2.0),
    ("NV", "Nevada",         "$1.3B",  1.3),
    ("WV", "West Virginia",  "$1.2B",  1.2),
    ("VT", "Vermont",        "$1.1B",  1.1),
    ("ME", "Maine",          "$1.1B",  1.1),
    ("HI", "Hawaii",         "$917M",  0.917),
    ("CT", "Connecticut",    "$909M",  0.909),
    ("MA", "Massachusetts",  "$782M",  0.782),
    ("NH", "New Hampshire",  "$340M",  0.340),
    ("RI", "Rhode Island",   "$139M",  0.139),
    ("AK", "Alaska",         "$71M",   0.071),
]
df = pd.DataFrame(data, columns=["abrev", "estado", "valor_txt", "valor_num"])

# ---------------------------------------------------------------------------
# 2) CENTROIDES APROXIMADOS DE CADA ESTADO (lat, lon) para colocar etiquetas
# ---------------------------------------------------------------------------
centroids = {
    "AL": (32.8, -86.8),  "AK": (63.6, -152.0), "AZ": (34.3, -111.7),
    "AR": (34.9, -92.4),  "CA": (37.2, -119.4), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7),  "DE": (39.0, -75.5),  "FL": (28.6, -81.5),
    "GA": (32.6, -83.4),  "HI": (20.3, -156.4), "ID": (44.4, -114.6),
    "IL": (40.0, -89.2),  "IN": (39.9, -86.3),  "IA": (42.0, -93.5),
    "KS": (38.5, -98.4),  "KY": (37.5, -85.3),  "LA": (31.0, -92.0),
    "ME": (45.4, -69.2),  "MD": (39.0, -76.8),  "MA": (42.3, -71.8),
    "MI": (44.3, -85.4),  "MN": (46.3, -94.3),  "MS": (32.7, -89.7),
    "MO": (38.4, -92.5),  "MT": (47.0, -109.6), "NE": (41.5, -99.8),
    "NV": (39.3, -116.6), "NH": (43.7, -71.6),  "NJ": (40.2, -74.7),
    "NM": (34.4, -106.1), "NY": (42.9, -75.5),  "NC": (35.5, -79.4),
    "ND": (47.4, -100.5), "OH": (40.3, -82.8),  "OK": (35.6, -97.5),
    "OR": (43.9, -120.6), "PA": (40.9, -77.8),  "RI": (41.7, -71.5),
    "SC": (33.9, -80.9),  "SD": (44.4, -100.2), "TN": (35.9, -86.4),
    "TX": (31.5, -99.3),  "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.9),  "WA": (47.4, -120.5), "WV": (38.6, -80.6),
    "WI": (44.6, -89.9),  "WY": (43.0, -107.5),
}

# Estados pequeños que se etiquetan FUERA del mapa con flecha (leader line).
# offset = (lat_texto, lon_texto) donde se coloca el cuadro de texto.
externos = {
    "VT": (47.5, -70.5), "NH": (46.5, -68.5), "MA": (44.5, -66.0),
    "RI": (42.6, -65.5), "CT": (40.8, -65.5), "NJ": (38.5, -68.5),
    "DE": (36.8, -70.0), "MD": (35.5, -73.0),
}

# ---------------------------------------------------------------------------
# 3) MAPA COROPLÉTICO (escala Greens: más verde = más producción)
# ---------------------------------------------------------------------------
fig = go.Figure(data=go.Choropleth(
    locations=df["abrev"],
    z=df["valor_num"],
    locationmode="USA-states",
    colorscale="Greens",
    marker_line_color="white",
    marker_line_width=0.6,
    colorbar_title="Producción<br>(USD miles<br>de millones)",
    hovertext=df["estado"] + ": " + df["valor_txt"],
    hoverinfo="text",
))

# ---------------------------------------------------------------------------
# 4) ETIQUETAS DENTRO DE LOS ESTADOS GRANDES/MEDIANOS
# ---------------------------------------------------------------------------
for _, row in df.iterrows():
    ab = row["abrev"]
    if ab in externos or ab not in centroids:
        continue
    lat, lon = centroids[ab]
    # Color de texto: blanco sobre estados muy verdes (oscuros), negro sobre claros
    color = "white" if row["valor_num"] >= 20 else "#111111"
    fig.add_trace(go.Scattergeo(
        lon=[lon], lat=[lat],
        text=f"<b>{row['estado']}</b><br>{row['valor_txt']}",
        mode="text",
        textfont=dict(size=7.5, color=color, family="Arial"),
        showlegend=False, hoverinfo="skip",
    ))

# ---------------------------------------------------------------------------
# 5) ETIQUETAS EXTERNAS CON FLECHA para los estados pequeños del noreste
# ---------------------------------------------------------------------------
for ab, (tlat, tlon) in externos.items():
    lat, lon = centroids[ab]
    row = df[df["abrev"] == ab].iloc[0]
    # Línea guía (flecha) desde el estado hasta el cuadro de texto
    fig.add_trace(go.Scattergeo(
        lon=[lon, tlon], lat=[lat, tlat],
        mode="lines",
        line=dict(width=0.7, color="gray"),
        showlegend=False, hoverinfo="skip",
    ))
    # Cuadro de texto
    fig.add_trace(go.Scattergeo(
        lon=[tlon], lat=[tlat],
        text=f"<b>{row['estado']}</b>  {row['valor_txt']}",
        mode="text",
        textfont=dict(size=8, color="#0b3d0b", family="Arial"),
        showlegend=False, hoverinfo="skip",
    ))

# ---------------------------------------------------------------------------
# 6) DISEÑO Y EXPORTACIÓN
# ---------------------------------------------------------------------------
fig.update_layout(
    geo=dict(
        scope="usa",
        projection=dict(type="albers usa"),
        showlakes=True, lakecolor="rgb(255,255,255)",
        bgcolor="rgba(0,0,0,0)",
    ),
    width=1400, height=900,
    margin=dict(l=10, r=10, t=90, b=10),
)

# --- EXPORTACIÓN PARA LA MEMORIA DEL TFM ---
# Vectorial (recomendado para insertar en el documento: nítido a cualquier zoom)
fig.write_image("mapa_produccion_agricola.pdf")            # PDF vectorial
fig.write_image("mapa_produccion_agricola.svg")            # SVG vectorial
# Mapa de bits en alta resolución (por si tu editor no admite vectoriales)
fig.write_image("mapa_produccion_agricola.png", scale=3)   # ~4200x2700 px
# Versión interactiva (para verlo en el navegador)
fig.write_html("mapa_produccion_agricola.html")

# Descomenta la siguiente línea si quieres que se abra en el navegador al ejecutar:
# fig.show()

print("OK: generados .pdf, .svg, .png y .html (mapa enfocado SOLO en EE.UU.)")