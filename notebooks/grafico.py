"""
================================================================================
 GRÁFICO DE BARRAS — CAMBIO DEL IAI POR CULTIVO (ESCENARIOS FUTUROS)
================================================================================
Genera dos versiones del gráfico de barras del cambio del IAI:
  1. Solo escenario severo (SSP5-8.5), con colores por signo.
  2. Ambos escenarios lado a lado (SSP2-4.5 y SSP5-8.5).

Usa los resultados guardados del notebook de predicción futura.
================================================================================
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# ─── Cargar los resultados ───
iai_futuro = pl.read_parquet("C:/Users/lesli/Documents/TFM/Data/iai_futuro_predicho_v3.parquet")
iai_2025 = pl.read_parquet("C:/Users/lesli/Documents/TFM/Data/iai_2025_predicho_v3.parquet")

CROP_DICT_ES = {
    75: "Almendras", 69: "Uvas", 204: "Pistachos", 76: "Nueces",
    54: "Tomates", 3: "Arroz", 221: "Fresas", 212: "Naranjas",
    36: "Alfalfa", 24: "Trigo", 227: "Lechuga", 2: "Algodón",
}

# ─── Calcular el IAI medio por cultivo en presente y futuro ───
# Presente (2025 predicho)
presente = (
    iai_2025.group_by("crop_id")
    .agg(pl.col("iai_pred").mean().alias("presente"))
)

# Futuro por escenario
futuro = (
    iai_futuro.group_by(["crop_id", "escenario"])
    .agg(pl.col("iai_pred").mean().alias("iai"))
)

# Unir y calcular el cambio
tabla = presente.join(futuro, on="crop_id")
tabla = tabla.with_columns((pl.col("iai") - pl.col("presente")).alias("cambio"))
tabla = tabla.with_columns(
    pl.col("crop_id").replace_strict(CROP_DICT_ES).alias("cultivo")
)

# Separar por escenario
df = tabla.to_pandas()
sev = df[df["escenario"] == "SSP5-8.5"].sort_values("cambio")
mod = df[df["escenario"] == "SSP2-4.5"].set_index("cultivo")

# Colores de la paleta del proyecto
VERDE = "#1D9E75"
ROJO = "#D85A30"
VERDE_OSCURO = "#0F6E56"
AMBAR = "#E0A83C"


# ══════════════════════════════════════════════════════════════════
#  VERSIÓN 1 — Solo escenario severo, colores por signo
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))

colores = [ROJO if c < 0 else VERDE for c in sev["cambio"]]
barras = ax.barh(sev["cultivo"], sev["cambio"], color=colores, edgecolor="white")

# Línea vertical en cero
ax.axvline(0, color="#555555", linewidth=0.8)

# Etiquetas de valor al final de cada barra
for barra, valor in zip(barras, sev["cambio"]):
    x = barra.get_width()
    offset = 0.002 if x >= 0 else -0.002
    ax.text(x + offset, barra.get_y() + barra.get_height()/2,
            f"{valor:+.3f}".replace(".", ","),
            va="center", ha="left" if x >= 0 else "right",
            fontsize=9, color="#333333")

ax.set_xlabel("Cambio en el IAI respecto al presente")

ax.spines[["top", "right"]].set_visible(False)
ax.margins(x=0.15)
plt.tight_layout()
plt.savefig("cambio_iai_severo.png", dpi=150, bbox_inches="tight")
print("Guardado: cambio_iai_severo_nt.png")
plt.close()


# ══════════════════════════════════════════════════════════════════
#  VERSIÓN 2 — Ambos escenarios lado a lado
# ══════════════════════════════════════════════════════════════════
# Ordenar por el cambio en el escenario severo
orden = sev["cultivo"].tolist()
mod_ord = [mod.loc[c, "cambio"] for c in orden]
sev_ord = sev["cambio"].tolist()

fig, ax = plt.subplots(figsize=(8, 7))
y = np.arange(len(orden))
altura = 0.4

ax.barh(y + altura/2, mod_ord, altura, label="Moderado (SSP2-4.5)",
        color=AMBAR, edgecolor="white")
ax.barh(y - altura/2, sev_ord, altura, label="Severo (SSP5-8.5)",
        color=VERDE_OSCURO, edgecolor="white")

ax.axvline(0, color="#555555", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(orden)
ax.set_xlabel("Cambio en el IAI respecto al presente")

ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.margins(x=0.12)
plt.tight_layout()
plt.savefig("cambio_iai_ambos.png", dpi=150, bbox_inches="tight")
print("Guardado: cambio_iai_ambos_nt.png")
plt.close()

