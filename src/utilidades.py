"""Funciones de salida compartidas por todos los análisis."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def guardar_tabla(ruta, encabezado, filas):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


def guardar_figura(fig, ruta, formatos=(".png",), dpi=170):
    """Guarda la figura (sin extensión en `ruta`) y la cierra."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for formato in formatos:
        fig.savefig(ruta.with_suffix(formato), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
