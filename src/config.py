"""Rutas y constantes del proyecto."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIRECTORIO_DATOS = RAIZ / "data"
CSV_VELOCIDADES = DIRECTORIO_DATOS / "Epsilon_Eridani_RVdata.csv"
CSV_ACTIVIDAD = DIRECTORIO_DATOS / "Epsilon_Eridani_SMW.csv"
DIRECTORIO_RESULTADOS = RAIZ / "results"

EPOCA_BJD = 2450000.0
INSTRUMENTO_APF = "APF_1"
PAQUETES_REGISTRADOS = ("numpy", "scipy", "matplotlib", "astropy", "celerite2", "dynesty")

# Modos de muestreo de dynesty: (nlive, dlogz)
MUESTREO = {"rapido": (50, 2.0), "normal": (80, 1.0), "preciso": (200, 0.1)}
MODO_POR_DEFECTO = "preciso"
