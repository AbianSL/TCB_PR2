"""Entrada unica: analisis global RV/S_MW, GP y Nested Sampling desde cero.

Para Spyder, cambiar MODO_POR_DEFECTO por "preciso" si se desea.
"""
import argparse

import matplotlib
matplotlib.use("Agg")

from src import config
from src.analisis import ANALISIS, crear
from src.datos import leer_series
from src.modelos import MODELOS


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Epsilon Eridani: RV, actividad y GP bayesiano")
    parser.add_argument("--analisis", nargs="+", choices=ANALISIS, default=list(ANALISIS))
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--sin-apf", action="store_true", help="Excluye el instrumento APF de RV")
    grupo_bayes = parser.add_argument_group("opciones de bayes")
    grupo_bayes.add_argument("--modo", choices=config.MUESTREO, default=config.MODO_POR_DEFECTO)
    grupo_bayes.add_argument("--nlive", type=int, help="Puntos vivos (sustituye al modo)")
    grupo_bayes.add_argument("--dlogz", type=float, help="Tolerancia de evidencia (sustituye al modo)")
    grupo_bayes.add_argument("--modelos", nargs="+", choices=MODELOS, default=list(MODELOS))
    grupo_bayes.add_argument("--priors-ampliados", action="store_true")
    grupo_bayes.add_argument("--reanudar", metavar="CARPETA", help="Reanuda una ejecución anterior")
    grupo_bayes.add_argument("--comprobar", action="store_true", help="Evalúa el GP sin muestrear")
    return parser.parse_args()


def main():
    argumentos = leer_argumentos()
    datos = leer_series(argumentos.sin_apf)
    for nombre_analisis in argumentos.analisis:
        print(f"\n== {nombre_analisis} ==")
        crear(nombre_analisis, argumentos).ejecutar(
            datos, config.DIRECTORIO_RESULTADOS / nombre_analisis)


if __name__ == "__main__":
    main()
