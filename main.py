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
    p = argparse.ArgumentParser(description="Epsilon Eridani: RV, actividad y GP bayesiano")
    p.add_argument("--analisis", nargs="+", choices=ANALISIS, default=list(ANALISIS))
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--sin-apf", action="store_true", help="Excluye el instrumento APF de RV")
    bayes = p.add_argument_group("opciones de bayes")
    bayes.add_argument("--modo", choices=config.MUESTREO, default=config.MODO_POR_DEFECTO)
    bayes.add_argument("--nlive", type=int, help="Puntos vivos (sustituye al modo)")
    bayes.add_argument("--dlogz", type=float, help="Tolerancia de evidencia (sustituye al modo)")
    bayes.add_argument("--modelos", nargs="+", choices=MODELOS, default=list(MODELOS))
    bayes.add_argument("--priors-ampliados", action="store_true")
    bayes.add_argument("--reanudar", metavar="CARPETA", help="Reanuda una ejecución anterior")
    bayes.add_argument("--comprobar", action="store_true", help="Evalúa el GP sin muestrear")
    return p.parse_args()


def main():
    args = leer_argumentos()
    datos = leer_series(args.sin_apf)
    for nombre in args.analisis:
        print(f"\n== {nombre} ==")
        crear(nombre, args).ejecutar(datos, config.DIRECTORIO_RESULTADOS / nombre)


if __name__ == "__main__":
    main()
