"""Entrada unica: analisis global RV/S_MW, GP y Nested Sampling desde cero.

Para Spyder, cambiar MODO_POR_DEFECTO por "preciso" si se desea.
"""
MODO_POR_DEFECTO = "preciso"  # "rapido", "normal" o "preciso"

import argparse
import sys
from pathlib import Path

# En Spyder pueden permanecer en memoria modulos de ejecuciones anteriores.
# Primero damos prioridad a la carpeta de ESTE proyecto y descartamos solo
# las posibles versiones anteriores de nuestro paquete "src".
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
for nombre_modulo in ("src.model_data", "src.exploratorio", "src"):
    sys.modules.pop(nombre_modulo, None)

from src import model_data as modelo, exploratorio


def main():
    parser = argparse.ArgumentParser(description="Epsilon Eridani: ajuste reproducible GP + Bayes")
    parser.add_argument("--solo-explorar", action="store_true",
                        help="Solo estadistica, periodogramas y ajuste de S_MW (rapido)")
    parser.add_argument("--comprobar", action="store_true", help="Prueba GP sin muestreo")
    parser.add_argument("--rapido", action="store_true", help="Muestreo exploratorio menos preciso")
    parser.add_argument("--preciso", action="store_true", help="Muestreo original, mas costoso")
    parser.add_argument("--nlive", type=int, help="Numero de puntos vivos")
    parser.add_argument("--dlogz", type=float, help="Tolerancia de evidencia")
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--modelos", nargs="+", choices=modelo.MODELOS,
                        default=list(modelo.MODELOS))
    parser.add_argument("--priors-ampliados", action="store_true")
    parser.add_argument("--sin-apf", action="store_true")
    parser.add_argument("--reanudar", metavar="CARPETA", help="Reanuda ejecucion anterior")
    args = parser.parse_args()
    if args.solo_explorar:
        exploratorio.analizar(RAIZ / "data", RAIZ / "results" / "exploratorio",
                            sin_apf=args.sin_apf)
        return
    if args.comprobar:
        modelo.comprobar()
        return
    if args.rapido and args.preciso:
        parser.error("Elegir --rapido o --preciso, no ambos")
    opciones = {"rapido": (50, 2.0), "normal": (80, 1.0), "preciso": (200, 0.1)}
    modo = "rapido" if args.rapido else "preciso" if args.preciso else MODO_POR_DEFECTO
    nlive, dlogz = opciones[modo]
    import importlib.util
    ausentes = [lib for lib in ("dynesty", "celerite2")
                if importlib.util.find_spec(lib) is None]
    if ausentes:
        parser.error("Faltan dependencias: " + ", ".join(ausentes) +
                     ". Instalar con: python -m pip install -r requirements.txt")
    if args.nlive is not None:
        nlive = args.nlive
    if args.dlogz is not None:
        dlogz = args.dlogz
    modelo.ajustar(tuple(args.modelos), nlive=nlive, dlogz=dlogz,
                  semilla=args.semilla, ampliados=args.priors_ampliados,
                  sin_apf=args.sin_apf, carpeta_reanudacion=args.reanudar)


if __name__ == "__main__":
    main()
