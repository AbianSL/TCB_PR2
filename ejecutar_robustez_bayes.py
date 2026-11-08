"""Análisis bayesiano de robustez (Epsilon Eridani, RV global).

Abre este archivo en Spyder y ejecútalo; no ejecutes main.py.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_data import main_robustez_bayes

# ===========================================================
# CONFIGURACIÓN: editar únicamente este apartado.
# ===========================================================
MODO = "RESUMEN"  # "PRUEBA", "M0", "M1", "AMBOS", "RESUMEN"
ESCENARIO = "BASE_SEMILLA_2"  # Opciones documentadas en la guia
NLIVE = 400
DLOGZ = 0.05
SEMILLA = 31415
SEMILLA_ALTERNATIVA = 27182
PERIODO_MIN = 200.0
PERIODO_MAX = 6000.0


def main():
    if not (ROOT / "src" / "model_data.py").is_file():
        raise FileNotFoundError("Debe existir src/model_data.py al lado de este ejecutor")
    if not (ROOT / "data" / "Epsilon_Eridani_RVdata.csv").is_file():
        raise FileNotFoundError("Debe existir data/Epsilon_Eridani_RVdata.csv")
    main_robustez_bayes(
        escenario=ESCENARIO, modo=MODO, nlive=NLIVE, dlogz=DLOGZ,
        semilla=SEMILLA, semilla_alternativa=SEMILLA_ALTERNATIVA,
        periodo_min=PERIODO_MIN, periodo_max=PERIODO_MAX)


if __name__ == "__main__":
    main()
