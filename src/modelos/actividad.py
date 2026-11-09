import numpy as np

from src.estadistica.distribuciones import log_uniforme, uniforme
from src.estadistica.espacio_parametros import EspacioParametros


class ModeloActividad:
    """M0: solo actividad estelar (GP). Base de los modelos con planeta.

    Los jitters y el GP son comunes; las subclases añaden la parte orbital
    sobrescribiendo `_priors_orbitales` y `velocidad_orbital`.
    """

    nombre = "M0"
    parametros_posterior = ("gp_periodo_rot_d",)

    def espacio_parametros(self, instrumentos, ampliados=False):
        """Priors originales; `ampliados` modifica solo Q0 y la escala lenta."""
        priors = [(f"jitter_{i}", log_uniforme, 0.01, 100.0) for i in instrumentos]
        priors += self._priors_gp(ampliados)
        priors += self._priors_orbitales()
        return EspacioParametros(priors)

    def velocidad_orbital(self, parametros, serie):
        return np.zeros_like(serie.t)

    def _priors_orbitales(self):
        return []

    @staticmethod
    def _priors_gp(ampliados):
        return [
            ("gp_sigma_rot_m_s", log_uniforme, 0.05, 100.0),
            ("gp_periodo_rot_d", uniforme, 8.0, 16.0),
            ("gp_Q0", log_uniforme, 0.03 if ampliados else 0.2, 20.0),
            ("gp_dQ", log_uniforme, 0.2, 50.0),
            ("gp_frac_armonico", uniforme, 0.02, 1.0),
            ("gp_sigma_lenta_m_s", log_uniforme, 0.05, 100.0),
            ("gp_escala_lenta_d", log_uniforme, 30.0 if ampliados else 150.0, 5000.0),
        ]
