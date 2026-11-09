from src.estadistica.distribuciones import log_uniforme, uniforme
from src.modelos.actividad import ModeloActividad
from src.modelos.kepler import velocidad_kepleriana


class ModeloPlanetaCircular(ModeloActividad):
    """M1: GP + planeta en órbita circular."""

    nombre = "M1"
    parametros_posterior = ("gp_periodo_rot_d", "planeta_periodo_d", "planeta_K_m_s")

    def velocidad_orbital(self, parametros, serie):
        excentricidad, omega = self._forma_orbita(parametros)
        return velocidad_kepleriana(
            serie.tiempos, parametros["planeta_periodo_d"], parametros["planeta_K_m_s"],
            excentricidad, omega, parametros["planeta_fase"], serie.tiempo_referencia)

    def _priors_orbitales(self):
        return [
            ("planeta_periodo_d", log_uniforme, 200.0, 6000.0),
            ("planeta_K_m_s", uniforme, 0.0, 80.0),
            ("planeta_fase", uniforme, 0.0, 1.0),
        ]

    def _forma_orbita(self, parametros):
        """(excentricidad, omega) de la órbita."""
        return 0.0, 0.0
