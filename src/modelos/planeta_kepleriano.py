import numpy as np

from src.estadistica.distribuciones import uniforme
from src.modelos.planeta_circular import ModeloPlanetaCircular


class ModeloPlanetaKepleriano(ModeloPlanetaCircular):
    """M2: GP + planeta kepleriano con excentricidad libre."""

    nombre = "M2"
    parametros_posterior = ModeloPlanetaCircular.parametros_posterior + ("planeta_e",)

    def _priors_orbitales(self):
        return super()._priors_orbitales() + [
            ("planeta_e", uniforme, 0.0, 0.8),
            ("planeta_omega_rad", uniforme, 0.0, 2.0 * np.pi),
        ]

    def _forma_orbita(self, parametros):
        return parametros["planeta_e"], parametros["planeta_omega_rad"]
