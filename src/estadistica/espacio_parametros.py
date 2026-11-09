import numpy as np

class EspacioParametros:
    """Parámetros de un modelo y sus priors. Es invocable: es el prior_transform de dynesty.

    Cada prior es una tupla (nombre, distribucion, minimo, maximo), donde
    `distribucion` es una función de distribuciones.py.
    """

    def __init__(self, priors):
        self.priors = list(priors)

    @property
    def nombres(self):
        return [nombre for nombre, *_ in self.priors]

    def __len__(self):
        return len(self.priors)

    def __call__(self, cubo_unidad):
        """dynesty explora [0,1]^D; lo convertimos a parámetros físicos."""
        return np.asarray([
            distribucion(cuantil, minimo, maximo)
            for cuantil, (_, distribucion, minimo, maximo) in zip(cubo_unidad, self.priors)])

    def como_diccionario(self, vector_parametros):
        return dict(zip(self.nombres, vector_parametros))
