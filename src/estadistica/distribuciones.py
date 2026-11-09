"""Transformaciones de un cuantil u ∈ [0,1] a un valor físico según el tipo de prior."""

import numpy as np


def uniforme(cuantil, minimo, maximo):
    return minimo + cuantil * (maximo - minimo)


def log_uniforme(cuantil, minimo, maximo):
    return np.exp(np.log(minimo) + cuantil * np.log(maximo / minimo))
