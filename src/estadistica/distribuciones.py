"""Transformaciones de u ∈ [0,1] a un valor físico según el tipo de prior."""

import numpy as np


def uniforme(u, minimo, maximo):
    return minimo + u * (maximo - minimo)


def log_uniforme(u, minimo, maximo):
    return np.exp(np.log(minimo) + u * np.log(maximo / minimo))
