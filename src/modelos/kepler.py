import numpy as np

MAX_ITERACIONES_NEWTON = 35
TOLERANCIA_NEWTON = 1e-12


def velocidad_kepleriana(tiempos, periodo, semiamplitud, excentricidad, omega, fase,
                         tiempo_referencia):
    """Resuelve M=E-e*sin(E) por Newton y devuelve la RV de una órbita."""
    if periodo <= 0 or semiamplitud < 0 or not 0 <= excentricidad < 1:
        raise ValueError("Parametros orbitales no fisicos")
    anomalia_media = 2.0 * np.pi * np.remainder(
        (tiempos - tiempo_referencia) / periodo - fase, 1.0)
    if excentricidad == 0.0:
        return semiamplitud * np.cos(anomalia_media + omega)
    anomalia_excentrica = anomalia_media + excentricidad * np.sin(anomalia_media)
    for _ in range(MAX_ITERACIONES_NEWTON):
        paso = ((anomalia_excentrica - excentricidad * np.sin(anomalia_excentrica)
                 - anomalia_media)
                / (1.0 - excentricidad * np.cos(anomalia_excentrica)))
        anomalia_excentrica -= paso
        if np.max(np.abs(paso)) < TOLERANCIA_NEWTON:
            break
    anomalia_verdadera = np.arctan2(
        np.sqrt(1.0 - excentricidad ** 2) * np.sin(anomalia_excentrica),
        np.cos(anomalia_excentrica) - excentricidad)
    return semiamplitud * (np.cos(anomalia_verdadera + omega)
                           + excentricidad * np.cos(omega))
