import numpy as np

def velocidad_kepleriana(t, periodo, K, excentricidad, omega, fase, t0):
    """Resuelve M=E-e*sin(E) por Newton y devuelve la RV de una órbita."""
    e = excentricidad
    if periodo <= 0 or K < 0 or not 0 <= e < 1:
        raise ValueError("Parametros orbitales no fisicos")
    anomalia_media = 2.0 * np.pi * np.remainder((t - t0) / periodo - fase, 1.0)
    if e == 0.0:
        return K * np.cos(anomalia_media + omega)
    anomalia_excentrica = anomalia_media + e * np.sin(anomalia_media)
    for _ in range(35):
        paso = ((anomalia_excentrica - e * np.sin(anomalia_excentrica) - anomalia_media)
                / (1.0 - e * np.cos(anomalia_excentrica)))
        anomalia_excentrica -= paso
        if np.max(np.abs(paso)) < 1e-12:
            break
    anomalia_verdadera = np.arctan2(np.sqrt(1.0 - e * e) * np.sin(anomalia_excentrica),
                                    np.cos(anomalia_excentrica) - e)
    return K * (np.cos(anomalia_verdadera + omega) + e * np.cos(omega))
