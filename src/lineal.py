"""Ajuste lineal global (offsets + tendencia + sinusoides) y periodograma de chi2."""

import numpy as np
from scipy.linalg import qr


def matriz_diseno(serie, periodos=()):
    """Un offset por instrumento, una tendencia común (por año) y un seno/coseno por periodo."""
    t = serie.t - serie.t0
    columnas = [(serie.indice == k).astype(float) for k in range(len(serie.nombres))]
    columnas.append(t / 365.25)
    for periodo in periodos:
        fase = 2 * np.pi * t / periodo
        columnas += [np.sin(fase), np.cos(fase)]
    return np.column_stack(columnas)


def ajustar_lineal(serie, periodos=()):
    """Mínimos cuadrados ponderados de toda la serie (no por instrumento)."""
    X = matriz_diseno(serie, periodos)
    w = 1.0 / serie.sigma
    beta = np.linalg.lstsq(X * w[:, None], serie.y * w, rcond=None)[0]
    modelo = X @ beta
    residuos = serie.y - modelo
    return dict(beta=beta, modelo=modelo, residuos=residuos,
                chi2=float(np.sum((residuos * w) ** 2)),
                rms=float(np.sqrt(np.mean(residuos**2))))


def periodograma(serie, minimo, maximo, previos=()):
    """Mejora fraccional de chi2 al añadir seno y coseno de cada periodo de prueba.

    QR proyecta fuera offsets, tendencia y periodos `previos`.
    Devuelve (periodos, potencias, periodo_del_maximo). No da FAP.
    """
    t = serie.t - serie.t0
    sigma = serie.sigma
    Q, _ = qr(matriz_diseno(serie, previos) / sigma[:, None], mode="economic")
    yw = serie.y / sigma
    r = yw - Q @ (Q.T @ yw)
    chi0 = float(r @ r)
    fmin, fmax = 1.0 / maximo, 1.0 / minimo
    nfreq = int(np.clip(np.ceil(5 * np.ptp(serie.t) * (fmax - fmin)), 800, 8500))
    frecuencias = np.linspace(fmin, fmax, nfreq)
    potencias = np.empty(nfreq)
    for desde in range(0, nfreq, 160):
        hasta = min(desde + 160, nfreq)
        fase = 2 * np.pi * t[:, None] * frecuencias[None, desde:hasta]
        S, C = np.sin(fase) / sigma[:, None], np.cos(fase) / sigma[:, None]
        S -= Q @ (Q.T @ S)
        C -= Q @ (Q.T @ C)
        ss, cc, sc = np.sum(S * S, axis=0), np.sum(C * C, axis=0), np.sum(S * C, axis=0)
        ys, yc = r @ S, r @ C
        det = ss * cc - sc**2
        mejora = np.divide(cc * ys**2 + ss * yc**2 - 2 * sc * ys * yc,
                           det, out=np.zeros_like(det), where=det > 1e-14)
        potencias[desde:hasta] = np.clip(mejora / max(chi0, 1e-15), 0, 1)
    return 1 / frecuencias, potencias, float(1 / frecuencias[np.argmax(potencias)])
