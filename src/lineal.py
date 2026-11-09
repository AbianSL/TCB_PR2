"""Ajuste lineal global (offsets + tendencia + sinusoides) y periodograma de chi2."""

import numpy as np
from scipy.linalg import qr

DIAS_POR_ANIO = 365.25
FRECUENCIAS_POR_LOTE = 160
MIN_FRECUENCIAS, MAX_FRECUENCIAS = 800, 8500
MUESTRAS_POR_RESOLUCION_DE_FOURIER = 5


def matriz_diseno(serie, periodos=()):
    """Un offset por instrumento, una tendencia común (por año) y un seno/coseno por periodo."""
    tiempos = serie.tiempos - serie.tiempo_referencia
    columnas = [(serie.indice_instrumento == posicion).astype(float)
                for posicion in range(len(serie.nombres_instrumentos))]
    columnas.append(tiempos / DIAS_POR_ANIO)
    for periodo in periodos:
        fase = 2 * np.pi * tiempos / periodo
        columnas += [np.sin(fase), np.cos(fase)]
    return np.column_stack(columnas)


def ajustar_lineal(serie, periodos=()):
    """Mínimos cuadrados ponderados de toda la serie (no por instrumento).

    `coeficientes` = [offset de cada instrumento, pendiente por año,
    (seno, coseno) de cada periodo].
    """
    matriz = matriz_diseno(serie, periodos)
    pesos = 1.0 / serie.sigma_descriptivo
    coeficientes = np.linalg.lstsq(matriz * pesos[:, None], serie.valores * pesos,
                                   rcond=None)[0]
    modelo = matriz @ coeficientes
    residuos = serie.valores - modelo
    return dict(coeficientes=coeficientes, modelo=modelo, residuos=residuos,
                chi2=float(np.sum((residuos * pesos) ** 2)),
                rms=float(np.sqrt(np.mean(residuos ** 2))))


def periodograma(serie, periodo_minimo, periodo_maximo, periodos_previos=()):
    """Mejora fraccional de chi2 al añadir seno y coseno de cada periodo de prueba.

    QR proyecta fuera offsets, tendencia y `periodos_previos`.
    Devuelve (periodos, potencias, periodo_del_maximo). No da FAP.
    """
    tiempos = serie.tiempos - serie.tiempo_referencia
    sigma = serie.sigma_descriptivo
    base_ortonormal, _ = qr(matriz_diseno(serie, periodos_previos) / sigma[:, None],
                            mode="economic")
    valores_ponderados = serie.valores / sigma
    residuo_ponderado = valores_ponderados - base_ortonormal @ (
        base_ortonormal.T @ valores_ponderados)
    chi2_sin_senal = float(residuo_ponderado @ residuo_ponderado)

    frecuencia_minima, frecuencia_maxima = 1.0 / periodo_maximo, 1.0 / periodo_minimo
    n_frecuencias = int(np.clip(
        np.ceil(MUESTRAS_POR_RESOLUCION_DE_FOURIER * np.ptp(serie.tiempos)
                * (frecuencia_maxima - frecuencia_minima)),
        MIN_FRECUENCIAS, MAX_FRECUENCIAS))
    frecuencias = np.linspace(frecuencia_minima, frecuencia_maxima, n_frecuencias)
    potencias = np.empty(n_frecuencias)

    for desde in range(0, n_frecuencias, FRECUENCIAS_POR_LOTE):
        hasta = min(desde + FRECUENCIAS_POR_LOTE, n_frecuencias)
        fase = 2 * np.pi * tiempos[:, None] * frecuencias[None, desde:hasta]
        seno = np.sin(fase) / sigma[:, None]
        coseno = np.cos(fase) / sigma[:, None]
        seno -= base_ortonormal @ (base_ortonormal.T @ seno)
        coseno -= base_ortonormal @ (base_ortonormal.T @ coseno)
        suma_seno_seno = np.sum(seno * seno, axis=0)
        suma_coseno_coseno = np.sum(coseno * coseno, axis=0)
        suma_seno_coseno = np.sum(seno * coseno, axis=0)
        proyeccion_seno = residuo_ponderado @ seno
        proyeccion_coseno = residuo_ponderado @ coseno
        determinante = suma_seno_seno * suma_coseno_coseno - suma_seno_coseno ** 2
        mejora_chi2 = np.divide(
            suma_coseno_coseno * proyeccion_seno ** 2
            + suma_seno_seno * proyeccion_coseno ** 2
            - 2 * suma_seno_coseno * proyeccion_seno * proyeccion_coseno,
            determinante, out=np.zeros_like(determinante), where=determinante > 1e-14)
        potencias[desde:hasta] = np.clip(mejora_chi2 / max(chi2_sin_senal, 1e-15), 0, 1)
    periodos = 1 / frecuencias
    return periodos, potencias, float(periodos[np.argmax(potencias)])
