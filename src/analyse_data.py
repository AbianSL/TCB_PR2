import numpy as np
import src.manage_data as manage_data
import src.figures as figures
from astropy.timeseries import LombScargle

def describe_data(rjd, y, sy, instrumento, observable):

    print("\nSerie:", observable)
    print("Número de medidas:", len(y))
    print("Instrumentos:", np.unique(instrumento))

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        valores = y[mascara]
        errores = sy[mascara]

        print("\nInstrumento:", nombre)
        print("Número de medidas:", len(valores))
        print("Duración (días):", tiempo.max() - tiempo.min())
        print("Media:", np.mean(valores))
        print("Dispersión:", np.std(valores))
        print("Error medio:", np.mean(errores))

def inspect_outliers(rjd, y, instrumento, observable):

    print("\nRevisión de puntos atípicos:", observable)

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        valores = y[mascara]

        keep = manage_data.sigma_clip(valores)

        print("\nInstrumento:", nombre)
        print("Puntos marcados:", np.sum(~keep))

        for t, valor in zip(tiempo[~keep], valores[~keep]):
            print("BJD - 2450000:", t, "Valor:", valor)

def inspect_region(rjd, y, sy, instrumento, nombre, tmin, tmax):

    mascara = instrumento == nombre

    tiempo = rjd[mascara]
    actividad = y[mascara]
    error = sy[mascara]

    keep = manage_data.sigma_clip(actividad)

    figures.plot_outliers(
        tiempo, actividad, error, keep, nombre, tmin, tmax
    ) 


def aux_ls(tiempo, valores, errores, frecuencias):

    errores_ls = np.sqrt(errores**2 + np.std(valores)**2)

    ls = LombScargle(tiempo, valores, errores_ls)
    potencia = ls.power(frecuencias)

    periodo_pico = 1 / frecuencias[np.argmax(potencia)]

    faps = [0.1, 0.01, 0.001]
    niveles = []

    for fap in faps:

        nivel = ls.false_alarm_level(
            fap,
            method="baluev",
            minimum_frequency=frecuencias.min(),
            maximum_frequency=frecuencias.max()
        )

        niveles.append(nivel)

    return potencia, periodo_pico, niveles


def compare_activity_periodograms(
    rjd, s, ss, instrumento,
    rjd_clean, s_clean, ss_clean, instrumento_clean
):

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre
        mascara_clean = instrumento_clean == nombre

        tiempo = rjd[mascara]
        tiempo_clean = rjd_clean[mascara_clean]

        # No explorar periodos mayores que la cobertura del instrumento
        periodo_max = min(1000, tiempo.max() - tiempo.min())
        frecuencias = np.linspace(1 / periodo_max, 1 / 2, 10000)

        potencia, periodo, niveles = aux_ls(
            tiempo, s[mascara], ss[mascara], frecuencias
        )

        potencia_clean, periodo_clean, niveles_clean = aux_ls(
            tiempo_clean, s_clean[mascara_clean],
            ss_clean[mascara_clean], frecuencias
        )

        print("\nPeriodograma de actividad:", nombre)
        print("Pico con todos los datos (días):", periodo)
        print("Pico con datos filtrados (días):", periodo_clean)

        figures.plot_periodogram_comparison(
            frecuencias, potencia, potencia_clean,
            nombre, niveles, niveles_clean
        )

def analyse_window(rjd, instrumento):

    frecuencias = np.linspace(0, 1.1, 20000)

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre
        tiempo = rjd[mascara]
        tiempo = tiempo - tiempo.min()

        ventana = np.zeros(len(frecuencias))

        for i, frecuencia in enumerate(frecuencias):

            fase = 2 * np.pi * frecuencia * tiempo

            ventana[i] = (
                np.mean(np.cos(fase))**2
                + np.mean(np.sin(fase))**2
            )

        figures.plot_window(frecuencias, ventana, nombre)

def analyse_rv_periodograms(rjd, rv, srv, instrumento):

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        velocidad = rv[mascara]
        error = srv[mascara]

        duracion = tiempo.max() - tiempo.min()

        # Explorar desde 2 días hasta la cobertura del instrumento
        periodo_max = duracion

        # Diez puntos por escala de resolución en frecuencia
        paso = 1 / (10 * duracion)

        frecuencias = np.arange(
            1 / periodo_max,
            1 / 2,
            paso
        )

        potencia, periodo, niveles = aux_ls(
            tiempo, velocidad, error, frecuencias
        )

        print("\nPeriodograma de RV:", nombre)
        print("Periodo del máximo (días):", periodo)

        figures.plot_rv_periodogram(
            frecuencias, potencia, niveles, nombre
        )