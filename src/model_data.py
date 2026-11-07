import numpy as np
import src.figures as figures
from scipy.optimize import curve_fit

def fit_rv_trends(rjd, rv, srv, instrumento):
    """Ajustar una constante y una recta por instrumento."""

    for nombre in np.unique(instrumento):
        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        velocidad = rv[mascara]
        error = srv[mascara]

        # Centrar el tiempo facilita el ajuste
        t0 = np.mean(tiempo)
        t = tiempo - t0

        # Modelo constante: media ponderada por las incertidumbres
        pesos = 1 / error**2
        constante = np.sum(pesos * velocidad) / np.sum(pesos)
        modelo_constante = np.full(len(t), constante)

        # Modelo lineal: pendiente e intercepto
        coeficientes = np.polyfit(t, velocidad, 1, w=1 / error)
        pendiente, intercepto = coeficientes
        modelo_lineal = np.polyval(coeficientes, t)

        # Residuos de ambos modelos
        residuos_constante = velocidad - modelo_constante
        residuos_lineal = velocidad - modelo_lineal

        rms_constante = np.sqrt(np.mean(residuos_constante**2))
        rms_lineal = np.sqrt(np.mean(residuos_lineal**2))

        chi2_constante = np.sum((residuos_constante / error)**2)
        chi2_lineal = np.sum((residuos_lineal / error)**2)

        n = len(velocidad)

        print("\nAjustes de RV:", nombre)
        print("Pendiente (m/s por año):", pendiente * 365.25)
        print("RMS constante (m/s):", rms_constante)
        print("RMS lineal (m/s):", rms_lineal)
        print("Chi² reducido constante:", chi2_constante / (n - 1))
        print("Chi² reducido lineal:", chi2_lineal / (n - 2))

        figures.plot_rv_trends(
            tiempo, velocidad, error,
            modelo_constante, modelo_lineal, nombre
        ) 

def linear_sine(t, c, m, a, b, periodo):
    """Recta más una oscilación sinusoidal."""
    fase = 2 * np.pi * t / periodo
    return c + m * t + a * np.sin(fase) + b * np.cos(fase)


def fit_rv_sine(rjd, rv, srv, instrumento):
    """Explorar los candidatos de unos 12 días."""

    nombres = ["ESPRESSO18_1", "HARPS15_1"]
    periodos_iniciales = [11.91, 12.34]

    for nombre, periodo_inicial in zip(nombres, periodos_iniciales):
        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        velocidad = rv[mascara]
        error = srv[mascara]

        # Centrar tiempo y velocidad para facilitar el ajuste
        t0 = np.mean(tiempo)
        referencia = np.mean(velocidad)
        t = tiempo - t0
        y = velocidad - referencia

        # Ajuste de referencia: recta
        pendiente, intercepto = np.polyfit(t, y, 1, w=1 / error)
        modelo_recta = intercepto + pendiente * t

        # Valores iniciales de c, m, a, b y periodo
        iniciales = [
            intercepto, pendiente, np.std(y), 0, periodo_inicial
        ]

        limites = (
            [-np.inf, -np.inf, -np.inf, -np.inf, 10],
            [np.inf, np.inf, np.inf, np.inf, 15]
        )

        parametros, covarianza = curve_fit(
            linear_sine, t, y,
            p0=iniciales,
            sigma=error,
            absolute_sigma=True,
            bounds=limites,
            maxfev=20000
        )

        modelo_seno = linear_sine(t, *parametros)

        residuos_recta = y - modelo_recta
        residuos_seno = y - modelo_seno

        rms_recta = np.sqrt(np.mean(residuos_recta**2))
        rms_seno = np.sqrt(np.mean(residuos_seno**2))

        chi2_recta = np.sum((residuos_recta / error)**2)
        chi2_seno = np.sum((residuos_seno / error)**2)

        c, m, a, b, periodo = parametros
        amplitud = np.sqrt(a**2 + b**2)
        n = len(y)

        print("\nRecta más sinusoide:", nombre)
        print("Periodo ajustado (días):", periodo)
        print("Semiamplitud (m/s):", amplitud)
        print("RMS recta (m/s):", rms_recta)
        print("RMS recta + seno (m/s):", rms_seno)
        print("Chi² reducido recta:", chi2_recta / (n - 2))
        print("Chi² reducido recta + seno:", chi2_seno / (n - 5))

        # Evaluar la curva en tiempos más densos para dibujarla
        t_curva = np.linspace(t.min(), t.max(), 20000)
        curva = linear_sine(t_curva, *parametros)

        figures.plot_rv_sine(
            tiempo, y, error, modelo_recta,
            t_curva + t0, curva, residuos_seno, nombre
        )