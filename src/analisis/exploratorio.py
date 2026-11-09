"""Exploración global de RV y S_MW: estadísticas, periodos y ajuste sinusoidal.

Los instrumentos aportan distintos offsets, pero una tendencia y señales
periódicas comunes. Los periodogramas miden reducción de chi2 y NO dan FAP.
Figuras con paleta Okabe-Ito y marcadores distintos por instrumento.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analisis.base import Analisis
from src.estilos import (COLOR_ESPECTRO, COLOR_MODELO, COLOR_PERIODO_LIBRE,
                         COLOR_PERIODO_USADO, estilo_instrumento)
from src.lineal import ajustar_lineal, periodograma
from src.utilidades import guardar_figura, guardar_tabla

# Se busca primero el periodo largo, se retira, y luego el corto tras retirarlo.
VENTANA_PERIODO_LARGO_DIAS = (200, 6000)
VENTANA_PERIODO_CORTO_DIAS = (5, 25)
# En S_MW el máximo global corto suele rondar 8 d. Para el ajuste ilustrativo se
# elige el mejor pico en 10-15 d (candidatos cercanos a 12 d de RV). NO es una
# detección independiente de rotación.
VENTANA_AJUSTE_ACTIVIDAD_DIAS = (10, 15)
PERIODOS_VENTANA_MUESTREO_DIAS = (100, 1600)
TOLERANCIA_PICOS_DISTINTOS_DIAS = 0.05


class Exploratorio(Analisis):
    def ejecutar(self, datos, salida):
        estadisticas = self._estadisticas_por_instrumento(datos)
        busquedas = {}
        for nombre_serie, serie in datos.items():
            busquedas[nombre_serie] = self._buscar_periodos(nombre_serie, serie)
            busqueda = busquedas[nombre_serie]
            print(f"  {nombre_serie}: P_largo={busqueda['periodo_largo']:.2f} d, "
                  f"P_corto_ajustado={busqueda['periodo_corto_usado']:.4f} d, "
                  f"maximo_corto_libre={busqueda['periodo_corto_libre']:.4f} d (exploratorios)")

        guardar_tabla(salida / "estadisticas_instrumentos.csv",
                      ["serie", "instrumento", "N", "BJD_min", "BJD_max", "mediana",
                       "std_muestral", "mediana_error_formal"], estadisticas)
        guardar_tabla(salida / "periodos_exploratorios.csv",
                      ["serie", "periodo_largo_d", "periodo_corto_ajustado_d",
                       "maximo_corto_libre_d", "RMS_residual"],
                      [[nombre, busqueda["periodo_largo"], busqueda["periodo_corto_usado"],
                        busqueda["periodo_corto_libre"], busqueda["ajuste"]["rms"]]
                       for nombre, busqueda in busquedas.items()])

        self._figura_series(datos, salida / "01_series")
        self._figura_periodogramas(busquedas, salida / "02_periodogramas")
        self._figura_ventana_muestreo(datos, salida / "03_ventana_anual")
        self._figura_ajuste_actividad(datos["S_MW"], busquedas["S_MW"]["ajuste"],
                                      salida / "04_actividad_ajuste")
        print(f"Figuras y tablas: {salida}")

    @staticmethod
    def _estadisticas_por_instrumento(datos):
        filas = []
        for nombre_serie, serie in datos.items():
            for instrumento, mascara in serie.instrumentos():
                filas.append([
                    nombre_serie, instrumento, int(mascara.sum()),
                    float(serie.tiempos[mascara].min()), float(serie.tiempos[mascara].max()),
                    float(np.median(serie.valores[mascara])),
                    float(np.std(serie.valores[mascara], ddof=1)),
                    float(np.median(serie.errores_formales[mascara]))])
        return filas

    @staticmethod
    def _buscar_periodos(nombre_serie, serie):
        """Periodo largo, luego periodo corto tras retirar el largo, y ajuste global."""
        periodos_largos, potencia_larga, periodo_largo = periodograma(
            serie, *VENTANA_PERIODO_LARGO_DIAS)
        periodos_cortos, potencia_corta, periodo_corto_libre = periodograma(
            serie, *VENTANA_PERIODO_CORTO_DIAS, (periodo_largo,))
        periodo_corto_usado = periodo_corto_libre
        if nombre_serie == "S_MW":
            minimo, maximo = VENTANA_AJUSTE_ACTIVIDAD_DIAS
            en_ventana = (periodos_cortos >= minimo) & (periodos_cortos <= maximo)
            periodo_corto_usado = float(
                periodos_cortos[en_ventana][np.argmax(potencia_corta[en_ventana])])
        return dict(
            periodo_largo=periodo_largo,
            periodo_corto_usado=periodo_corto_usado,
            periodo_corto_libre=periodo_corto_libre,
            espectro_largo=(periodos_largos, potencia_larga),
            espectro_corto=(periodos_cortos, potencia_corta),
            ajuste=ajustar_lineal(serie, (periodo_largo, periodo_corto_usado)))

    @staticmethod
    def _potencia_ventana_muestreo(serie):
        """|media(exp(2*pi*i*f*t))|^2: revela periodicidades del propio muestreo."""
        tiempos = serie.tiempos - serie.tiempo_referencia
        minimo, maximo = PERIODOS_VENTANA_MUESTREO_DIAS
        frecuencias = np.linspace(1 / maximo, 1 / minimo, 700)
        potencia = np.empty_like(frecuencias)
        for posicion, frecuencia in enumerate(frecuencias):
            fase = 2 * np.pi * frecuencia * tiempos
            potencia[posicion] = np.mean(np.cos(fase)) ** 2 + np.mean(np.sin(fase)) ** 2
        return 1 / frecuencias, potencia

    @staticmethod
    def _figura_series(datos, ruta):
        """Series completas con error formal (1 sigma); el centrado es solo visual."""
        figura, ejes = plt.subplots(len(datos), 1, figsize=(10, 7), sharex=True)
        for eje, (nombre_serie, serie) in zip(ejes, datos.items()):
            for instrumento, mascara in serie.instrumentos():
                color, marcador = estilo_instrumento(instrumento)
                eje.errorbar(
                    serie.tiempos[mascara], serie.valores[mascara] - np.median(serie.valores[mascara]),
                    yerr=serie.errores_formales[mascara],  # error formal; NO el piso empírico
                    fmt=marcador, color=color, markersize=3.1, markeredgewidth=0.25,
                    elinewidth=0.55, capsize=0, alpha=0.7, linestyle="none",
                    label=instrumento)
            eje.set_ylabel("RV - mediana (m/s)" if nombre_serie == "RV"
                           else "S_MW - mediana")
            eje.legend(fontsize=7, ncol=4)
        ejes[-1].set_xlabel("BJD - 2450000 (días)")
        figura.suptitle("Series completas con incertidumbres (1 sigma); centrado solo visual")
        guardar_figura(figura, ruta, dpi=300)

    @staticmethod
    def _figura_periodogramas(busquedas, ruta):
        figura, ejes = plt.subplots(len(busquedas), 2, figsize=(11, 7))
        titulos = ("200-6000 d", "5-25 d; tras quitar periodo largo")
        for fila, (nombre_serie, busqueda) in enumerate(busquedas.items()):
            espectros = (busqueda["espectro_largo"], busqueda["espectro_corto"])
            periodos_usados = (busqueda["periodo_largo"], busqueda["periodo_corto_usado"])
            for columna, ((periodos, potencia), usado) in enumerate(zip(espectros, periodos_usados)):
                eje = ejes[fila, columna]
                eje.plot(periodos, potencia, lw=1, color=COLOR_ESPECTRO)
                eje.axvline(usado, color=COLOR_PERIODO_USADO, ls="--", lw=1,
                            label="Periodo utilizado")
                periodo_libre = busqueda["periodo_corto_libre"]
                if columna == 1 and abs(periodo_libre - usado) > TOLERANCIA_PICOS_DISTINTOS_DIAS:
                    eje.axvline(periodo_libre, color=COLOR_PERIODO_LIBRE, ls=":", lw=1.3,
                                label="Máximo libre")
                    eje.legend(fontsize=8)
                eje.set(xlabel="Periodo (días)", ylabel="Reducción fraccional de chi2",
                        title=f"{nombre_serie}: {titulos[columna]}")
                if columna == 0:
                    eje.set_xscale("log")
        guardar_figura(figura, ruta)

    def _figura_ventana_muestreo(self, datos, ruta):
        figura, ejes = plt.subplots(len(datos), 1, figsize=(9, 6), sharex=True)
        for eje, (nombre_serie, serie) in zip(ejes, datos.items()):
            periodos, potencia = self._potencia_ventana_muestreo(serie)
            eje.plot(periodos, potencia, lw=1, color=COLOR_ESPECTRO)
            eje.axvline(365.25, color=COLOR_PERIODO_USADO, ls="--", label="1 año")
            eje.set(xlabel="Periodo de la ventana (días)", ylabel="Potencia",
                    title=nombre_serie)
            eje.legend()
        guardar_figura(figura, ruta)

    @staticmethod
    def _figura_ajuste_actividad(serie, ajuste, ruta):
        """S_MW con periodos compartidos y offsets instrumentales distintos (no GP)."""
        offset_por_medida = ajuste["coeficientes"][serie.indice_instrumento]
        actividad_sin_offset = serie.valores - offset_por_medida
        modelo_sin_offset = ajuste["modelo"] - offset_por_medida
        figura, (eje_modelo, eje_residuos) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for instrumento, mascara in serie.instrumentos():
            color, marcador = estilo_instrumento(instrumento)
            eje_modelo.scatter(serie.tiempos[mascara], actividad_sin_offset[mascara], s=12,
                               alpha=0.65, c=color, marker=marcador, label=instrumento)
            eje_residuos.scatter(serie.tiempos[mascara], ajuste["residuos"][mascara], s=8,
                                 alpha=0.5, c=color, marker=marcador)
        eje_modelo.plot(serie.tiempos, modelo_sin_offset, ".", ms=2, color=COLOR_MODELO,
                        label="Modelo común")
        eje_modelo.set_ylabel("S_MW - offset")
        eje_modelo.legend(fontsize=7)
        eje_residuos.axhline(0, c="black", ls="--", lw=0.8)
        eje_residuos.set(xlabel="BJD - 2450000 (días)", ylabel="Residuos S_MW")
        figura.suptitle("Actividad: ajustes sinusoidales globales (no GP)")
        guardar_figura(figura, ruta)
