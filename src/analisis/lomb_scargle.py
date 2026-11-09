"""Periodogramas Lomb-Scargle de RV y S_MW por instrumento.

Las FAP analíticas suponen ruido independiente; no prueban planetas.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle

from src.analisis.base import Analisis
from src.datos import filtrar_sigma
from src.utilidades import guardar_figura, guardar_tabla

NIVELES_FAP = (0.10, 0.01, 0.001)
COLOR_AZUL, COLOR_NARANJA = "steelblue", "darkorange"
PERIODO_MAXIMO_DIAS = 6000.0
PERIODO_MINIMO_DIAS = 2.0
MUESTRAS_POR_RESOLUCION_DE_FOURIER = 10
N_PICOS_PRINCIPALES = 5
MIN_MEDIDAS = 10
FRECUENCIAS_POR_LOTE = 200


class PeriodogramaLS(Analisis):
    @staticmethod
    def _calcular_ls(tiempos, valores, errores):
        """GLS ponderado, umbrales FAP y los cinco picos principales."""
        duracion = np.ptp(tiempos)
        if len(tiempos) < MIN_MEDIDAS or duracion <= PERIODO_MINIMO_DIAS:
            raise ValueError(f"necesita >= {MIN_MEDIDAS} medidas y "
                             f"> {PERIODO_MINIMO_DIAS:g} días de cobertura")
        periodo_maximo = min(PERIODO_MAXIMO_DIAS, duracion)
        n_frecuencias = int(np.ceil(
            MUESTRAS_POR_RESOLUCION_DE_FOURIER * duracion
            * (1 / PERIODO_MINIMO_DIAS - 1 / periodo_maximo))) + 1
        frecuencias = np.linspace(1 / periodo_maximo, 1 / PERIODO_MINIMO_DIAS,
                                  n_frecuencias)

        lomb_scargle = LombScargle(tiempos - tiempos.min(), valores, errores,
                                   fit_mean=True, center_data=True)
        potencia = lomb_scargle.power(frecuencias)
        umbrales_fap = [lomb_scargle.false_alarm_level(
            nivel, method="baluev", minimum_frequency=frecuencias[0],
            maximum_frequency=frecuencias[-1]) for nivel in NIVELES_FAP]

        maximos_locales = np.where((potencia[1:-1] > potencia[:-2]) &
                                   (potencia[1:-1] >= potencia[2:]))[0] + 1
        candidatos = np.unique(np.r_[maximos_locales, np.argmax(potencia)])
        candidatos = candidatos[np.argsort(potencia[candidatos])[::-1]]
        picos_elegidos = []
        for candidato in candidatos:
            if all(abs(frecuencias[candidato] - frecuencias[elegido]) >= 1 / duracion
                   for elegido in picos_elegidos):
                picos_elegidos.append(int(candidato))
            if len(picos_elegidos) == N_PICOS_PRINCIPALES:
                break
        return dict(
            frecuencias=frecuencias, potencia=potencia, umbrales_fap=umbrales_fap,
            n_medidas=len(tiempos), periodo_limite=periodo_maximo,
            picos=[(1 / frecuencias[indice_pico], potencia[indice_pico])
                   for indice_pico in picos_elegidos],
            pico_en_borde=bool(np.argmax(potencia) in (0, len(potencia) - 1)))

    @staticmethod
    def _potencia_ventana_muestreo(tiempos, frecuencias):
        """Ventana de muestreo: |media(exp(2*pi*i*f*t))|^2."""
        tiempos = tiempos - tiempos.min()
        potencia = np.empty(len(frecuencias))
        for inicio in range(0, len(frecuencias), FRECUENCIAS_POR_LOTE):
            lote = slice(inicio, inicio + FRECUENCIAS_POR_LOTE)
            fases = 2 * np.pi * frecuencias[lote, None] * tiempos[None, :]
            potencia[lote] = (np.mean(np.cos(fases), axis=1) ** 2
                              + np.mean(np.sin(fases), axis=1) ** 2)
        return potencia

    @staticmethod
    def _dibujar_periodogramas(curvas, titulo, ruta, marcar_rotacion=False):
        """`curvas` = [(resultado, etiqueta, color)]. Marca FAP 1 % y el pico máximo."""
        figura, eje = plt.subplots(figsize=(8, 4))
        for resultado, etiqueta, color in curvas:
            eje.plot(1 / resultado["frecuencias"], resultado["potencia"],
                     color=color, label=etiqueta)
            eje.axhline(resultado["umbrales_fap"][1], ls="--", lw=1, color=color,
                        label=f"FAP 1 % ({etiqueta})")
            eje.axvline(resultado["picos"][0][0], ls=":", lw=0.8, color=color)
        if marcar_rotacion:
            eje.axvspan(11.35, 11.55, color="gray", alpha=0.2,
                        label="Rotación (literatura)")
        eje.set(xscale="log",
                xlim=(PERIODO_MINIMO_DIAS,
                      min(resultado["periodo_limite"] for resultado, _, _ in curvas)),
                xlabel="Periodo (d)", ylabel="Potencia Lomb-Scargle", title=titulo)
        eje.legend(fontsize=8)
        guardar_figura(figura, ruta, formatos=(".png", ".pdf"), dpi=250)

    @staticmethod
    def _dibujar_ventana(tiempos, titulo, ruta):
        figura, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
        tramos = ((ejes[0], np.linspace(1 / 2000, 0.02, 1600), 1 / 365.25, "1 año"),
                  (ejes[1], np.linspace(0.02, 1.05, 4000), 1.0, "1 día"))
        for eje, frecuencias, frecuencia_referencia, etiqueta in tramos:
            eje.plot(frecuencias,
                     PeriodogramaLS._potencia_ventana_muestreo(tiempos, frecuencias),
                     color="black")
            eje.axvline(frecuencia_referencia, color=COLOR_NARANJA, ls="--",
                        label=etiqueta)
            eje.set(xlabel="Frecuencia (1/d)", ylabel="Potencia ventana")
            eje.legend(fontsize=8)
        figura.suptitle(titulo)
        guardar_figura(figura, ruta, formatos=(".png", ".pdf"), dpi=250)

    def ejecutar(self, datos, salida):
        series = {("RV", "Original"): datos["RV"],
                  ("S_MW", "Original"): datos["S_MW"],
                  ("S_MW", "Filtrada"): filtrar_sigma(datos["S_MW"])}
        resultados, filas_resumen, filas_picos = {}, [], []
        for (observable, seleccion), serie in series.items():
            for instrumento, mascara in serie.instrumentos():
                try:
                    resultado = self._calcular_ls(
                        serie.tiempos[mascara], serie.valores[mascara],
                        serie.errores_formales[mascara])
                except ValueError as error:
                    print(f"{observable:4} {instrumento:23} {seleccion:9} omitido: {error}")
                    continue
                resultados[(observable, seleccion, instrumento)] = resultado
                periodo_maximo, potencia_maxima = resultado["picos"][0]
                umbral_fap_1 = resultado["umbrales_fap"][1]
                supera_fap_1 = potencia_maxima > umbral_fap_1
                filas_resumen.append([
                    observable, seleccion, instrumento, resultado["n_medidas"],
                    periodo_maximo, potencia_maxima, *resultado["umbrales_fap"],
                    supera_fap_1, resultado["pico_en_borde"]])
                filas_picos += [
                    [observable, seleccion, instrumento, rango, periodo, potencia,
                     potencia > umbral_fap_1]
                    for rango, (periodo, potencia) in enumerate(resultado["picos"], start=1)]
                print(f"{observable:4} {instrumento:23} {seleccion:9} "
                      f"P = {periodo_maximo:9.3f} d | supera FAP 1 %: {supera_fap_1}")

        for instrumento in sorted({clave[2] for clave in resultados}):
            resultado_rv = resultados.get(("RV", "Original", instrumento))
            actividad_original = resultados.get(("S_MW", "Original", instrumento))
            actividad_filtrada = resultados.get(("S_MW", "Filtrada", instrumento))
            if resultado_rv:
                self._dibujar_periodogramas(
                    [(resultado_rv, "RV", COLOR_AZUL)], f"RV: {instrumento}",
                    salida / "RV" / instrumento)
            if actividad_original and actividad_filtrada:
                self._dibujar_periodogramas(
                    [(actividad_original, "Original", COLOR_AZUL),
                     (actividad_filtrada, "Filtrada a 5 sigma", COLOR_NARANJA)],
                    f"Actividad: {instrumento}", salida / "actividad" / instrumento)
            if resultado_rv and actividad_filtrada:
                self._dibujar_periodogramas(
                    [(resultado_rv, "RV", COLOR_AZUL),
                     (actividad_filtrada, "Actividad filtrada", COLOR_NARANJA)],
                    f"RV frente a actividad: {instrumento}",
                    salida / "RV_vs_actividad" / instrumento, marcar_rotacion=True)
        
        for observable in ("RV", "S_MW"):
            for instrumento, mascara in datos[observable].instrumentos():
                self._dibujar_ventana(datos[observable].tiempos[mascara],
                                      f"Ventana: {observable} - {instrumento}",
                                      salida / "ventanas" / observable / instrumento)

        guardar_tabla(salida / "resumen_ls.csv",
                      ["serie", "seleccion", "instrumento", "N", "periodo_maximo_d",
                       "potencia_maxima", "FAP_10", "FAP_1", "FAP_0_1",
                       "supera_FAP_1", "maximo_en_borde"], filas_resumen)
        guardar_tabla(salida / "picos_ls.csv",
                      ["serie", "seleccion", "instrumento", "rango", "periodo_d",
                       "potencia", "supera_umbral_FAP_1"], filas_picos)
        print(f"\nResultados guardados en {salida}/")
