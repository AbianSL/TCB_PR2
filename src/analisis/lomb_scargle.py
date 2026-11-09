"""Periodogramas Lomb-Scargle de RV y S_MW por instrumento.

Las FAP analíticas suponen ruido independiente; no prueban planetas.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle

from src.analisis.base import Analisis
from src.datos import filtrar_sigma
from src.utilidades import guardar_figura, guardar_tabla

FAP = (0.10, 0.01, 0.001)
AZUL, NARANJA = "steelblue", "darkorange"


class PeriodogramaLS(Analisis):
    @staticmethod
    def _calcular_ls(t, y, dy):
        """GLS ponderado, umbrales FAP y los cinco picos principales."""
        if len(t) < 10 or np.ptp(t) <= 2:
            raise ValueError("necesita >= 10 medidas y > 2 días de cobertura")
        duracion = np.ptp(t)
        pmax = min(6000.0, duracion)
        # Diez muestras por resolución de Fourier (1/duración).
        nfreq = int(np.ceil(10 * duracion * (1 / 2 - 1 / pmax))) + 1
        f = np.linspace(1 / pmax, 1 / 2, nfreq)

        ls = LombScargle(t - t.min(), y, dy, fit_mean=True, center_data=True)
        potencia = ls.power(f)
        niveles = [ls.false_alarm_level(fap, method="baluev",
                                        minimum_frequency=f[0], maximum_frequency=f[-1])
                   for fap in FAP]

        # Máximos locales, sin repetir el mismo pico por sobremuestreo.
        candidatos = np.where((potencia[1:-1] > potencia[:-2]) &
                              (potencia[1:-1] >= potencia[2:]))[0] + 1
        candidatos = np.unique(np.r_[candidatos, np.argmax(potencia)])
        candidatos = candidatos[np.argsort(potencia[candidatos])[::-1]]
        mejores = []
        for i in candidatos:
            if all(abs(f[i] - f[j]) >= 1 / duracion for j in mejores):
                mejores.append(int(i))
            if len(mejores) == 5:
                break
        return dict(f=f, potencia=potencia, niveles=niveles, n=len(t), periodo_limite=pmax,
                    picos=[(1 / f[i], potencia[i]) for i in mejores],
                    pico_en_borde=bool(np.argmax(potencia) in (0, len(potencia) - 1)))

    @staticmethod
    def _ventana_espectral(t, frecuencias):
        """Ventana de muestreo: |media(exp(2*pi*i*f*t))|^2."""
        t = t - t.min()
        potencia = np.empty(len(frecuencias))
        for inicio in range(0, len(frecuencias), 200):
            fases = 2 * np.pi * frecuencias[inicio:inicio + 200, None] * t[None, :]
            potencia[inicio:inicio + 200] = (np.mean(np.cos(fases), axis=1) ** 2
                                             + np.mean(np.sin(fases), axis=1) ** 2)
        return potencia

    @staticmethod
    def _dibujar_periodogramas(curvas, titulo, ruta, rotacion=False):
        """`curvas` = [(resultado, etiqueta, color)]. Marca FAP 1 % y el pico máximo."""
        fig, ax = plt.subplots(figsize=(8, 4))
        for res, etiqueta, color in curvas:
            ax.plot(1 / res["f"], res["potencia"], color=color, label=etiqueta)
            ax.axhline(res["niveles"][1], ls="--", lw=1, color=color,
                       label=f"FAP 1 % ({etiqueta})")
            ax.axvline(res["picos"][0][0], ls=":", lw=0.8, color=color)
        if rotacion:
            ax.axvspan(11.35, 11.55, color="gray", alpha=0.2, label="Rotación (literatura)")
        ax.set(xscale="log", xlim=(2, min(r["periodo_limite"] for r, _, _ in curvas)),
               xlabel="Periodo (d)", ylabel="Potencia Lomb-Scargle", title=titulo)
        ax.legend(fontsize=8)
        guardar_figura(fig, ruta, formatos=(".png", ".pdf"), dpi=250)

    @staticmethod
    def _dibujar_ventana(t, titulo, ruta):
        fig, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
        for eje, f, ref, texto in ((ejes[0], np.linspace(1 / 2000, 0.02, 1600), 1 / 365.25, "1 año"),
                                   (ejes[1], np.linspace(0.02, 1.05, 4000), 1.0, "1 día")):
            eje.plot(f, PeriodogramaLS._ventana_espectral(t, f), color="black")
            eje.axvline(ref, color=NARANJA, ls="--", label=texto)
            eje.set(xlabel="Frecuencia (1/d)", ylabel="Potencia ventana")
            eje.legend(fontsize=8)
        fig.suptitle(titulo)
        guardar_figura(fig, ruta, formatos=(".png", ".pdf"), dpi=250)

    def ejecutar(self, datos, salida):
        series = {("RV", "Original"): datos["RV"],
                  ("S_MW", "Original"): datos["S_MW"],
                  ("S_MW", "Filtrada"): filtrar_sigma(datos["S_MW"])}
        resultados, resumen, picos = {}, [], []
        for (obs, sel), serie in series.items():
            for inst, m in serie.instrumentos():
                try:
                    res = self._calcular_ls(serie.t[m], serie.y[m], serie.err[m])
                except ValueError as error:
                    print(f"{obs:4} {inst:23} {sel:9} omitido: {error}")
                    continue
                resultados[(obs, sel, inst)] = res
                periodo, potencia = res["picos"][0]
                supera = potencia > res["niveles"][1]
                resumen.append([obs, sel, inst, res["n"], periodo, potencia,
                                *res["niveles"], supera, res["pico_en_borde"]])
                picos += [[obs, sel, inst, rango, p, a, a > res["niveles"][1]]
                          for rango, (p, a) in enumerate(res["picos"], start=1)]
                print(f"{obs:4} {inst:23} {sel:9} P = {periodo:9.3f} d | "
                      f"supera FAP 1 %: {supera}")

        # Figuras: RV, actividad original vs filtrada, y RV vs actividad filtrada.
        for inst in sorted({clave[2] for clave in resultados}):
            rv = resultados.get(("RV", "Original", inst))
            orig = resultados.get(("S_MW", "Original", inst))
            filt = resultados.get(("S_MW", "Filtrada", inst))
            if rv:
                self._dibujar_periodogramas([(rv, "RV", AZUL)], f"RV: {inst}",
                                      salida / "RV" / inst)
            if orig and filt:
                self._dibujar_periodogramas([(orig, "Original", AZUL),
                                       (filt, "Filtrada a 5 sigma", NARANJA)],
                                      f"Actividad: {inst}", salida / "actividad" / inst)
            if rv and filt:
                self._dibujar_periodogramas([(rv, "RV", AZUL), (filt, "Actividad filtrada", NARANJA)],
                                      f"RV frente a actividad: {inst}",
                                      salida / "RV_vs_actividad" / inst, rotacion=True)
        # La ventana depende solo de los instantes, no de los valores.
        for obs in ("RV", "S_MW"):
            for inst, m in datos[obs].instrumentos():
                self._dibujar_ventana(datos[obs].t[m], f"Ventana: {obs} - {inst}",
                                salida / "ventanas" / obs / inst)

        guardar_tabla(salida / "resumen_ls.csv",
                      ["serie", "seleccion", "instrumento", "N", "periodo_maximo_d",
                       "potencia_maxima", "FAP_10", "FAP_1", "FAP_0_1",
                       "supera_FAP_1", "maximo_en_borde"], resumen)
        guardar_tabla(salida / "picos_ls.csv",
                      ["serie", "seleccion", "instrumento", "rango", "periodo_d",
                       "potencia", "supera_umbral_FAP_1"], picos)
        print(f"\nResultados guardados en {salida}/")
