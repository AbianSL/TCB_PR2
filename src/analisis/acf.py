"""DCF (autocorrelación irregular) de RV y S_MW con test de permutación.

Las bandas y los valores p contrastan ausencia de correlación temporal
residual; NO son significancias frente a ruido estelar correlacionado.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analisis.base import Analisis
from src.estilos import COLOR_BANDA_NULA, estilo_serie
from src.lineal import ajustar_lineal, periodograma
from src.utilidades import guardar_figura, guardar_tabla


class ACF(Analisis):
    ANCHO_BIN_DIAS = 2.0  
    RETARDO_MAXIMO_DIAS = 100.0 
    RETARDO_MINIMO_BUSQUEDA_DIAS = 5.0
    MIN_PARES_POR_BIN = 100 
    QUITAR_PERIODO_LARGO = True
    N_PERMUTACIONES = 1000

    def ejecutar(self, datos, salida):
        generador = np.random.default_rng(self.args.semilla)
        resultados = {}
        for nombre_serie, serie in datos.items():
            print(f"{nombre_serie}: calculando DCF y {self.N_PERMUTACIONES} permutaciones")
            resultado = self._dcf_con_permutaciones(serie, generador)
            print(f"  Pico global: {resultado['retardo_pico']:.1f} dias; "
                  f"p_global = {resultado['p_global']:.4f}")
            resultados[nombre_serie] = resultado
        self._guardar(datos, resultados, salida)

    # ---------- cálculo ----------
    def _agrupar_pares_por_retardo(self, tiempos):
        """Agrupa los pares (a < b) por retardo; se reutilizan en las permutaciones."""
        tiempos = np.asarray(tiempos)
        fin_ventana = np.searchsorted(tiempos, tiempos + self.RETARDO_MAXIMO_DIAS,
                                      side="right")
        pares_por_punto = np.maximum(fin_ventana - np.arange(len(tiempos)) - 1, 0)
        primero = np.repeat(np.arange(len(tiempos), dtype=np.int32), pares_por_punto)
        segundo = np.concatenate([np.arange(punto + 1, fin_ventana[punto], dtype=np.int32)
                                  for punto in np.flatnonzero(pares_por_punto)])
        bin_del_par = ((tiempos[segundo] - tiempos[primero]) / self.ANCHO_BIN_DIAS
                       ).astype(np.int32)
        n_bins = int(np.ceil(self.RETARDO_MAXIMO_DIAS / self.ANCHO_BIN_DIAS))
        valido = (tiempos[segundo] > tiempos[primero]) & (bin_del_par < n_bins)
        primero, segundo, bin_del_par = primero[valido], segundo[valido], bin_del_par[valido]
        retardos = (np.arange(n_bins) + 0.5) * self.ANCHO_BIN_DIAS
        return retardos, primero, segundo, bin_del_par, np.bincount(bin_del_par,
                                                                    minlength=n_bins)

    def _calcular_dcf(self, valores, primero, segundo, bin_del_par, pares_por_bin):
        """Media de productos de desviaciones normalizadas en cada bin."""
        desviaciones = valores - np.mean(valores)
        varianza = np.mean(desviaciones ** 2)
        if varianza <= 0:
            raise ValueError("Serie sin variación: no se puede calcular la DCF")
        suma_productos = np.bincount(
            bin_del_par, weights=desviaciones[primero] * desviaciones[segundo],
            minlength=len(pares_por_bin))
        return np.divide(suma_productos, pares_por_bin * varianza,
                         out=np.full(len(pares_por_bin), np.nan),
                         where=pares_por_bin >= self.MIN_PARES_POR_BIN)

    def _dcf_con_permutaciones(self, serie, generador):
        """DCF observada y su test de permutación (mismos pares y fechas)."""
        periodos_a_quitar = ()
        if self.QUITAR_PERIODO_LARGO:
            periodo_largo = periodograma(serie, 200, 6000)[-1]
            periodos_a_quitar = (periodo_largo,)
            print(f"  Periodo largo eliminado: {periodo_largo:.1f} dias")

        residuos = ajustar_lineal(serie, periodos_a_quitar)["residuos"]
        retardos, primero, segundo, bin_del_par, pares_por_bin = \
            self._agrupar_pares_por_retardo(serie.tiempos)
        dcf_observada = self._calcular_dcf(residuos, primero, segundo, bin_del_par,
                                           pares_por_bin)

        # Hipótesis nula: se conservan fechas e instrumentos y se mezclan solo
        # los residuos de un mismo instrumento.
        indices_por_instrumento = [np.flatnonzero(mascara)
                                   for _, mascara in serie.instrumentos()]
        dcf_nulas = np.empty((self.N_PERMUTACIONES, len(retardos)))
        for numero_permutacion in range(self.N_PERMUTACIONES):
            residuos_mezclados = residuos.copy()
            for indices in indices_por_instrumento:
                residuos_mezclados[indices] = residuos[generador.permutation(indices)]
            dcf_nulas[numero_permutacion] = self._calcular_dcf(residuos_mezclados, primero, segundo,
                                              bin_del_par, pares_por_bin)

        bin_representable = pares_por_bin >= self.MIN_PARES_POR_BIN
        banda_nula = np.full((2, len(retardos)), np.nan)  # filas: 2.5 % y 97.5 %
        banda_nula[:, bin_representable] = np.percentile(
            dcf_nulas[:, bin_representable], [2.5, 97.5], axis=0)

        # Prueba global: máximo positivo buscado entre 5 d y el retardo máximo.
        region_busqueda = (bin_representable
                           & (retardos >= self.RETARDO_MINIMO_BUSQUEDA_DIAS)
                           & (retardos < self.RETARDO_MAXIMO_DIAS))
        if not region_busqueda.any():
            raise ValueError(f"Ningún bin entre {self.RETARDO_MINIMO_BUSQUEDA_DIAS:g} y "
                             f"{self.RETARDO_MAXIMO_DIAS:g} d tiene >= "
                             f"{self.MIN_PARES_POR_BIN} pares")
        maximo_observado = np.nanmax(dcf_observada[region_busqueda])
        maximos_nulos = np.nanmax(dcf_nulas[:, region_busqueda], axis=1)
        p_global = (1 + np.count_nonzero(maximos_nulos >= maximo_observado)) \
            / (self.N_PERMUTACIONES + 1)
        retardo_pico = retardos[region_busqueda][np.nanargmax(dcf_observada[region_busqueda])]
        return dict(retardos=retardos, dcf=dcf_observada, pares_por_bin=pares_por_bin,
                    banda_nula=banda_nula, retardo_pico=float(retardo_pico),
                    p_global=float(p_global))

    def _guardar(self, datos, resultados, salida):
        figura, ejes = plt.subplots(len(datos), 1, figsize=(10, 7), sharex=True)
        filas_dcf, filas_resumen = [], []
        for eje, (nombre_serie, resultado) in zip(ejes, resultados.items()):
            color, marcador = estilo_serie(nombre_serie)
            filas_resumen.append([nombre_serie, resultado["retardo_pico"], resultado["p_global"],
                                  self.N_PERMUTACIONES])

            filas_dcf.append([nombre_serie, 0.0, 1.0, len(datos[nombre_serie].tiempos), "", ""])
            filas_dcf += [[nombre_serie, *fila] for fila in
                          zip(resultado["retardos"], resultado["dcf"], resultado["pares_por_bin"], *resultado["banda_nula"])]
            
            eje.fill_between(resultado["retardos"], *resultado["banda_nula"], color=COLOR_BANDA_NULA,
                             alpha=0.3, label="Permutaciones: banda puntual del 95 %")
            eje.plot(np.r_[0, resultado["retardos"]], np.r_[1, resultado["dcf"]], ls="-", marker=marcador,
                     color=color, ms=3.5, lw=1.2, label="DCF observada")
            eje.axhline(0, color="black", lw=0.8, ls="--")
            eje.set_ylabel(f"DCF {nombre_serie}")
            eje.grid(alpha=0.2)
            eje.legend(fontsize=8)
        
        ejes[-1].set(xlabel="Retardo temporal (dias)", xlim=(0, self.RETARDO_MAXIMO_DIAS))
        figura.suptitle("Autocorrelacion irregular (DCF) y bandas de permutacion")
        guardar_figura(figura, salida / "acf", dpi=180)
        guardar_tabla(salida / "acf.csv",
                      ["serie", "lag_d", "dcf", "n_pares", "banda_2_5", "banda_97_5"],
                      filas_dcf)
        guardar_tabla(salida / "resumen_permutaciones.csv",
                      ["serie", "pico_global_5_100_d", "p_global", "n_permutaciones"],
                      filas_resumen)
        print(f"Resultados: {salida / 'acf.png'}, acf.csv y resumen_permutaciones.csv")
        print("Aviso: las permutaciones no conservan el ruido estelar correlacionado.")
