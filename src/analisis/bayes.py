"""Ajuste bayesiano de RV: GP + Nested Sampling (dynesty) para M0, M1 y M2."""

import csv
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import dynesty
import matplotlib.pyplot as plt
import numpy as np
from dynesty.utils import resample_equal
from scipy.special import logsumexp

from src import config
from src.analisis.base import Analisis
from src.modelos import MODELOS
from src.modelos.verosimilitud import VerosimilitudGP
from src.utilidades import guardar_figura, guardar_tabla


class Bayes(Analisis):
    def ejecutar(self, datos, salida):
        serie = datos["RV"]
        if self.args.comprobar:
            return self._comprobar(serie)
        base = self._carpeta(salida)
        nlive, dlogz = self._muestreo()
        self._guardar_configuracion(base, serie, nlive, dlogz)
        print(f"RV: {len(serie.tiempos)} observaciones; modelos: {', '.join(self.args.modelos)}")
        print(f"Carpeta de esta ejecución: {base}")

        evidencias = {}
        for nombre in self.args.modelos:
            evidencias[nombre] = self._ajustar(serie, nombre, base, nlive, dlogz)
            _registrar_evidencias(base, evidencias)
        print(f"\nTERMINADO. Resultados en:\n{base}")
        if "M0" in evidencias and "M1" in evidencias:
            print(f"Delta ln Z (M1-M0) = {evidencias['M1'][0] - evidencias['M0'][0]:.3f}")

    def _muestreo(self):
        nlive, dlogz = config.MUESTREO[self.args.modo]
        if self.args.nlive is not None:
            nlive = self.args.nlive
        if self.args.dlogz is not None:
            dlogz = self.args.dlogz
        return nlive, dlogz

    def _carpeta(self, salida):
        """Carpeta nueva por ejecución (nunca reutiliza resultados), o la de reanudación."""
        if self.args.reanudar:
            base = Path(self.args.reanudar)
            base = (base if base.is_absolute() else config.RAIZ / base).resolve()
            if not base.is_dir():
                raise FileNotFoundError(f"No existe carpeta para reanudar: {base}")
            return base
        base = salida / datetime.now().strftime("ajuste_%Y%m%d_%H%M%S_%f")
        base.mkdir(parents=True)
        return base

    def _guardar_configuracion(self, base, serie, nlive, dlogz):
        argumentos = self.args
        configuracion = dict(
            hash_datos=serie.hash(), instrumentos=serie.nombres_instrumentos.tolist(),
            nlive=int(nlive), dlogz=float(dlogz), semilla=int(argumentos.semilla),
            priors_ampliados=bool(argumentos.priors_ampliados),
            excluir_APF=bool(argumentos.sin_apf),
            modelos=list(argumentos.modelos),
            versiones={paquete: version(paquete)
                       for paquete in config.PAQUETES_REGISTRADOS})
        ruta = base / "configuracion.json"
        if ruta.exists():
            if json.loads(ruta.read_text(encoding="utf-8")) != configuracion:
                raise ValueError("Configuración distinta a la original: no se puede reanudar")
        else:
            ruta.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    def _comprobar(self, serie):
        """Evalúa las tres verosimilitudes sin muestrear."""
        print(f"RV: {len(serie.tiempos)} observaciones, {len(serie.nombres_instrumentos)} instrumentos")
        for nombre, clase in MODELOS.items():
            modelo = clase()
            espacio = modelo.espacio_parametros(serie.nombres_instrumentos)
            ln_verosimilitud = VerosimilitudGP(serie, modelo, espacio)(
                espacio(np.full(len(espacio), 0.5)))
            if not np.isfinite(ln_verosimilitud):
                raise RuntimeError(f"Verosimilitud no finita para {nombre}")
            print(f"{nombre}: lnL = {ln_verosimilitud:.3f}")
        print("OK: todos los modelos se pueden calcular desde los CSV.")

    def _ajustar(self, serie, nombre, base, nlive, dlogz):
        modelo = MODELOS[nombre]()
        espacio = modelo.espacio_parametros(serie.nombres_instrumentos,
                                            self.args.priors_ampliados)
        if nlive < 2 * len(espacio):
            raise ValueError(f"nlive debe ser al menos {2 * len(espacio)} para {nombre}")
        carpeta = base / nombre
        carpeta.mkdir(exist_ok=True)

        ruta_evidencia = carpeta / "evidencia.csv"
        if ruta_evidencia.exists():
            with ruta_evidencia.open(encoding="utf-8", newline="") as archivo:
                fila = next(csv.DictReader(archivo))
            print(f"{nombre}: ya terminado; se conserva")
            return float(fila["lnZ"]), float(fila["lnZerr"])

        verosimilitud = VerosimilitudGP(serie, modelo, espacio)
        checkpoint = carpeta / "checkpoint.dynesty"
        reanudar = checkpoint.exists()
        if reanudar:
            print(f"{nombre}: restaurando checkpoint")
            sampler = dynesty.NestedSampler.restore(str(checkpoint))
        else:
            print(f"{nombre}: Nested Sampling ({len(espacio)} parámetros, "
                  f"nlive={nlive}, dlogz={dlogz})")
            sampler = dynesty.NestedSampler(
                verosimilitud, espacio, ndim=len(espacio), nlive=nlive,
                sample="rwalk", bound="multi",
                rstate=np.random.default_rng(self.args.semilla))
        sampler.run_nested(dlogz=dlogz, checkpoint_file=str(checkpoint),
                           checkpoint_every=120, resume=reanudar, print_progress=True)

        resultados = sampler.results
        pesos = np.exp(resultados.logwt - logsumexp(resultados.logwt))
        muestras = resample_equal(resultados.samples, pesos,
                                  rstate=np.random.default_rng(self.args.semilla + 1))
        percentil_16, mediana, percentil_84 = np.percentile(muestras, [16, 50, 84], axis=0)
        guardar_tabla(carpeta / "parametros.csv", ["parametro", "p16", "p50", "p84"],
                      [[nombre_parametro, percentil_16[posicion], mediana[posicion],
                        percentil_84[posicion]]
                       for posicion, nombre_parametro in enumerate(espacio.nombres)])
        np.savez_compressed(carpeta / "muestras.npz", theta=resultados.samples,
                            logwt=resultados.logwt, logl=resultados.logl,
                            muestras=muestras, logz=resultados.logz,
                            logzerr=resultados.logzerr)
        _dibujar_posterior(carpeta, modelo, espacio, muestras)

        componentes = verosimilitud.evaluar(mediana, componentes=True) \
            or verosimilitud.evaluar(resultados.samples[np.argmax(resultados.logl)],
                                     componentes=True)
        if componentes is None:
            raise RuntimeError(f"No se pudo representar el GP para {nombre}")
        _dibujar_componentes(serie, componentes, carpeta, nombre)

        ln_evidencia = float(resultados.logz[-1])
        error_ln_evidencia = float(resultados.logzerr[-1])
        guardar_tabla(ruta_evidencia, ["modelo", "lnZ", "lnZerr"], [[nombre, ln_evidencia, error_ln_evidencia]])
        print(f"{nombre}: ln Z = {ln_evidencia:.3f} +/- {error_ln_evidencia:.3f}")
        return ln_evidencia, error_ln_evidencia


def _dibujar_componentes(serie, componentes, carpeta, nombre_modelo):
    """RV sin offsets, tendencia + órbita, GP condicionado y residuos."""
    figura, (eje_rv, eje_residuos) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    rv_sin_offset = serie.valores - componentes["offsets"][serie.indice_instrumento]
    for instrumento, mascara in serie.instrumentos():
        eje_rv.scatter(serie.tiempos[mascara], rv_sin_offset[mascara], s=8, alpha=.45,
                       label=instrumento)
    tendencia_mas_orbita = componentes["tendencia"] + componentes["orbita"]
    eje_rv.plot(serie.tiempos, tendencia_mas_orbita, ".", color="tab:red", ms=2,
                label="Tendencia + órbita")
    eje_rv.plot(serie.tiempos, tendencia_mas_orbita + componentes["gp"], ".",
                color="black", ms=2,
                label="GP condicionado (no es predicción independiente)")
    eje_rv.set(ylabel="RV - offset (m/s)",
               title=f"{nombre_modelo}: GP + modelo orbital (medianas)")
    eje_rv.legend(fontsize=7, ncol=3)
    eje_residuos.scatter(serie.tiempos, componentes["residuos"], s=8, alpha=.5)
    eje_residuos.axhline(0, color="black", ls="--", lw=.8)
    eje_residuos.set(xlabel="BJD - 2450000 (d)", ylabel="Residuos (m/s)")
    guardar_figura(figura, carpeta / f"{nombre_modelo}_componentes", dpi=160)


def _dibujar_posterior(carpeta, modelo, espacio, muestras):
    columnas = modelo.parametros_posterior
    figura, ejes = plt.subplots(1, len(columnas), figsize=(4 * len(columnas), 3))
    for eje, nombre in zip(np.atleast_1d(ejes), columnas):
        valores = muestras[:, espacio.nombres.index(nombre)]
        eje.hist(valores, bins=45, color="steelblue", alpha=0.65)
        eje.axvline(np.median(valores), color="black", ls="--")
        eje.set(xlabel=nombre.replace("_", " "), ylabel="Muestras")
    figura.suptitle(f"{modelo.nombre}: posterior de esta ejecución")
    guardar_figura(figura, carpeta / f"{modelo.nombre}_posteriores")


def _registrar_evidencias(base, evidencias):
    """Tabla y figura de ln Z; comparación válida solo con los mismos datos y priors."""
    ln_z_referencia = (evidencias["M0"][0] if "M0" in evidencias
                       else next(iter(evidencias.values()))[0])
    guardar_tabla(base / "comparacion_evidencias.csv",
                  ["modelo", "lnZ", "lnZerr", "delta_lnZ_vs_M0"],
                  [[nombre_modelo, ln_z, error_ln_z,
                    ln_z - evidencias["M0"][0] if "M0" in evidencias else ""]
                   for nombre_modelo, (ln_z, error_ln_z) in evidencias.items()])
    figura, eje = plt.subplots(figsize=(6.5, 4))
    eje.errorbar(list(evidencias),
                 [ln_z - ln_z_referencia for ln_z, _ in evidencias.values()],
                 yerr=[error_ln_z for _, error_ln_z in evidencias.values()],
                 fmt="o", capsize=4)
    eje.axhline(0, color="black", ls="--", lw=0.8)
    eje.set(xlabel="Modelo", ylabel=r"$\ln Z - \ln Z_\mathrm{M0}$",
            title="Evidencias calculadas desde cero")
    guardar_figura(figura, base / "comparacion_evidencias")
