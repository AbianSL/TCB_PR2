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
        print(f"RV: {len(serie.t)} observaciones; modelos: {', '.join(self.args.modelos)}")
        print(f"Carpeta de esta ejecución: {base}")

        evidencias = {}
        for nombre in self.args.modelos:
            evidencias[nombre] = self._ajustar(serie, nombre, base, nlive, dlogz)
            _registrar_evidencias(base, evidencias)
        print(f"\nTERMINADO. Resultados en:\n{base}")
        if "M0" in evidencias and "M1" in evidencias:
            print(f"Delta ln Z (M1-M0) = {evidencias['M1'][0] - evidencias['M0'][0]:.3f}")

    # ---------- preparación ----------
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
        a = self.args
        configuracion = dict(
            hash_datos=serie.hash(), instrumentos=serie.nombres.tolist(),
            nlive=int(nlive), dlogz=float(dlogz), semilla=int(a.semilla),
            priors_ampliados=bool(a.priors_ampliados), excluir_APF=bool(a.sin_apf),
            modelos=list(a.modelos),
            versiones={p: version(p) for p in config.PAQUETES_REGISTRADOS})
        ruta = base / "configuracion.json"
        if ruta.exists():
            if json.loads(ruta.read_text(encoding="utf-8")) != configuracion:
                raise ValueError("Configuración distinta a la original: no se puede reanudar")
        else:
            ruta.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    def _comprobar(self, serie):
        """Evalúa las tres verosimilitudes sin muestrear."""
        print(f"RV: {len(serie.t)} observaciones, {len(serie.nombres)} instrumentos")
        for nombre, clase in MODELOS.items():
            modelo = clase()
            espacio = modelo.espacio_parametros(serie.nombres)
            lnL = VerosimilitudGP(serie, modelo, espacio)(espacio(np.full(len(espacio), 0.5)))
            if not np.isfinite(lnL):
                raise RuntimeError(f"Verosimilitud no finita para {nombre}")
            print(f"{nombre}: lnL = {lnL:.3f}")
        print("OK: todos los modelos se pueden calcular desde los CSV.")

    # ---------- un modelo ----------
    def _ajustar(self, serie, nombre, base, nlive, dlogz):
        modelo = MODELOS[nombre]()
        espacio = modelo.espacio_parametros(serie.nombres, self.args.priors_ampliados)
        if nlive < 2 * len(espacio):
            raise ValueError(f"nlive debe ser al menos {2 * len(espacio)} para {nombre}")
        carpeta = base / nombre
        carpeta.mkdir(exist_ok=True)

        # evidencia.csv marca un modelo COMPLETO: se escribe al final.
        ruta_evidencia = carpeta / "evidencia.csv"
        if ruta_evidencia.exists():
            with ruta_evidencia.open(encoding="utf-8", newline="") as f:
                fila = next(csv.DictReader(f))
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

        res = sampler.results
        pesos = np.exp(res.logwt - logsumexp(res.logwt))
        muestras = resample_equal(res.samples, pesos,
                                  rstate=np.random.default_rng(self.args.semilla + 1))
        p16, p50, p84 = np.percentile(muestras, [16, 50, 84], axis=0)
        guardar_tabla(carpeta / "parametros.csv", ["parametro", "p16", "p50", "p84"],
                      [[n, p16[i], p50[i], p84[i]] for i, n in enumerate(espacio.nombres)])
        np.savez_compressed(carpeta / "muestras.npz", theta=res.samples, logwt=res.logwt,
                            logl=res.logl, muestras=muestras, logz=res.logz,
                            logzerr=res.logzerr)
        _dibujar_posterior(carpeta, modelo, espacio, muestras)

        # Si la mediana conjunta es numéricamente inestable, se usa la mejor muestra.
        componentes = verosimilitud.evaluar(p50, componentes=True) \
            or verosimilitud.evaluar(res.samples[np.argmax(res.logl)], componentes=True)
        if componentes is None:
            raise RuntimeError(f"No se pudo representar el GP para {nombre}")
        _dibujar_componentes(serie, componentes, carpeta, nombre)

        lnZ, lnZerr = float(res.logz[-1]), float(res.logzerr[-1])
        guardar_tabla(ruta_evidencia, ["modelo", "lnZ", "lnZerr"], [[nombre, lnZ, lnZerr]])
        print(f"{nombre}: ln Z = {lnZ:.3f} +/- {lnZerr:.3f}")
        return lnZ, lnZerr


# ---------- figuras y tablas ----------
def _dibujar_componentes(serie, c, carpeta, nombre):
    """RV sin offsets, tendencia + órbita, GP condicionado y residuos."""
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    centrada = serie.y - c["offsets"][serie.indice]
    for inst, m in serie.instrumentos():
        ax[0].scatter(serie.t[m], centrada[m], s=8, alpha=.45, label=inst)
    modelo = c["tendencia"] + c["orbita"]
    ax[0].plot(serie.t, modelo, ".", color="tab:red", ms=2, label="Tendencia + órbita")
    ax[0].plot(serie.t, modelo + c["gp"], ".", color="black", ms=2,
               label="GP condicionado (no es predicción independiente)")
    ax[0].set(ylabel="RV - offset (m/s)", title=f"{nombre}: GP + modelo orbital (medianas)")
    ax[0].legend(fontsize=7, ncol=3)
    ax[1].scatter(serie.t, c["residuos"], s=8, alpha=.5)
    ax[1].axhline(0, color="black", ls="--", lw=.8)
    ax[1].set(xlabel="BJD - 2450000 (d)", ylabel="Residuos (m/s)")
    guardar_figura(fig, carpeta / f"{nombre}_componentes", dpi=160)


def _dibujar_posterior(carpeta, modelo, espacio, muestras):
    columnas = modelo.parametros_posterior
    fig, ejes = plt.subplots(1, len(columnas), figsize=(4 * len(columnas), 3))
    for eje, nombre in zip(np.atleast_1d(ejes), columnas):
        valores = muestras[:, espacio.nombres.index(nombre)]
        eje.hist(valores, bins=45, color="steelblue", alpha=0.65)
        eje.axvline(np.median(valores), color="black", ls="--")
        eje.set(xlabel=nombre.replace("_", " "), ylabel="Muestras")
    fig.suptitle(f"{modelo.nombre}: posterior de esta ejecución")
    guardar_figura(fig, carpeta / f"{modelo.nombre}_posteriores")


def _registrar_evidencias(base, evidencias):
    """Tabla y figura de ln Z; comparación válida solo con los mismos datos y priors."""
    origen = evidencias["M0"][0] if "M0" in evidencias else next(iter(evidencias.values()))[0]
    guardar_tabla(base / "comparacion_evidencias.csv",
                  ["modelo", "lnZ", "lnZerr", "delta_lnZ_vs_M0"],
                  [[m, z, e, z - evidencias["M0"][0] if "M0" in evidencias else ""]
                   for m, (z, e) in evidencias.items()])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(list(evidencias), [z - origen for z, _ in evidencias.values()],
                yerr=[e for _, e in evidencias.values()], fmt="o", capsize=4)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set(xlabel="Modelo", ylabel=r"$\ln Z - \ln Z_\mathrm{M0}$",
           title="Evidencias calculadas desde cero")
    guardar_figura(fig, base / "comparacion_evidencias")
