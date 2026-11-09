"""Único lector de datos del proyecto: series RV y S_MW como objetos `Serie`."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config


@dataclass
class Serie:
    """Serie multiinstrumento ordenada por fecha. Tiempo: BJD - 2450000."""

    t: np.ndarray
    y: np.ndarray
    err: np.ndarray    # incertidumbre formal
    inst: np.ndarray   # nombre del instrumento de cada medida

    def __post_init__(self):
        if len(self.t) == 0:
            raise ValueError("La serie no contiene observaciones")
        if not all(np.isfinite(a).all() for a in (self.t, self.y, self.err)) \
                or (self.err <= 0).any():
            raise ValueError("Valores no finitos o incertidumbres no positivas")
        orden = np.argsort(self.t)
        self.t, self.y, self.err, self.inst = (
            a[orden] for a in (self.t, self.y, self.err, self.inst))
        self.nombres, self.indice = np.unique(self.inst, return_inverse=True)
        self.t0 = float(np.median(self.t))
        self.sigma = self._sigma_descriptivo()

    def _sigma_descriptivo(self):
        """Error formal + piso empírico por instrumento (MAD o 5 % de la std).

        Solo se usa en el análisis exploratorio; el GP bayesiano usa `err`.
        """
        pisos = np.empty(len(self.nombres))
        for j, (_, m) in enumerate(self.instrumentos()):
            v = self.y[m]
            mad = 1.4826 * np.median(np.abs(v - np.median(v)))
            pisos[j] = max(mad, 0.05 * np.std(v), 1e-10)
        return np.sqrt(self.err**2 + pisos[self.indice] ** 2)

    def instrumentos(self):
        """Itera (nombre, máscara) por instrumento."""
        for j, nombre in enumerate(self.nombres):
            yield str(nombre), self.indice == j

    def seleccionar(self, mascara):
        return Serie(self.t[mascara], self.y[mascara], self.err[mascara],
                     self.inst[mascara])

    def hash(self):
        """Firma de los datos, para comprobar que una reanudación usa los mismos."""
        h = hashlib.sha256()
        for a in (self.t, self.y, self.err, self.indice):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def leer_serie(ruta, actividad=False, excluir=None):
    """Lee un CSV de RV o de S_MW. `excluir` quita un instrumento por nombre."""
    with Path(ruta).open(newline="", encoding="utf-8-sig") as archivo:
        filas = list(csv.DictReader(archivo))
    factor = 1e-3 if actividad else 1.0
    campo = "S_MW x1000" if actividad else "RV (m/s)"
    campo_err = "sig S_MW x1000" if actividad else "err RV (m/s)"
    serie = Serie(
        t=np.array([float(f["BJD"]) - config.EPOCA_BJD for f in filas]),
        y=factor * np.array([float(f[campo]) for f in filas]),
        err=factor * np.array([float(f[campo_err]) for f in filas]),
        inst=np.array([f["Inst"] for f in filas]),
    )
    if excluir is not None:
        if excluir not in serie.nombres:
            raise ValueError(f"No existe el instrumento {excluir}")
        serie = serie.seleccionar(serie.inst != excluir)
    return serie


def leer_series(sin_apf=False):
    """Devuelve {"RV": Serie, "S_MW": Serie}. Lo reciben todos los análisis."""
    return {
        "RV": leer_serie(config.CSV_VELOCIDADES,
                         excluir=config.INSTRUMENTO_APF if sin_apf else None),
        "S_MW": leer_serie(config.CSV_ACTIVIDAD, actividad=True),
    }


def filtrar_sigma(serie, n_sigma=5.0):
    """Quita, por instrumento, las medidas a más de n_sigma (MAD) de su mediana."""
    conservar = np.ones(len(serie.t), dtype=bool)
    for _, m in serie.instrumentos():
        v = serie.y[m]
        dispersion = 1.4826 * np.median(np.abs(v - np.median(v)))
        conservar[m] = np.abs(v - np.median(v)) <= n_sigma * max(dispersion, 1e-12)
    return serie.seleccionar(conservar)
