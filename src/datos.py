"""Único lector de datos del proyecto: series RV y S_MW como objetos `Serie`."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config

FACTOR_ACTIVIDAD = 1e-3  # la columna "S_MW x1000" se pasa a unidades de S_MW


@dataclass
class Serie:
    """Serie multiinstrumento ordenada por fecha. Tiempo: BJD - 2450000."""

    tiempos: np.ndarray
    valores: np.ndarray
    errores_formales: np.ndarray
    instrumento_por_medida: np.ndarray 

    def __post_init__(self):
        if len(self.tiempos) == 0:
            raise ValueError("La serie no contiene observaciones")
        if (not all(np.isfinite(columna).all() for columna in
                    (self.tiempos, self.valores, self.errores_formales))
                or (self.errores_formales <= 0).any()):
            raise ValueError("Valores no finitos o incertidumbres no positivas")
        orden_temporal = np.argsort(self.tiempos)
        self.tiempos, self.valores, self.errores_formales, self.instrumento_por_medida = (
            columna[orden_temporal] for columna in
            (self.tiempos, self.valores, self.errores_formales,
             self.instrumento_por_medida))
        self.nombres_instrumentos, self.indice_instrumento = np.unique(
            self.instrumento_por_medida, return_inverse=True)
        self.tiempo_referencia = float(np.median(self.tiempos))
        self.sigma_descriptivo = self._calcular_sigma_descriptivo()

    def _calcular_sigma_descriptivo(self):
        """Error formal + piso empírico por instrumento (MAD o 5 % de la std).

        Solo se usa en el análisis exploratorio; el GP bayesiano usa
        `errores_formales`.
        """
        pisos_empiricos = np.empty(len(self.nombres_instrumentos))
        for posicion, (_, mascara) in enumerate(self.instrumentos()):
            valores_instrumento = self.valores[mascara]
            desviacion_absoluta_mediana = 1.4826 * np.median(np.abs(
                valores_instrumento - np.median(valores_instrumento)))
            pisos_empiricos[posicion] = max(desviacion_absoluta_mediana,
                                            0.05 * np.std(valores_instrumento), 1e-10)
        return np.sqrt(self.errores_formales ** 2
                       + pisos_empiricos[self.indice_instrumento] ** 2)

    def instrumentos(self):
        """Itera (nombre_instrumento, mascara_de_sus_medidas)."""
        for posicion, nombre in enumerate(self.nombres_instrumentos):
            yield str(nombre), self.indice_instrumento == posicion

    def seleccionar(self, mascara):
        return Serie(self.tiempos[mascara], self.valores[mascara],
                     self.errores_formales[mascara],
                     self.instrumento_por_medida[mascara])

    def hash(self):
        """Firma de los datos, para comprobar que una reanudación usa los mismos."""
        firma = hashlib.sha256()
        for columna in (self.tiempos, self.valores, self.errores_formales,
                        self.indice_instrumento):
            firma.update(np.ascontiguousarray(columna).tobytes())
        return firma.hexdigest()


def leer_serie(ruta, actividad=False, excluir=None):
    """Lee un CSV de RV o de S_MW. `excluir` quita un instrumento por nombre."""
    with Path(ruta).open(newline="", encoding="utf-8-sig") as archivo:
        filas = list(csv.DictReader(archivo))
    factor = FACTOR_ACTIVIDAD if actividad else 1.0
    columna_valor = "S_MW x1000" if actividad else "RV (m/s)"
    columna_error = "sig S_MW x1000" if actividad else "err RV (m/s)"
    serie = Serie(
        tiempos=np.array([float(fila["BJD"]) - config.EPOCA_BJD for fila in filas]),
        valores=factor * np.array([float(fila[columna_valor]) for fila in filas]),
        errores_formales=factor * np.array([float(fila[columna_error])
                                            for fila in filas]),
        instrumento_por_medida=np.array([fila["Inst"] for fila in filas]),
    )
    if excluir is not None:
        if excluir not in serie.nombres_instrumentos:
            raise ValueError(f"No existe el instrumento {excluir}")
        serie = serie.seleccionar(serie.instrumento_por_medida != excluir)
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
    conservar = np.ones(len(serie.tiempos), dtype=bool)
    for _, mascara in serie.instrumentos():
        valores_instrumento = serie.valores[mascara]
        mediana = np.median(valores_instrumento)
        dispersion_mad = 1.4826 * np.median(np.abs(valores_instrumento - mediana))
        conservar[mascara] = (np.abs(valores_instrumento - mediana)
                              <= n_sigma * max(dispersion_mad, 1e-12))
    return serie.seleccionar(conservar)
