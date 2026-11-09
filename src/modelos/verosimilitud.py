"""Verosimilitud marginal de RV: GP + ruido instrumental + modelo orbital."""

import numpy as np
from celerite2 import GaussianProcess, terms
from scipy.linalg import cho_factor, cho_solve

from src.lineal import matriz_diseno

SIGMA_PRIOR_OFFSET_MS = 300.0
SIGMA_PRIOR_TENDENCIA_MS_ANIO = 20.0
ERRORES_NUMERICOS = (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError)


class VerosimilitudGP:
    """Invocable y serializable (dynesty): parámetros -> ln L marginal.

    Integra los offsets instrumentales y la pendiente con priors gaussianos:
    matriz_precision = Xt C^-1 X + D^-1;  termino_lineal = Xt C^-1 residuo;
    ln L = ln N(residuo | 0, C) + 0.5 termino_lineal' matriz_precision^-1 termino_lineal
           - 0.5 ln(|D| |matriz_precision|),
    con X = matriz de diseño, C = covarianza (GP + ruido) y D = covarianza prior.
    """

    def __init__(self, serie, modelo, espacio):
        self.serie, self.modelo, self.espacio = serie, modelo, espacio
        self.matriz_diseno = matriz_diseno(serie)
        # Las medianas instrumentales solo trasladan la RV; no se elimina nada.
        self.medianas_instrumentales = np.array(
            [np.median(serie.valores[mascara]) for _, mascara in serie.instrumentos()])
        self.rv_centrada = (serie.valores
                            - self.medianas_instrumentales[serie.indice_instrumento])
        self.sigma_prior_coeficientes = np.r_[
            np.full(len(serie.nombres_instrumentos), SIGMA_PRIOR_OFFSET_MS),
            SIGMA_PRIOR_TENDENCIA_MS_ANIO]

    def __call__(self, vector_parametros):
        ln_verosimilitud = self.evaluar(vector_parametros)
        return -np.inf if ln_verosimilitud is None else ln_verosimilitud

    @staticmethod
    def _terminos_gp(parametros):
        termino_rotacion = terms.RotationTerm(
            sigma=parametros["gp_sigma_rot_m_s"], period=parametros["gp_periodo_rot_d"],
            Q0=parametros["gp_Q0"], dQ=parametros["gp_dQ"],
            f=parametros["gp_frac_armonico"])
        termino_lento = terms.Matern32Term(sigma=parametros["gp_sigma_lenta_m_s"],
                                           rho=parametros["gp_escala_lenta_d"])
        return termino_rotacion + termino_lento

    def evaluar(self, vector_parametros, componentes=False):
        """ln L (float), o un dict con las componentes del modelo si `componentes`.

        Devuelve None si el cálculo falla numéricamente o no es finito.
        """
        serie, matriz = self.serie, self.matriz_diseno
        n_instrumentos = len(serie.nombres_instrumentos)
        try:
            parametros = self.espacio.como_diccionario(vector_parametros)
            jitter_por_medida = np.asarray(
                vector_parametros[:n_instrumentos])[serie.indice_instrumento]
            varianza_total = serie.errores_formales ** 2 + jitter_por_medida ** 2
            proceso_gaussiano = GaussianProcess(self._terminos_gp(parametros), mean=0.0)
            velocidad_orbital = self.modelo.velocidad_orbital(parametros, serie)
            rv_sin_orbita = self.rv_centrada - velocidad_orbital
            proceso_gaussiano.compute(serie.tiempos, diag=varianza_total)
            inversa_aplicada = proceso_gaussiano.apply_inverse(
                np.column_stack((rv_sin_orbita, matriz)))
            matriz_precision = (matriz.T @ inversa_aplicada[:, 1:]
                                + np.diag(1.0 / self.sigma_prior_coeficientes ** 2))
            matriz_precision = (matriz_precision + matriz_precision.T) * 0.5
            termino_lineal = matriz.T @ inversa_aplicada[:, 0]
            factor_cholesky = cho_factor(matriz_precision, lower=True, check_finite=False)
            coeficientes = cho_solve(factor_cholesky, termino_lineal, check_finite=False)
            ln_det_precision = 2.0 * np.sum(np.log(np.diag(factor_cholesky[0])))
            ln_verosimilitud = (
                proceso_gaussiano.log_likelihood(rv_sin_orbita)
                + 0.5 * termino_lineal @ coeficientes
                - 0.5 * (np.sum(np.log(self.sigma_prior_coeficientes ** 2))
                         + ln_det_precision))
            if not np.isfinite(ln_verosimilitud):
                return None
            if not componentes:
                return float(ln_verosimilitud)
            # Predicción CONDICIONAL: no es una validación independiente.
            gp_media = proceso_gaussiano.predict(
                rv_sin_orbita - matriz @ coeficientes, return_var=False)
            return dict(ln_verosimilitud=float(ln_verosimilitud), orbita=velocidad_orbital,
                        gp=gp_media,
                        tendencia=matriz[:, -1] * coeficientes[-1],
                        offsets=coeficientes[:-1] + self.medianas_instrumentales,
                        residuos=rv_sin_orbita - matriz @ coeficientes - gp_media)
        except ERRORES_NUMERICOS:
            return None
