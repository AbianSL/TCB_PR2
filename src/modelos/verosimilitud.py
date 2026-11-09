"""Verosimilitud marginal de RV: GP + ruido instrumental + modelo orbital."""

import numpy as np
from celerite2 import GaussianProcess, terms
from scipy.linalg import cho_factor, cho_solve

from src.lineal import matriz_diseno

SIGMA_PRIOR_OFFSET_MS = 300.0
SIGMA_PRIOR_TENDENCIA_MS_ANIO = 20.0
ERRORES_NUMERICOS = (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError)


class VerosimilitudGP:
    """Invocable y serializable (dynesty): theta -> ln L marginal.

    Integra los offsets instrumentales y la pendiente con priors gaussianos:
    A = X' C^-1 X + D^-1;  b = X' C^-1 r;
    ln L = ln N(r | 0, C) + 0.5 b' A^-1 b - 0.5 ln(|D| |A|).
    """

    def __init__(self, serie, modelo, espacio):
        self.serie, self.modelo, self.espacio = serie, modelo, espacio
        self.X = matriz_diseno(serie)
        # Las medianas instrumentales solo trasladan la RV; no se elimina nada.
        self.centros = np.array([np.median(serie.y[m]) for _, m in serie.instrumentos()])
        self.z = serie.y - self.centros[serie.indice]
        self.sig_beta = np.r_[np.full(len(serie.nombres), SIGMA_PRIOR_OFFSET_MS),
                              SIGMA_PRIOR_TENDENCIA_MS_ANIO]

    def __call__(self, theta):
        lnL = self.evaluar(theta)
        return -np.inf if lnL is None else lnL

    @staticmethod
    def _terminos_gp(p):
        rotacion = terms.RotationTerm(
            sigma=p["gp_sigma_rot_m_s"], period=p["gp_periodo_rot_d"],
            Q0=p["gp_Q0"], dQ=p["gp_dQ"], f=p["gp_frac_armonico"])
        lento = terms.Matern32Term(sigma=p["gp_sigma_lenta_m_s"],
                                   rho=p["gp_escala_lenta_d"])
        return rotacion + lento

    def evaluar(self, theta, componentes=False):
        """ln L (float), o un dict con las componentes del modelo si `componentes`.

        Devuelve None si el cálculo falla numéricamente o no es finito.
        """
        s, X = self.serie, self.X
        try:
            p = self.espacio.como_diccionario(theta)
            var = s.err**2 + np.asarray(theta[:len(s.nombres)])[s.indice] ** 2
            gp = GaussianProcess(self._terminos_gp(p), mean=0.0)
            orbita = self.modelo.velocidad_orbital(p, s)
            r = self.z - orbita
            gp.compute(s.t, diag=var)
            inversa = gp.apply_inverse(np.column_stack((r, X)))
            A = X.T @ inversa[:, 1:] + np.diag(1.0 / self.sig_beta**2)
            A = (A + A.T) * 0.5
            b = X.T @ inversa[:, 0]
            factor = cho_factor(A, lower=True, check_finite=False)
            beta = cho_solve(factor, b, check_finite=False)
            ln_det_A = 2.0 * np.sum(np.log(np.diag(factor[0])))
            lnL = gp.log_likelihood(r) + 0.5 * b @ beta - 0.5 * (
                np.sum(np.log(self.sig_beta**2)) + ln_det_A)
            if not np.isfinite(lnL):
                return None
            if not componentes:
                return float(lnL)
            # Predicción CONDICIONAL: no es una validación independiente.
            gp_media = gp.predict(r - X @ beta, return_var=False)
            return dict(lnL=float(lnL), orbita=orbita, gp=gp_media,
                        tendencia=X[:, -1] * beta[-1],
                        offsets=beta[:-1] + self.centros,
                        residuos=r - X @ beta - gp_media)
        except ERRORES_NUMERICOS:
            return None
