"""Analisis desde cero de las RV de Epsilon Eridani con GP y dynesty.

M0: actividad estelar GP; M1: GP + planeta circular;
M2: GP + planeta kepleriano con excentricidad libre.

Los siete offsets instrumentales y la pendiente se marginalizan con
priors gaussianos (identicos en los tres modelos).
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from importlib.metadata import version

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "Epsilon_Eridani_RVdata.csv"
RESULTADOS = RAIZ / "results"
MODELOS = ("M0", "M1", "M2")

def leer_datos(ruta=DATOS, excluir=None):
    """Lee la SERIE COMPLETA, ordena las fechas y crea un indicador por instrumento."""
    with Path(ruta).open(newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise ValueError("El CSV no contiene observaciones")
    t = np.array([float(r["BJD"]) - 2450000.0 for r in filas])
    y = np.array([float(r["RV (m/s)"]) for r in filas])
    error = np.array([float(r["err RV (m/s)"]) for r in filas])
    inst = np.array([r["Inst"] for r in filas])
    if excluir:
        keep = inst != excluir
        if keep.all():
            raise ValueError(f"No existe el instrumento {excluir}")
        t, y, error, inst = t[keep], y[keep], error[keep], inst[keep]
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))
            and np.all(np.isfinite(error)) and np.all(error > 0)):
        raise ValueError("Medidas no finitas o incertidumbres no positivas")
    orden = np.argsort(t)
    t, y, error, inst = t[orden], y[orden], error[orden], inst[orden]
    nombres, indice = np.unique(inst, return_inverse=True)
    t0 = float(np.median(t))
    # Medianas instrumentales: traslacion fija de RV, NO se eliminan medidas.
    centros = np.array([np.median(y[indice == j]) for j in range(len(nombres))])
    z = y - centros[indice]
    X = np.column_stack([*( (indice == j).astype(float) for j in range(len(nombres)) ),
                         (t - t0) / 365.25])
    # Priors gaussianos PROPIOS sobre offsets relativos y tendencia,
    # iguales para M0, M1, M2. Centrados empiricamente sobre medianas.
    sig_beta = np.r_[np.repeat(300.0, len(nombres)), 20.0]
    h = hashlib.sha256()
    for a in (t, y, error, indice):
        h.update(np.ascontiguousarray(a).tobytes())
    return dict(t=t, y=y, err=error, indice=indice, nombres=nombres,
                t0=t0, centros=centros, z=z, X=X, sig_beta=sig_beta,
                sig_beta_inv2=1.0 / sig_beta**2, hash=h.hexdigest())


def priors_del_modelo(datos, modelo, ampliados=False):
    """Priors originales; --ampliados modifica solo Q0 y escala lenta."""
    if modelo not in MODELOS:
        raise ValueError("Modelo desconocido")
    p = [(f"jitter_{i}", "log", 0.01, 100.0) for i in datos["nombres"]]
    p += [
        ("gp_sigma_rot_m_s", "log", 0.05, 100.0),
        ("gp_periodo_rot_d", "lin", 8.0, 16.0),
        ("gp_Q0", "log", 0.03 if ampliados else 0.2, 20.0),
        ("gp_dQ", "log", 0.2, 50.0),
        ("gp_frac_armonico", "lin", 0.02, 1.0),
        ("gp_sigma_lenta_m_s", "log", 0.05, 100.0),
        ("gp_escala_lenta_d", "log", 30.0 if ampliados else 150.0, 5000.0),
    ]
    if modelo != "M0":
        p += [("planeta_periodo_d", "log", 200.0, 6000.0),
              ("planeta_K_m_s", "lin", 0.0, 80.0),
              ("planeta_fase", "lin", 0.0, 1.0)]
    if modelo == "M2":
        p += [("planeta_e", "lin", 0.0, 0.8),
              ("planeta_omega_rad", "lin", 0.0, 2.0 * np.pi)]
    return p


def transformar_prior(u, priors):
    """dynesty explora [0,1]^D; lo convertimos a parametros fisicos."""
    out = []
    for x, (_, tipo, minimo, maximo) in zip(u, priors):
        if tipo == "log":
            out.append(np.exp(np.log(minimo) + x * np.log(maximo / minimo)))
        else:
            out.append(minimo + x * (maximo - minimo))
    return np.asarray(out)


def velocidad_kepleriana(t, P, K, e, omega, fase, t0):
    """Resuelve M=E-e*sin(E) por Newton y devuelve la RV de una orbita."""
    if P <= 0 or K < 0 or not 0 <= e < 1:
        raise ValueError("Parametros orbitales no fisicos")
    M = 2.0 * np.pi * np.remainder((t - t0) / P - fase, 1.0)
    if e == 0.0:
        return K * np.cos(M + omega)
    E = M + e * np.sin(M)
    for _ in range(35):
        paso = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= paso
        if np.max(np.abs(paso)) < 1e-12:
            break
    nu = np.arctan2(np.sqrt(1.0 - e*e) * np.sin(E), np.cos(E) - e)
    return K * (np.cos(nu + omega) + e * np.cos(omega))


def orbita(params, modelo, datos):
    if modelo == "M0":
        return np.zeros_like(datos["t"])
    return velocidad_kepleriana(datos["t"], params["planeta_periodo_d"],
        params["planeta_K_m_s"], params.get("planeta_e", 0.),
        params.get("planeta_omega_rad", 0.), params["planeta_fase"], datos["t0"])


def loglike(theta, datos, modelo, priors, componentes=False):
    """Likelihood marginal de RV con GP + ruido instrumental.

    Integra los 7 offsets y la pendiente con priors gaussianos propios:
    A=X' C^-1 X+D^-1; b=X' C^-1 r.
    ln L_marg = ln N(r | 0,C) + 0.5*b' A^-1 b - 0.5*ln(|D| |A|).
    El GP usa RotationTerm + Matern32Term, exactamente como el original.
    """
    from celerite2 import GaussianProcess, terms
    p = dict(zip((v[0] for v in priors), theta))
    ninst = len(datos["nombres"])
    var = datos["err"]**2 + theta[:ninst][datos["indice"]]**2
    rot = terms.RotationTerm(sigma=p["gp_sigma_rot_m_s"],
        period=p["gp_periodo_rot_d"], Q0=p["gp_Q0"],
        dQ=p["gp_dQ"], f=p["gp_frac_armonico"])
    lento = terms.Matern32Term(sigma=p["gp_sigma_lenta_m_s"],
                              rho=p["gp_escala_lenta_d"])
    gp = GaussianProcess(rot + lento, mean=0.0)
    o = orbita(p, modelo, datos)
    r = datos["z"] - o
    X = datos["X"]
    try:
        gp.compute(datos["t"], diag=var)
        inv = gp.apply_inverse(np.column_stack((r, X)))
        Cr, CX = inv[:, 0], inv[:, 1:]
        A = X.T @ CX + np.diag(datos["sig_beta_inv2"])
        A = (A + A.T) * 0.5
        b = X.T @ Cr
        fac = cho_factor(A, lower=True, check_finite=False)
        beta = cho_solve(fac, b, check_finite=False)
        ln_det_A = 2.0 * np.sum(np.log(np.diag(fac[0])))
        lnL = gp.log_likelihood(r) + 0.5 * b @ beta - 0.5 * (
            np.sum(np.log(datos["sig_beta"]**2)) + ln_det_A)
        if not np.isfinite(lnL):
            return None if componentes else -np.inf
        if not componentes:
            return float(lnL)
        # Prediccion CONDICIONAL: no es una validacion independiente.
        gp_mean = gp.predict(r - X @ beta, return_var=False)
        return dict(lnL=float(lnL), orbita=o, gp=gp_mean,
                    tendencia=X[:, -1] * beta[-1],
                    offsets=beta[:-1] + datos["centros"],
                    residuos=r - X @ beta - gp_mean,
                    sigma=np.sqrt(var))
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
        return None if componentes else -np.inf


class LogLikelihood:
    """Objeto invocable y serializable para dynesty/checkpoints en Windows."""
    def __init__(self, datos, modelo, priors):
        self.datos, self.modelo, self.priors = datos, modelo, priors

    def __call__(self, theta):
        return loglike(theta, self.datos, self.modelo, self.priors)


class PriorTransform:
    def __init__(self, priors):
        self.priors = priors

    def __call__(self, cubo):
        return transformar_prior(cubo, self.priors)


def dibujar(datos, componentes, salida, modelo):
    """Figura simple: RV corregida de offsets, orbita, GP y residuos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    salida.mkdir(parents=True, exist_ok=True)
    t = datos["t"]
    centrada = datos["y"] - componentes["offsets"][datos["indice"]]
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    for j, inst in enumerate(datos["nombres"]):
        m = datos["indice"] == j
        ax[0].scatter(t[m], centrada[m], s=8, alpha=.45, label=inst)
    ax[0].plot(t, componentes["tendencia"] + componentes["orbita"],
               ".", color="tab:red", ms=2, label="Tendencia + orbita")
    ax[0].plot(t, componentes["tendencia"] + componentes["orbita"] + componentes["gp"],
               ".", color="black", ms=2, label="GP condicionado (no prediccion independiente)")
    ax[0].set_ylabel("RV - offset (m/s)")
    ax[0].set_title(f"{modelo}: GP + modelo orbital (medianas posteriores)")
    ax[0].legend(fontsize=7, ncol=3)
    ax[1].scatter(t, componentes["residuos"], s=8, alpha=.5)
    ax[1].axhline(0, color="black", ls="--", lw=.8)
    ax[1].set(xlabel="BJD - 2450000 (dias)", ylabel="Residuos (m/s)")
    fig.tight_layout()
    fig.savefig(salida / f"{modelo}_componentes.png", dpi=160)
    plt.close(fig)


def versiones():
    """Guarda las versiones realmente usadas para facilitar reproducibilidad."""
    paquetes = ("numpy", "scipy", "matplotlib", "celerite2", "dynesty")
    return {nombre: version(nombre) for nombre in paquetes}


def comprobar():
    """Verifica librerias, CSV y las tres verosimilitudes sin muestrear."""
    datos = leer_datos()
    print(f"RV: {len(datos['t'])} observaciones, {len(datos['nombres'])} instrumentos")
    for modelo in MODELOS:
        priors = priors_del_modelo(datos, modelo)
        theta = transformar_prior(np.full(len(priors), 0.5), priors)
        lnL = loglike(theta, datos, modelo, priors)
        if not np.isfinite(lnL):
            raise RuntimeError(f"Verosimilitud no finita para {modelo}")
        print(f"{modelo}: likelihood GP evaluada, lnL={lnL:.3f}")
    print("OK: todos los modelos pueden calcularse desde los CSV.")


def escribir_csv(ruta, cabeceras, filas):
    with ruta.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cabeceras)
        writer.writerows(filas)


def graficar_posterior(carpeta, modelo, priors, muestras):
    """Muestras reales calculadas por dynesty, no valores precalculados."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columnas = ["gp_periodo_rot_d"]
    if modelo != "M0":
        columnas += ["planeta_periodo_d", "planeta_K_m_s"]
    if modelo == "M2":
        columnas += ["planeta_e"]
    nombres = [p[0] for p in priors]
    fig, axes = plt.subplots(1, len(columnas), figsize=(4 * len(columnas), 3))
    for eje, nombre in zip(np.atleast_1d(axes), columnas):
        valores = muestras[:, nombres.index(nombre)]
        eje.hist(valores, bins=45, color="steelblue", alpha=0.65)
        eje.axvline(np.median(valores), color="black", linestyle="--")
        eje.set(xlabel=nombre.replace("_", " "), ylabel="Muestras")
    fig.suptitle(f"{modelo}: posterior obtenida en esta ejecucion")
    fig.tight_layout()
    fig.savefig(carpeta / f"{modelo}_posteriores.png", dpi=170)
    plt.close(fig)


def graficar_evidencias(base, evidencias):
    """Comparacion valida solo entre modelos calculados con los mismos datos/priors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nombres = list(evidencias)
    z = np.array([evidencias[m][0] for m in nombres])
    errores = np.array([evidencias[m][1] for m in nombres])
    origen = evidencias["M0"][0] if "M0" in evidencias else z[0]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(nombres, z - origen, yerr=errores, fmt="o", capsize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Modelo", ylabel=r"$\ln Z - \ln Z_\mathrm{M0}$",
           title="Evidencias calculadas desde cero")
    fig.tight_layout()
    fig.savefig(base / "comparacion_evidencias.png", dpi=170)
    plt.close(fig)



def registrar_resultados(base, resultados):
    """Actualiza tabla de evidencias solo con modelos ya terminados."""
    base.mkdir(parents=True, exist_ok=True)
    escribir_csv(base / "comparacion_evidencias.csv",
        ["modelo", "lnZ", "lnZerr", "delta_lnZ_vs_M0"],
        [[m, *resultados[m], (resultados[m][0] - resultados["M0"][0])
            if "M0" in resultados else ""] for m in resultados])
    graficar_evidencias(base, resultados)


def ajustar(modelos=MODELOS, nlive=80, dlogz=1.0, semilla=42,
            ampliados=False, sin_apf=False, carpeta_reanudacion=None):
    """Ejecuta realmente dynesty desde cero; checkpoint opcional para reanudar.

    No lee ni necesita muestras previas; las tablas/figuras se generan al terminar
    cada modelo. Las configuraciones reducidas dan resultados exploratorios.
    """
    import dynesty
    from dynesty.utils import resample_equal

    datos = leer_datos(excluir="APF_1" if sin_apf else None)
    if carpeta_reanudacion:
        base = Path(carpeta_reanudacion)
        # Las rutas relativas se interpretan respecto al proyecto, no a Spyder.
        base = (base if base.is_absolute() else RAIZ / base).resolve()
        if not base.is_dir():
            raise FileNotFoundError(f"No existe carpeta para reanudar: {base}")
    else:
        # Se crea una carpeta NUEVA: ejecutar dos veces nunca reutiliza las
        # evidencias ni los resultados de una ejecucion anterior.
        etiqueta = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = RESULTADOS / f"ajuste_{etiqueta}"
        base.mkdir(parents=True, exist_ok=False)

    configuracion = dict(
        hash_datos=datos["hash"], instrumentos=list(datos["nombres"]),
        nlive=int(nlive), dlogz=float(dlogz), semilla=int(semilla),
        priors_ampliados=bool(ampliados), excluir_APF=bool(sin_apf),
        modelos=list(modelos), versiones=versiones())
    config_path = base / "configuracion.json"
    if config_path.exists():
        previa = json.loads(config_path.read_text(encoding="utf-8"))
        if previa != configuracion:
            raise ValueError("Configuracion distinta a la original: no se puede reanudar")
    else:
        config_path.write_text(json.dumps(configuracion, indent=2,
                               ensure_ascii=False), encoding="utf-8")

    # Etapa rapida y reproducible de ambas series, sin afectar al GP de RV.
    from src import exploratorio
    exploratorio.analizar(RAIZ / "data", base / "exploratorio", sin_apf=sin_apf)

    print(f"Observaciones de RV: {len(datos['t'])}; modelos: {', '.join(modelos)}")
    print("Los modelos se calculan con dynesty + celerite2; no se leen resultados previos.")
    print(f"Carpeta de esta ejecucion: {base}")
    resultados = {}
    for modelo in modelos:
        carpeta = base / modelo
        carpeta.mkdir(exist_ok=True)
        priors = priors_del_modelo(datos, modelo, ampliados=ampliados)
        if nlive < 2 * len(priors):
            raise ValueError(f"nlive debe ser al menos {2*len(priors)} para {modelo}")

        ruta_evidencia = carpeta / "evidencia.csv"
        if ruta_evidencia.exists():
            if not carpeta_reanudacion:
                raise RuntimeError("Carpeta nueva con resultado inesperado")
            with ruta_evidencia.open(encoding="utf-8", newline="") as f:
                fila = next(csv.DictReader(f))
            resultados[modelo] = (float(fila["lnZ"]), float(fila["lnZerr"]))
            print(f"{modelo}: ya estaba terminado en esta ejecucion; se conserva")
            registrar_resultados(base, resultados)
            continue

        checkpoint = carpeta / "checkpoint.dynesty"
        if checkpoint.exists():
            if not carpeta_reanudacion:
                raise RuntimeError("Checkpoint inesperado en carpeta nueva")
            print(f"{modelo}: restaurando checkpoint")
            sampler = dynesty.NestedSampler.restore(str(checkpoint))
            reanudar = True
        else:
            print(f"{modelo}: iniciando Nested Sampling ({len(priors)} parametros, "
                  f"nlive={nlive}, dlogz={dlogz})")
            sampler = dynesty.NestedSampler(
                LogLikelihood(datos, modelo, priors), PriorTransform(priors),
                ndim=len(priors), nlive=nlive, sample="rwalk", bound="multi",
                rstate=np.random.default_rng(semilla))
            reanudar = False

        sampler.run_nested(dlogz=dlogz, checkpoint_file=str(checkpoint),
                           checkpoint_every=120, resume=reanudar,
                           print_progress=True)
        res = sampler.results
        pesos = np.exp(res.logwt - logsumexp(res.logwt))
        muestras = resample_equal(res.samples, pesos,
                                  rstate=np.random.default_rng(semilla + 1))
        p16, p50, p84 = np.percentile(muestras, [16, 50, 84], axis=0)
        escribir_csv(carpeta / "parametros.csv", ["parametro", "p16", "p50", "p84"],
            [[p[0], p16[i], p50[i], p84[i]] for i, p in enumerate(priors)])
        # Incluye muestras ponderadas y muestras de posterior para verificacion.
        np.savez_compressed(carpeta / "muestras.npz", theta=res.samples,
                            logwt=res.logwt, logl=res.logl,
                            muestras=muestras, logz=res.logz, logzerr=res.logzerr)
        graficar_posterior(carpeta, modelo, priors, muestras)
        componentes = loglike(p50, datos, modelo, priors, componentes=True)
        if componentes is None:
            # Los parametros marginales pueden tener una mediana conjunta
            # numericamente inestable; usamos entonces la muestra con mayor lnL.
            mejor = res.samples[np.argmax(res.logl)]
            componentes = loglike(mejor, datos, modelo, priors, componentes=True)
        if componentes is None:
            raise RuntimeError(f"No se pudo representar el GP para {modelo}")
        dibujar(datos, componentes, carpeta, modelo)
        # El archivo evidencia.csv es el marcador de modelo COMPLETO;
        # se escribe al final para no dar por terminado un ajuste a medias.
        escribir_csv(ruta_evidencia, ["modelo", "lnZ", "lnZerr"],
                     [[modelo, res.logz[-1], res.logzerr[-1]]])
        resultados[modelo] = float(res.logz[-1]), float(res.logzerr[-1])
        registrar_resultados(base, resultados)
        print(f"{modelo}: ln Z = {res.logz[-1]:.3f} +/- {res.logzerr[-1]:.3f}")

    print(f"\nTERMINADO. Resultados de esta ejecucion en:\n{base}")
    if "M0" in resultados and "M1" in resultados:
        print(f"Delta ln Z (M1-M0) = {resultados['M1'][0]-resultados['M0'][0]:.3f}")
    return base
