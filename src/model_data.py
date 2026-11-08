"""Modelado global de las series de Epsilon Eridani (TCB, practica 2).

Cada observable (RV y S_MW) se ajusta con TODAS sus mediciones a la vez.
Los periodos, amplitudes, fases y tendencias son comunes a los instrumentos;
cada instrumento tiene su nivel cero y su jitter independientes.

Uso desde la raiz del repositorio:
    cp model_data.py src/model_data.py
    python -m src.model_data
    python -m src.model_data --diagnosticos
    python ejecutar_diagnosticos.py  # opcion facil, tambien desde Spyder
    python -m src.model_data --bayes --periodo-rot 11.2 --periodo-largo 2800

Dependencias basicas: numpy, scipy, matplotlib.
El ajuste bayesiano es OPCIONAL y requiere dynesty; corner tambien es opcional.
Los periodos del ejemplo de ejecucion NO son resultados finales y deben
obtenerse/validarse mediante el analisis previo de las observaciones.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

DIAS_POR_ANIO = 365.25
# Rutas absolutas para que las salidas siempre vayan al repositorio,
# aunque se ejecute el script desde Spyder u otro directorio.
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIRECTORIO_RESULTADOS = RAIZ_PROYECTO / "results" / "modelado_global"


# -----------------------------------------------------------------------------
# 1. PREPARACION: una unica serie, varias etiquetas instrumentales
# -----------------------------------------------------------------------------
def preparar_datos(rjd, y, sy, instrumento):
    """Comprobar arrays y construir las mascaras instrumentales.

    RJD = BJD - 2450000, en dias; y y sy deben tener la misma unidad.
    No se eliminan medidas automaticamente: cualquier descarte debe
    analizarse y justificarse previamente, no esconderse en el ajuste.
    """
    t = np.asarray(rjd, dtype=float)
    v = np.asarray(y, dtype=float)
    e = np.asarray(sy, dtype=float)
    nombres_original = np.asarray(instrumento, dtype=str)

    if not (t.ndim == v.ndim == e.ndim == nombres_original.ndim == 1):
        raise ValueError("Los cuatro argumentos deben ser arrays de una dimension")
    if not (len(t) == len(v) == len(e) == len(nombres_original)):
        raise ValueError("Los cuatro arrays deben tener igual longitud")
    if len(t) < 5:
        raise ValueError("No hay suficientes medidas para ajustar el modelo")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))
            and np.all(np.isfinite(e)) and np.all(e > 0)):
        raise ValueError("Hay valores no finitos o errores no positivos")

    orden = np.argsort(t)
    t, v, e, nombres_original = (z[orden] for z in (t, v, e, nombres_original))
    instrumentos, indice = np.unique(nombres_original, return_inverse=True)

    # Restar un mismo t0 evita numeros enormes en el calculo de la pendiente.
    # Dividir por 365.25 expresa directamente la pendiente en unidad/anio.
    t0 = float(np.median(t))
    t_anios = (t - t0) / DIAS_POR_ANIO
    mascara = np.arange(len(instrumentos))[:, None] == indice[None, :]

    return {
        "t": t, "y": v, "e": e, "t0": t0, "t_anios": t_anios,
        "nombres": instrumentos, "indice": indice, "mascaras": mascara,
        "ninst": len(instrumentos), "n": len(t),
    }


def matriz_modelo(datos, tendencia=False, seniales=()):
    """Construir la matriz X del modelo lineal y una etiqueta por coeficiente.

    y(t) = gamma_instrumento + m*(t-t0) +
           sum_k [a_k sin(2*pi*k*(t-t0)/P) + b_k cos(...)].

    'seniales' contiene tuplas (periodo_en_dias, numero_armonicos).
    Los coeficientes a y b son lineales SI LOS PERIODOS ESTAN FIJADOS.
    Los primeros coeficientes son niveles cero, uno por instrumento.
    """
    columnas = [(datos["indice"] == j).astype(float)
                for j in range(datos["ninst"])]
    etiquetas = [f"Offset_{n}" for n in datos["nombres"]]

    if tendencia:
        columnas.append(datos["t_anios"])
        etiquetas.append("Pendiente_unidades_anio")

    delta_t = datos["t"] - datos["t0"]
    for periodo, armonicos in seniales:
        if not np.isfinite(periodo) or periodo <= 0:
            raise ValueError("Todos los periodos deben ser positivos y finitos")
        for orden_armonico in range(1, armonicos + 1):
            fase = 2 * np.pi * orden_armonico * delta_t / periodo
            columnas.extend([np.sin(fase), np.cos(fase)])
            etiquetas.extend([f"Seno_P{periodo:.6f}_h{orden_armonico}",
                              f"Coseno_P{periodo:.6f}_h{orden_armonico}"])

    return np.column_stack(columnas), etiquetas


# -----------------------------------------------------------------------------
# 2. MINIMOS CUADRADOS PONDERADOS + JITTER POR INSTRUMENTO
# -----------------------------------------------------------------------------
def errores_efectivos(datos, jitters):
    """Combinar errores publicados y dispersion adicional en cuadratura."""
    return np.sqrt(datos["e"]**2 + np.asarray(jitters)[datos["indice"]]**2)


def ajustar_matriz(datos, X, estimar_jitter=True):
    """Maximizar la verosimilitud gaussiana con offset/tendencia/senos lineales.

    En cada prueba de jitter se resuelven los coeficientes lineales mediante
    minimos cuadrados ponderados. Luego se optimizan los jitters incluyendo
    el termino de normalizacion ln(sigma), que no se debe omitir cuando
    sigma depende de parametros libres.
    """
    y, e, indice = datos["y"], datos["e"], datos["indice"]
    ninst = datos["ninst"]

    # Escalas iniciales por instrumento: diferencias entre la dispersion
    # observada y la incertidumbre instrumental tipica.
    dispersion = np.array([np.std(y[indice == j]) for j in range(ninst)])
    error_tipico = np.array([np.median(e[indice == j]) for j in range(ninst)])
    j0 = np.sqrt(np.maximum(0, dispersion**2 - error_tipico**2))
    limite = np.maximum(5 * dispersion, 10 * error_tipico)
    limite = np.maximum(limite, np.max(e))

    def evaluar(jitters):
        sigma = np.sqrt(e**2 + jitters[indice]**2)
        Xw = X / sigma[:, None]
        yw = y / sigma
        # lstsq es mas estable que invertir explicitamente X.T @ W @ X.
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        residuos = y - X @ beta
        lnL = -0.5 * np.sum((residuos / sigma)**2
                            + np.log(2 * np.pi * sigma**2))
        return lnL, beta, residuos, sigma

    if estimar_jitter:
        # Optimizacion limitada: un jitter nunca puede ser negativo.
        opt = minimize(lambda j: -evaluar(j)[0], j0, method="L-BFGS-B",
                       bounds=[(0.0, float(x)) for x in limite],
                       options={"maxiter": 250, "ftol": 1e-10})
        if not opt.success and not np.isfinite(opt.fun):
            raise RuntimeError(f"No converge el ajuste del jitter: {opt.message}")
        jitters = np.maximum(opt.x, 0)
    else:
        jitters = np.zeros(ninst)

    lnL, beta, residuos, sigma = evaluar(jitters)
    Xw = X / sigma[:, None]
    # La incertidumbre formal de beta es CONDICIONAL al jitter encontrado.
    # No incluye la incertidumbre adicional por haberlo estimado con los datos.
    cov = np.linalg.pinv(Xw.T @ Xw)
    sig_beta = np.sqrt(np.maximum(np.diag(cov), 0))

    chi2 = float(np.sum((residuos / sigma)**2))
    return {"beta": beta, "sig_beta": sig_beta, "jitter": jitters,
            "residuos": residuos, "sigma": sigma, "prediccion": X @ beta,
            "lnL": float(lnL), "chi2": chi2,
            "rms": float(np.sqrt(np.mean(residuos**2)))}


def ajustar_modelo(datos, nombre, tendencia=False, seniales=(),
                   periodos_estimados=0):
    """Obtener estimadores, residuos y criterios de seleccion de un modelo."""
    X, etiquetas = matriz_modelo(datos, tendencia, seniales)
    if datos["n"] <= X.shape[1] + datos["ninst"]:
        raise ValueError("Demasiados parametros para el numero de observaciones")

    resultado = ajustar_matriz(datos, X)
    # Contamos tambien jitters y periodos seleccionados con estos mismos datos.
    k = X.shape[1] + datos["ninst"] + int(periodos_estimados)
    resultado.update({
        "nombre": nombre, "etiquetas": etiquetas, "tendencia": tendencia,
        "seniales": tuple(seniales), "k": k,
        "bic": float(k * np.log(datos["n"]) - 2 * resultado["lnL"]),
        "chi2red": float(resultado["chi2"] / (datos["n"] - k)),
    })
    return resultado


# -----------------------------------------------------------------------------
# 3. BUSQUEDA GLOBAL DE PERIODOS SIN AJUSTAR CADA INSTRUMENTO POR SEPARADO
# -----------------------------------------------------------------------------
def buscar_periodos(datos, modelo_base, pmin, pmax, nfrecuencias=None):
    """Periodograma por mejora de chi2 frente a un modelo nulo GLOBAL.

    El modelo nulo ya incluye un offset por instrumento y (si procede)
    una tendencia y otras sinusoides. En cada frecuencia probamos DOS
    columnas compartidas (seno/coseno), y perfilamos los offsets por WLS.
    No es la potencia normalizada de Lomb-Scargle ni aporta aqui una FAP.
    """
    if not (0 < pmin < pmax):
        raise ValueError("Se requiere 0 < periodo_minimo < periodo_maximo")

    # Para explorar periodos cortos se requiere una malla mas fina:
    # unas cuatro muestras por resolucion nominal 1 / baseline.
    frecuencia_min = 1 / pmax
    frecuencia_max = 1 / pmin
    baseline = np.ptp(datos["t"])
    if nfrecuencias is None:
        nfrecuencias = int(np.clip(4 * baseline * (frecuencia_max - frecuencia_min),
                                  900, 12000))
    frecuencia = np.linspace(frecuencia_min, frecuencia_max, nfrecuencias)

    X0, _ = matriz_modelo(datos, modelo_base["tendencia"],
                          modelo_base["seniales"])
    sigma = errores_efectivos(datos, modelo_base["jitter"])
    X0w = X0 / sigma[:, None]
    yw = datos["y"] / sigma

    # Proyectar primero sobre el modelo base: las columnas periodicas
    # competiran SOLO por la estructura no explicada por dicho modelo.
    Q, _ = np.linalg.qr(X0w, mode="reduced")
    y_res = yw - Q @ (Q.T @ yw)
    norma_res = np.dot(y_res, y_res)
    dt = datos["t"] - datos["t0"]

    def mejora_en_frecuencia(f):
        Z = np.column_stack((np.sin(2 * np.pi * f * dt),
                             np.cos(2 * np.pi * f * dt))) / sigma[:, None]
        Z -= Q @ (Q.T @ Z)  # quitar la parte degenerada con offsets/tendencia
        H = Z.T @ Z
        b = Z.T @ y_res
        try:
            return float(b @ np.linalg.solve(H, b))
        except np.linalg.LinAlgError:
            return 0.0

    mejora = np.array([mejora_en_frecuencia(f) for f in frecuencia])
    idx = int(np.argmax(mejora))

    # Refinar el pico localmente, sin abandonar la vecindad encontrada.
    f0 = frecuencia[max(0, idx - 1)]
    f1 = frecuencia[min(len(frecuencia) - 1, idx + 1)]
    opt = minimize_scalar(lambda f: -mejora_en_frecuencia(f),
                          bounds=(f0, f1), method="bounded",
                          options={"xatol": 1e-12})
    fmejor = float(opt.x) if opt.success else float(frecuencia[idx])

    return {
        "periodo": 1 / fmejor, "frecuencia": frecuencia,
        "periodos": 1 / frecuencia,
        "potencia": np.clip(mejora / max(norma_res, np.finfo(float).eps), 0, 1),
        "mejora_chi2": mejora,
    }


# -----------------------------------------------------------------------------
# 4. FIGURAS Y TABLAS: LA SERIE COMPLETA EN UN UNICO GRAFICO
# -----------------------------------------------------------------------------
def _serie_fisica(t, datos, resultado):
    """Evaluar SOLO la componente comun; sin offsets instrumentales."""
    t = np.asarray(t, dtype=float)
    beta = resultado["beta"]
    pos = datos["ninst"]
    compartida = np.zeros(len(t), dtype=float)
    if resultado["tendencia"]:
        compartida += beta[pos] * (t - datos["t0"]) / DIAS_POR_ANIO
        pos += 1
    for periodo, armonicos in resultado["seniales"]:
        for arm in range(1, armonicos + 1):
            angulo = 2 * np.pi * arm * (t - datos["t0"]) / periodo
            compartida += beta[pos] * np.sin(angulo)
            compartida += beta[pos + 1] * np.cos(angulo)
            pos += 2
    return compartida


def graficar_ajuste(datos, resultado, observable, ruta):
    """Datos sin offsets + modelo fisico compartido + residuos (ambas series)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

    # Se quita UNICAMENTE el offset del modelo, no se reajustan las series
    # individualmente para modificar sus oscilaciones o periodos.
    offsets = resultado["beta"][:datos["ninst"]]
    y_centrado = datos["y"] - offsets[datos["indice"]]
    for j, nombre in enumerate(datos["nombres"]):
        mask = datos["indice"] == j
        ax.errorbar(datos["t"][mask], y_centrado[mask],
                    yerr=datos["e"][mask], fmt=".", markersize=3,
                    alpha=0.5, elinewidth=0.4, label=nombre)
        axr.plot(datos["t"][mask], resultado["residuos"][mask],
                 ".", markersize=3, alpha=0.5)

    # Para periodos cortos y bases temporales de decadas, un grid que cubra
    # TODA la serie generaria lineas muy densas. Se muestran predicciones
    # en los tiempos observados y no una interpolacion artificial entre huecos.
    if resultado["seniales"] and min(x[0] for x in resultado["seniales"]) < 50:
        ax.plot(datos["t"], _serie_fisica(datos["t"], datos, resultado),
                ".", color="black", markersize=1.0, alpha=0.35,
                label="Modelo en tiempos observados")
    else:
        tgrid = np.linspace(datos["t"].min(), datos["t"].max(), 2000)
        ax.plot(tgrid, _serie_fisica(tgrid, datos, resultado),
                "-", color="black", lw=1.4, label="Modelo comun")

    ax.axhline(0, color="gray", ls=":", lw=0.8)
    axr.axhline(0, color="black", ls="--", lw=1)
    ax.set_ylabel(f"{observable} sin offsets")
    axr.set_ylabel("Residuos")
    axr.set_xlabel("BJD - 2450000 (dias)")
    ax.set_title(f"Epsilon Eridani: {resultado['nombre']} (serie completa)")
    ax.legend(ncol=4, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.subplots_adjust(hspace=0.12, bottom=0.18)
    fig.savefig(ruta.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def graficar_periodograma(busqueda, titulo, ruta):
    """Mostrar potencia = fraccion de chi2 reducida frente al modelo nulo."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(busqueda["periodos"], busqueda["potencia"], lw=0.9)
    ax.axvline(busqueda["periodo"], color="darkred", linestyle="--",
               label=f"P = {busqueda['periodo']:.3f} d")
    ax.set_xscale("log")
    ax.set_xlabel("Periodo (dias)")
    ax.set_ylabel("Mejora fraccional de $\\chi^2$")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def guardar_comparacion(datos, resultados, destino):
    """Guardar metadatos y parametros: permiten reproducir tablas de la memoria."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    with (destino / "comparacion_modelos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "N", "num_parametros", "lnL", "BIC", "chi2red", "RMS"])
        for res in sorted(resultados, key=lambda r: r["bic"]):
            w.writerow([res["nombre"], datos["n"], res["k"], res["lnL"],
                        res["bic"], res["chi2red"], res["rms"]])

    with (destino / "parametros_modelos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "parametro", "valor", "error_formal_condicional"])
        for res in resultados:
            for nombre, valor, err in zip(res["etiquetas"], res["beta"], res["sig_beta"]):
                w.writerow([res["nombre"], nombre, valor, err])
            for nombre, jitter in zip(datos["nombres"], res["jitter"]):
                w.writerow([res["nombre"], f"Jitter_{nombre}", jitter, ""])
            for i, (periodo, armonicos) in enumerate(res["seniales"], 1):
                w.writerow([res["nombre"], f"Periodo_{i}_dias", periodo, ""])
                w.writerow([res["nombre"], f"Armonicos_{i}", armonicos, ""])


def imprimir_comparacion(resultados):
    """Comparacion orientativa con BIC (menor es preferible)."""
    print("\nModelo                         RMS       chi2red      BIC")
    for res in sorted(resultados, key=lambda r: r["bic"]):
        print(f"{res['nombre']:<27} {res['rms']:9.3f} {res['chi2red']:11.3f} {res['bic']:11.2f}")
    print("El BIC penaliza la complejidad. No demuestra que una señal sea planetaria.")


# -----------------------------------------------------------------------------
# 5. RV: constantes y pendientes GLOBALMENTE (compatibilidad con main.py)
# -----------------------------------------------------------------------------
def fit_rv_trends(rjd, rv, srv, instrumento):
    """Comparar offsets constantes y pendiente comun a los siete instrumentos."""
    datos = preparar_datos(rjd, rv, srv, instrumento)
    constante = ajustar_modelo(datos, "OFFSETS", tendencia=False)
    lineal = ajustar_modelo(datos, "OFFSETS+LINEAL", tendencia=True)
    resultados = [constante, lineal]

    carpeta = DIRECTORIO_RESULTADOS / "RV_tendencia"
    guardar_comparacion(datos, resultados, carpeta)
    graficar_ajuste(datos, lineal, "RV (m/s)", carpeta / "tendencia_global")
    print(f"\nRV global: N={datos['n']}, instrumentos={datos['ninst']}")
    imprimir_comparacion(resultados)
    print(f"Pendiente compartida = {lineal['beta'][datos['ninst']]:.5f} "
          f"+/- {lineal['sig_beta'][datos['ninst']]:.5f} m/s/anio (error formal)")
    return resultados


# -----------------------------------------------------------------------------
# 6. ACTIVIDAD: mismo ajuste compartido, ahora del indice S_MW
# -----------------------------------------------------------------------------
def fit_activity_models(rjd, s, ss, instrumento, pmin=5., pmax=30.,
                        rango_ciclo=(200., 5000.)):
    """Buscar un periodo corto en la serie COMPLETA del indice de actividad.

    Los datos de actividad deben llegar en unidades S_MW, NO x1000.
    El periodo es exploratorio y no se interpreta automaticamente como
    un valor preciso del periodo de rotacion de la estrella.
    """
    datos = preparar_datos(rjd, s, ss, instrumento)
    base = ajustar_modelo(datos, "SMW_OFFSETS+LINEAL", tendencia=True)
    busqueda = buscar_periodos(datos, base, pmin=pmin, pmax=pmax)
    periodo = busqueda["periodo"]

    rot1 = ajustar_modelo(datos, "SMW_ROT1", tendencia=True,
                          seniales=((periodo, 1),), periodos_estimados=1)
    rot2 = ajustar_modelo(datos, "SMW_ROT2", tendencia=True,
                          seniales=((periodo, 2),), periodos_estimados=1)
    resultados = [base, rot1, rot2]

    # La actividad estelar tambien puede variar en escalas de cientos o
    # miles de dias. Buscar el ciclo DESPUES de ajustar la rotacion ayuda
    # a comprobar si una señal larga de RV podria ser actividad magnetica.
    periodo_ciclo = None
    ciclo = None
    max_ciclo = min(rango_ciclo[1], float(np.ptp(datos["t"])))
    if max_ciclo > rango_ciclo[0]:
        ciclo = buscar_periodos(datos, rot2, rango_ciclo[0], max_ciclo)
        periodo_ciclo = ciclo["periodo"]
        rot_ciclo = ajustar_modelo(
            datos, "SMW_ROT2+CICLO", tendencia=True,
            seniales=((periodo, 2), (periodo_ciclo, 1)),
            periodos_estimados=2)
        resultados.append(rot_ciclo)

    mejor = min(resultados, key=lambda r: r["bic"])

    carpeta = DIRECTORIO_RESULTADOS / "SMW"
    guardar_comparacion(datos, resultados, carpeta)
    graficar_periodograma(busqueda, "Actividad S_MW (toda la serie)",
                          carpeta / "periodograma_actividad")
    if ciclo is not None:
        graficar_periodograma(ciclo, "S_MW: variabilidad larga tras ROT2",
                              carpeta / "periodograma_ciclo")
    graficar_ajuste(datos, mejor, "S_MW", carpeta / "ajuste_actividad")
    print(f"\nActividad global: N={datos['n']}, instrumentos={datos['ninst']}")
    print(f"Periodo exploratorio de actividad = {periodo:.4f} d")
    if periodo_ciclo is not None:
        print(f"Periodo largo exploratorio en S_MW = {periodo_ciclo:.2f} d")
    imprimir_comparacion(resultados)
    return {"periodo": periodo, "periodo_ciclo": periodo_ciclo,
            "resultados": resultados, "mejor": mejor}


# -----------------------------------------------------------------------------
# 7. RV: seniales cortas y largas en la SERIE COMPLETA
# -----------------------------------------------------------------------------
def fit_rv_sine(rjd, rv, srv, instrumento, period_rot=None,
                rango_rot=(5., 30.), rango_largo=(200., 10000.),
                period_cycle=None):
    """Comparar tendencias + seniales periodicas GLOBALMENTE.

    Se prueba una componente corta con uno y dos armonicos. Despues se
    explora una componente de periodo largo en los residuos del modelo
    con dos armonicos. Los offsets y jitters se reajustan conjuntamente.

    Si se proporciona period_rot, procede preferiblemente del analisis
    del S_MW, evitando seleccionar el periodo de rotacion usando solo RV.
    La sinusoide larga es un CANDIDATO: puede ser actividad magnetica,
    senal orbital, un alias del muestreo u otro fenomeno.
    """
    datos = preparar_datos(rjd, rv, srv, instrumento)
    base = ajustar_modelo(datos, "RV_OFFSETS+LINEAL", tendencia=True)

    if period_rot is None:
        corto = buscar_periodos(datos, base, *rango_rot)
        periodo_corto = corto["periodo"]
        cuenta_periodo_corto = 1
    else:
        periodo_corto = float(period_rot)
        cuenta_periodo_corto = 0
        corto = buscar_periodos(datos, base, *rango_rot)

    rot1 = ajustar_modelo(datos, "RV_ROT1", tendencia=True,
                          seniales=((periodo_corto, 1),),
                          periodos_estimados=cuenta_periodo_corto)
    rot2 = ajustar_modelo(datos, "RV_ROT2", tendencia=True,
                          seniales=((periodo_corto, 2),),
                          periodos_estimados=cuenta_periodo_corto)

    # Acotar la busqueda larga para no pedir periodos imposibles/degenerados.
    # Son limites de EXPLORACION, no priors astrofisicos justificados.
    minimo_largo, maximo_largo = rango_largo
    baseline = float(np.ptp(datos["t"]))
    maximo_largo = min(maximo_largo, baseline)
    resultados = [base, rot1, rot2]

    if period_cycle is not None:
        # Contrastar una oscilacion RV con EL MISMO PERIODO que aparece en
        # el indicador de actividad; no implica que tengan igual fase.
        modelo_ciclo = ajustar_modelo(
            datos, "RV_ROT2+CICLO_SMW", tendencia=True,
            seniales=((periodo_corto, 2), (float(period_cycle), 1)),
            periodos_estimados=cuenta_periodo_corto)
        resultados.append(modelo_ciclo)

    if maximo_largo > minimo_largo:
        largo = buscar_periodos(datos, rot2, minimo_largo, maximo_largo)
        periodo_largo = largo["periodo"]
        rot_largo = ajustar_modelo(datos, "RV_ROT2+LARGO", tendencia=True,
                                   seniales=((periodo_corto, 2), (periodo_largo, 1)),
                                   periodos_estimados=cuenta_periodo_corto + 1)
        resultados.append(rot_largo)
    else:
        largo = None
        periodo_largo = None

    mejor = min(resultados, key=lambda r: r["bic"])
    carpeta = DIRECTORIO_RESULTADOS / "RV_periodicas"
    guardar_comparacion(datos, resultados, carpeta)
    graficar_periodograma(corto, "RV: periodicidad corta (serie completa)",
                          carpeta / "periodograma_corto")
    if largo is not None:
        graficar_periodograma(largo, "RV: periodicidad larga tras ROT2",
                              carpeta / "periodograma_largo")
    graficar_ajuste(datos, mejor, "RV (m/s)", carpeta / "ajuste_rv_periodico")

    print(f"\nRV global: N={datos['n']}, instrumentos={datos['ninst']}")
    print(f"Periodo corto ({'actividad' if period_rot is not None else 'RV'}) "
          f"= {periodo_corto:.4f} dias")
    if periodo_largo is not None:
        print(f"Candidato de periodo largo = {periodo_largo:.2f} dias")
    imprimir_comparacion(resultados)
    return {"periodo_corto": periodo_corto, "periodo_largo": periodo_largo,
            "resultados": resultados, "mejor": mejor}


# -----------------------------------------------------------------------------
# 8. EJECUCION INDEPENDIENTE, SIN CAMBIAR POR AHORA EL main.py DEL EQUIPO
# -----------------------------------------------------------------------------
def leer_csv(ruta, variable, incertidumbre, factor=1.):
    """Leer los CSV originales relativos a la raiz real del repositorio."""
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = RAIZ_PROYECTO / ruta
    t, v, e, nombres = [], [], [], []
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            t.append(float(fila["BJD"]) - 2450000.)
            v.append(float(fila[variable]) * factor)
            e.append(float(fila[incertidumbre]) * factor)
            nombres.append(fila["Inst"])
    return np.array(t), np.array(v), np.array(e), np.array(nombres)


def main():
    """Ejecutar los dos analisis sobre las observaciones del repositorio."""
    try:
        rjd, rv, srv, inst_rv = leer_csv(
            "data/Epsilon_Eridani_RVdata.csv", "RV (m/s)", "err RV (m/s)")
        ts, s, ss, inst_s = leer_csv(
            "data/Epsilon_Eridani_SMW.csv", "S_MW x1000", "sig S_MW x1000",
            factor=1 / 1000.)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"Error leyendo los CSV desde la raiz del proyecto: {exc}") from exc

    fit_rv_trends(rjd, rv, srv, inst_rv)
    actividad = fit_activity_models(ts, s, ss, inst_s)
    fit_rv_sine(rjd, rv, srv, inst_rv, period_rot=actividad["periodo"],
                period_cycle=actividad["periodo_ciclo"])
    print(f"\nFiguras y tablas guardadas en {DIRECTORIO_RESULTADOS}")



# -----------------------------------------------------------------------------
# 9. INFERENCIA BAYESIANA: NESTED SAMPLING CON PARAMETROS LINEALES MARGINALIZADOS
# -----------------------------------------------------------------------------
def configuracion_bayes(datos, tendencia=True, seniales=(),
                        escala_pendiente=None, escala_senial=None):
    """Especificar priors propios para el modelo lineal y los periodos.

    seniales: tuplas (periodo_central, numero_armonicos, fraccion_semianchura).
    Por ejemplo, (11.2, 2, 0.15) usa P uniforme entre 0.85 y 1.15 P_central.

    NOTA METODOLOGICA: los centros y escalas de offsets y amplitudes se
    estiman de las propias mediciones. Por tanto, es un analisis exploratorio
    de tipo empirico-bayesiano, no una evidencia enteramente preespecificada.
    Comparar evidencias requiere mantener identicos estos priors comunes.
    """
    from scipy.stats import halfnorm

    # Escala interna SIN mezclar niveles cero de distintos instrumentos.
    dispersion = np.array([np.std(datos["y"][m], ddof=1)
                           for m in datos["mascaras"]])
    error_med = np.array([np.median(datos["e"][m])
                          for m in datos["mascaras"]])
    escala_comun = float(np.median(dispersion))
    if not np.isfinite(escala_comun) or escala_comun <= 0:
        raise ValueError("La dispersión de las series debe ser positiva")

    # Jitter positivo con prior half-normal. Es preferible a permitir
    # jitters negativos o usar un prior de anchura infinita e impropio.
    jitter_scales = np.maximum(dispersion, error_med)
    jitter_priors = [halfnorm(scale=float(s)) for s in jitter_scales]

    # Offsets: el centro depende de cada instrumento, pero la amplitud
    # del prior es mas ancha que la variacion temporal observada.
    medias_offset = np.array([np.median(datos["y"][m])
                              for m in datos["mascaras"]])
    sigmas_offset = np.maximum(5 * dispersion, 10 * error_med)

    beta_mu = list(medias_offset)
    beta_sigma = list(sigmas_offset)
    beta_labels = [f"Offset_{nombre}" for nombre in datos["nombres"]]

    if tendencia:
        if escala_pendiente is None:
            # m/s por anio en RV, S_MW/anio en actividad.
            # Se impone una escala flexible a lo largo de la cobertura.
            escala_pendiente = 3 * escala_comun / max(np.ptp(datos["t_anios"]), 1)
        if escala_pendiente <= 0:
            raise ValueError("La escala del prior de pendiente debe ser positiva")
        beta_mu.append(0.0)
        beta_sigma.append(float(escala_pendiente))
        beta_labels.append("Pendiente_unidades_anio")

    if escala_senial is None:
        escala_senial = 2 * escala_comun
    if escala_senial <= 0:
        raise ValueError("La escala del prior de amplitud debe ser positiva")
    for indice_senal, (periodo, armonicos, fraccion) in enumerate(seniales, 1):
        if not (np.isfinite(periodo) and periodo > 0 and isinstance(armonicos, int)
                and armonicos >= 1 and 0 <= fraccion < 1):
            raise ValueError("Especificación de señal bayesiana no válida")
        for arm in range(1, armonicos + 1):
            beta_mu.extend((0., 0.))
            beta_sigma.extend((float(escala_senial), float(escala_senial)))
            beta_labels.extend((f"Seno_{indice_senal}_arm{arm}",
                                f"Coseno_{indice_senal}_arm{arm}"))

    return {
        "tendencia": tendencia,
        "seniales": tuple(seniales),
        "mu_beta": np.asarray(beta_mu, dtype=float),
        "sig_beta": np.asarray(beta_sigma, dtype=float),
        "labels_beta": beta_labels,
        "jitter_priors": jitter_priors,
        "labels_nonlineales": [f"Jitter_{n}" for n in datos["nombres"]]
                               + [f"Periodo_{k}_dias" for k in range(1, len(seniales) + 1)],
    }


def transformar_prior_bayes(cubo, config, ninst):
    """Transformar el hipercubo unitario de dynesty en jitters y periodos.

    Los jitters siguen distribuciones half-normal; los periodos, uniformes
    alrededor de un valor inicial. Evitamos los extremos 0 y 1 para que
    la funcion cuantíl (ppf) de la half-normal sea finita.
    """
    cubo = np.asarray(cubo, dtype=float)
    if len(cubo) != ninst + len(config["seniales"]):
        raise ValueError("Dimension incorrecta del cubo de priors")
    q = np.clip(cubo[:ninst], 1e-12, 1. - 1e-12)
    jitters = np.array([prior.ppf(qi) for prior, qi in
                        zip(config["jitter_priors"], q)])
    periodos = np.array([p0 * (1 - frac + 2 * frac * qi)
                         for (p0, _, frac), qi in
                         zip(config["seniales"], cubo[ninst:])])
    return np.concatenate((jitters, periodos))


def logverosimilitud_marginal(datos, config, theta, devolver_condicional=False):
    """ln p(y | jitter, periodos), integrando exactamente offsets/senos.

    Sea y|beta ~ N(X beta, D) y beta ~ N(mu, Lambda), con Lambda diagonal.
    La integral sobre beta es gaussiana. Usamos identidad de Woodbury:

      A = Lambda^-1 + X^T D^-1 X
      b = X^T D^-1(y - X mu)
      ln|Cov(y)| = ln|D| + ln|Lambda| + ln|A|
      chi2_marg = r^T D^-1 r - b^T A^-1 b

    Asi dynesty solo muestrea 4-7 jitters y 0-2 periodos, en vez de
    10-20 parametros. Es una marginalizacion EXACTA para el modelo lineal
    y los priors gaussianos indicados, no una aproximacion tipo BIC.

    Si devolver_condicional=True se devuelve tambien el valor esperado
    posterior de los coeficientes lineales, y su covarianza condicional.
    """
    from scipy.linalg import cho_factor, cho_solve

    theta = np.asarray(theta, dtype=float)
    ninst = datos["ninst"]
    jitters = theta[:ninst]
    periodos = theta[ninst:]
    if (len(theta) != ninst + len(config["seniales"])
            or np.any(~np.isfinite(theta)) or np.any(jitters < 0)
            or np.any(periodos <= 0)):
        return -np.inf if not devolver_condicional else (-np.inf, None, None)

    seniales = tuple((float(p), conf[1]) for p, conf in
                     zip(periodos, config["seniales"]))
    X, _ = matriz_modelo(datos, config["tendencia"], seniales)
    mu = config["mu_beta"]
    sigma_beta = config["sig_beta"]
    if X.shape[1] != len(mu):
        raise ValueError("El numero de columnas no coincide con el prior lineal")

    var = datos["e"]**2 + jitters[datos["indice"]]**2
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        return -np.inf if not devolver_condicional else (-np.inf, None, None)
    pesos = 1. / var
    residuos_prior = datos["y"] - X @ mu
    Xw = X * np.sqrt(pesos[:, None])
    A = Xw.T @ Xw + np.diag(1. / sigma_beta**2)
    b = X.T @ (residuos_prior * pesos)

    try:
        chol, low = cho_factor(A, lower=True, check_finite=False)
        correccion = cho_solve((chol, low), b, check_finite=False)
        logdet_A = 2 * np.log(np.diag(chol)).sum()
    except (ValueError, np.linalg.LinAlgError):
        return -np.inf if not devolver_condicional else (-np.inf, None, None)

    chi2 = np.dot(residuos_prior * pesos, residuos_prior) - np.dot(b, correccion)
    # Estabilizacion para error de redondeo (chi2 teoricamente no negativo).
    chi2 = max(chi2, 0.)
    logdet = np.log(var).sum() + np.log(sigma_beta**2).sum() + logdet_A
    logl = -0.5 * (datos["n"] * np.log(2 * np.pi) + logdet + chi2)

    if not devolver_condicional:
        return float(logl)
    cov_beta = cho_solve((chol, low), np.eye(len(mu)), check_finite=False)
    media_beta = mu + correccion
    return float(logl), media_beta, cov_beta


def graficar_posterior_bayes(muestras, config, nombre, destino):
    """Dibujar marginales de periodos y jitters, utiles para la memoria.

    Los histogramas visualizan incertidumbre y posible multimodalidad.
    El grafico de jitters permite comparar la dispersion adicional por
    instrumento SIN realizar ajustes fisicos instrumento por instrumento.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ninst = len(config["jitter_priors"])
    nperiodos = len(config["seniales"])

    if nperiodos > 0:
        fig, axes = plt.subplots(1, nperiodos, figsize=(5 * nperiodos, 4),
                                 squeeze=False)
        for i, ax in enumerate(axes[0]):
            valores = muestras[:, ninst + i]
            ax.hist(valores, bins=35, density=True, histtype="stepfilled", alpha=0.65)
            ax.axvline(np.median(valores), color="black", ls="--",
                       label=f"Mediana = {np.median(valores):.2f} d")
            ax.set_xlabel(f"Periodo {i + 1} (dias)")
            ax.set_ylabel("Densidad posterior")
            ax.legend(fontsize=9)
        fig.suptitle(f"Distribuciones posteriores de periodos: {nombre}")
        fig.tight_layout()
        fig.savefig(destino / f"posterior_periodos_{nombre}.png", dpi=220)
        fig.savefig(destino / f"posterior_periodos_{nombre}.pdf")
        plt.close(fig)

    p16, p50, p84 = np.percentile(muestras[:, :ninst], [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(max(8, ninst + 2), 4))
    ax.errorbar(np.arange(ninst), p50,
                yerr=np.vstack((p50 - p16, p84 - p50)), fmt="o", capsize=3)
    ax.set_xticks(np.arange(ninst))
    ax.set_xticklabels([n.replace("Jitter_", "") for n in
                        config["labels_nonlineales"][:ninst]], rotation=45,
                       ha="right")
    ax.set_ylabel("Jitter (unidades del observable)")
    ax.set_title(f"Ruido adicional por instrumento: {nombre}")
    fig.tight_layout()
    fig.savefig(destino / f"posterior_jitters_{nombre}.png", dpi=220)
    fig.savefig(destino / f"posterior_jitters_{nombre}.pdf")
    plt.close(fig)

    # Corner es ilustrativo, pero no se vuelve una dependencia obligatoria.
    try:
        import corner
    except ImportError:
        return
    fig = corner.corner(muestras, labels=config["labels_nonlineales"],
                        show_titles=True, quantiles=(0.16, 0.5, 0.84))
    fig.savefig(destino / f"corner_{nombre}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ajustar_modelo_bayes(datos, nombre, config, destino,
                         nlive=160, dlogz=0.15, seed=42):
    """Ajustar con dynesty y guardar evidencia y parametros posteriores.

    No se importan dynesty ni corner al arrancar el programa: solo son
    necesarios si se solicita explicitamente el muestreo bayesiano.
    La evidencia depende de los priors; no es intercambiable con el BIC.
    """
    try:
        import dynesty
        from dynesty.utils import resample_equal
    except ImportError as exc:
        raise ImportError("Falta dynesty: instala con 'pip install dynesty'") from exc

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    ndim = datos["ninst"] + len(config["seniales"])
    if nlive < 2 * ndim:
        raise ValueError("Usa al menos el doble de live points que de dimensiones")

    def _lnlike(params):
        return logverosimilitud_marginal(datos, config, params)

    def _transformar(cubo):
        return transformar_prior_bayes(cubo, config, datos["ninst"])

    # CPU=1 por defecto: evita problemas con procesos y Spyder/Windows.
    sampler = dynesty.NestedSampler(
        _lnlike, _transformar, ndim=ndim, nlive=nlive,
        sample="rwalk", rstate=np.random.default_rng(seed))
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    resultado = sampler.results
    logz = float(resultado.logz[-1])
    logzerr = float(resultado.logzerr[-1])

    # En Nested Sampling las muestras tienen pesos distintos. Para
    # percentiles, convertimos a muestras de peso igual sin sesgo.
    from scipy.special import logsumexp
    pesos = np.exp(resultado.logwt - logsumexp(resultado.logwt))
    muestras = resample_equal(resultado.samples, pesos,
                              rstate=np.random.default_rng(seed + 1))
    percentiles = np.percentile(muestras, [16, 50, 84], axis=0)

    with (destino / f"posterior_{nombre}.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["parametro", "p16", "mediana", "p84"])
        for etiqueta, (p16, p50, p84) in zip(config["labels_nonlineales"], percentiles.T):
            escritor.writerow([etiqueta, p16, p50, p84])

    # Para cada muestra calculamos la media posterior condicional de los
    # coeficientes lineales. No es aun una muestra de su incertidumbre
    # completa, por eso sumamos tambien su covarianza esperada.
    medias = []
    varianzas = []
    for parametros in muestras:
        _, media, cov = logverosimilitud_marginal(
            datos, config, parametros, devolver_condicional=True)
        medias.append(media)
        varianzas.append(np.diag(cov))
    medias = np.asarray(medias)
    varianzas = np.asarray(varianzas)
    media_lineal = medias.mean(axis=0)
    varianza_lineal = varianzas.mean(axis=0) + medias.var(axis=0)

    with (destino / f"lineales_{nombre}.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["parametro", "media_posterior", "desviacion_posterior"])
        for etiqueta, m, v in zip(config["labels_beta"], media_lineal, varianza_lineal):
            escritor.writerow([etiqueta, m, np.sqrt(max(v, 0.))])

    np.savetxt(destino / f"muestras_{nombre}.csv", muestras, delimiter=",",
               header=",".join(config["labels_nonlineales"]), comments="")
    graficar_posterior_bayes(muestras, config, nombre, destino)
    with (destino / f"evidencia_{nombre}.csv").open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["modelo", "lnZ", "incertidumbre_lnZ", "nlive", "dlogz"])
        escritor.writerow([nombre, logz, logzerr, nlive, dlogz])
    print(f"{nombre}: lnZ = {logz:.2f} +/- {logzerr:.2f}")
    return {"nombre": nombre, "lnZ": logz, "lnZerr": logzerr,
            "muestras": muestras, "lineales": media_lineal}


def main_bayes(periodo_rot, periodo_largo=None, nlive=160, dlogz=0.15):
    """Comparar modelos RV bayesianos globales tras elegir periodos candidatos.

    La eleccion de periodos a partir de los mismos datos es exploratoria.
    Para un Bayes factor plenamente confirmatorio habria que justificar
    los priors independientemente y examinar alias y ruido correlacionado.
    """
    try:
        rjd, rv, srv, inst = leer_csv(
            "data/Epsilon_Eridani_RVdata.csv", "RV (m/s)", "err RV (m/s)")
    except FileNotFoundError as exc:
        raise SystemExit("No se encuentra el CSV: ejecuta desde la raiz del repositorio") from exc
    datos = preparar_datos(rjd, rv, srv, inst)
    modelos = [
        ("RV_BASE", configuracion_bayes(datos, tendencia=True)),
        ("RV_ROT", configuracion_bayes(datos, tendencia=True,
                  seniales=((periodo_rot, 2, 0.15),))),
    ]
    if periodo_largo is not None:
        modelos.append(("RV_ROT+LARGO", configuracion_bayes(
            datos, tendencia=True, seniales=((periodo_rot, 2, 0.15),
                                              (periodo_largo, 1, 0.25)))))

    salida = DIRECTORIO_RESULTADOS / "RV_bayes"
    resultados = [ajustar_modelo_bayes(datos, nombre, conf, salida,
                                     nlive=nlive, dlogz=dlogz)
                  for nombre, conf in modelos]
    base = resultados[0]
    print("\nComparación de evidencias (respecto al modelo de referencia):")
    for r in resultados:
        print(f"{r['nombre']}: delta lnZ = {r['lnZ']-base['lnZ']:.2f}")
    return resultados



# -----------------------------------------------------------------------------
# 10. DIAGNOSTICOS: VENTANA, FASE, RESIDUOS Y ESTABILIDAD INSTRUMENTAL
# -----------------------------------------------------------------------------
def ventana_espectral(tiempos, frecuencias, bloque=256):
    """Calcular |sum(exp(2*pi*i*f*t))/N|^2 para el muestreo real.

    La ventana NO utiliza RV ni S_MW: solo las fechas de observacion.
    Una estructura a f = 1/365.25 d^-1 o cerca de f = 1 d^-1 indica
    que la cobertura temporal puede producir alias anuales o diarios.

    Los calculos se procesan en bloques para no crear una matriz inmensa
    (n_frecuencias x n_observaciones) en la memoria del ordenador.
    """
    tiempos = np.asarray(tiempos, float)
    frecuencias = np.asarray(frecuencias, float)
    if tiempos.ndim != 1 or frecuencias.ndim != 1 or len(tiempos) < 2:
        raise ValueError("Ventana: se necesitan dos vectores unidimensionales")

    # Centrar el tiempo no cambia el modulo al cuadrado, pero evita fases
    # enormes que provocan perdida de precision numerica.
    dt = tiempos - np.median(tiempos)
    potencia = np.empty(len(frecuencias))
    for inicio in range(0, len(frecuencias), bloque):
        fin = inicio + bloque
        fase = 2 * np.pi * frecuencias[inicio:fin, None] * dt[None, :]
        parte_real = np.cos(fase).mean(axis=1)
        parte_imaginaria = np.sin(fase).mean(axis=1)
        potencia[inicio:fin] = parte_real**2 + parte_imaginaria**2
    return potencia


def picos_periodograma(frecuencias, potencia, numero=8, separacion=3):
    """Obtener los maximos locales mejor separados de un espectro.

    Un pico local no es automaticamente significativo. La tabla resultante
    es una lista de candidatos para inspeccion visual, no una tabla FAP.
    """
    from scipy.signal import find_peaks

    indices, _ = find_peaks(np.asarray(potencia), distance=separacion)
    indices = sorted(indices, key=lambda i: potencia[i], reverse=True)[:numero]
    return [(float(frecuencias[i]), float(potencia[i])) for i in indices]


def guardar_ventana(datos_rv, datos_s, carpeta):
    """Representar lado a lado las ventanas RV y S_MW.

    Se usan DOS rangos: bajo (donde esta el alias anual) y alrededor
    de 1 ciclo/dia (donde estan los alias diarios). Mostrar ambos rangos
    evita confundir un pico del muestreo con una periodicidad estelar.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    # Una malla densa a frecuencia baja permite resolver 1/365.25 d^-1;
    # una segunda malla se concentra en el alias de 1 dia.
    frecuencias_bajas = np.linspace(1 / 10000., 0.05, 8000)
    frecuencias_diarias = np.linspace(0.90, 1.10, 5000)

    fig, ejes = plt.subplots(2, 1, figsize=(10, 7))
    colores = ("#255a92", "#bd5b22")
    registros = []
    for serie, datos, color in zip(("RV", "S_MW"), (datos_rv, datos_s), colores):
        baja = ventana_espectral(datos["t"], frecuencias_bajas)
        diaria = ventana_espectral(datos["t"], frecuencias_diarias)
        ejes[0].plot(frecuencias_bajas, baja, lw=0.9, label=serie, color=color)
        ejes[1].plot(frecuencias_diarias, diaria, lw=0.9, label=serie, color=color)

        # Guardamos las curvas completas: el origen de cada grafica puede
        # comprobarse sin tener que volver a ejecutar el programa.
        for rango, frecs, potencias in (("baja", frecuencias_bajas, baja),
                                        ("diaria", frecuencias_diarias, diaria)):
            np.savetxt(carpeta / f"ventana_{serie}_{rango}.csv",
                       np.column_stack((frecs, potencias)), delimiter=",",
                       header="frecuencia_dia_inv,potencia_ventana", comments="")
            for f, potencia in picos_periodograma(frecs, potencias, 12):
                registros.append([serie, rango, f, 1 / f, potencia])

    ejes[0].axvline(1 / DIAS_POR_ANIO, color="black", ls="--", lw=1.1,
                    label="1/año")
    ejes[1].axvline(1.0, color="black", ls="--", lw=1.1, label="1/día")
    for ax in ejes:
        ax.set_ylabel("Potencia de ventana (0–1)")
        ax.set_xlabel("Frecuencia (ciclos/día)")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(alpha=0.15)
    ejes[0].set_title("Ventana espectral: muestreo a bajas frecuencias")
    ejes[1].set_title("Ventana espectral: alias próximo a un día")
    fig.tight_layout()
    fig.savefig(carpeta / "ventana_espectral.png", dpi=220)
    fig.savefig(carpeta / "ventana_espectral.pdf")
    plt.close(fig)

    with (carpeta / "picos_ventana.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["serie", "rango", "frecuencia_dia_inv", "periodo_dias", "potencia"])
        writer.writerows(registros)
    return registros


def _componente_senal(datos, modelo, indice_senal, tiempos):
    """Evaluar una senal sinusoidal del modelo, sin tendencia ni offsets.

    indice_senal=0 es la senal corta; indice_senal=1 es la senal larga.
    Los coeficientes se leen en el mismo orden que matriz_modelo().
    """
    tiempos = np.asarray(tiempos, float)
    if indice_senal < 0 or indice_senal >= len(modelo["seniales"]):
        raise IndexError("La señal solicitada no existe en el modelo")

    pos = datos["ninst"] + int(modelo["tendencia"])
    for k, (periodo, armonicos) in enumerate(modelo["seniales"]):
        if k == indice_senal:
            senal = np.zeros(len(tiempos))
            for h in range(1, armonicos + 1):
                angulo = 2 * np.pi * h * (tiempos - datos["t0"]) / periodo
                senal += modelo["beta"][pos] * np.sin(angulo)
                senal += modelo["beta"][pos + 1] * np.cos(angulo)
                pos += 2
            return senal
        pos += 2 * armonicos
    raise RuntimeError("No se ha podido localizar la señal")


def graficar_fase(datos, modelo, indice_senal, observable, ruta):
    """Plegar la serie global en fase, retirando TODAS las demas componentes.

    Es crucial corregir el offset instrumental, la tendencia y la otra
    sinusoide antes del plegado. De lo contrario, se introduciria una
    dispersion artificial y podria parecer que la senal no existe.
    """
    if indice_senal >= len(modelo["seniales"]):
        return
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    periodo, _ = modelo["seniales"][indice_senal]

    # Modelo total = offsets + tendencia + todas las senales.
    # Al quitar todas las componentes menos la escogida, los puntos
    # plegados representan SOLO la forma periodica que evaluamos.
    parte_objetivo = _componente_senal(datos, modelo, indice_senal, datos["t"])
    valores_fase = datos["y"] - (modelo["prediccion"] - parte_objetivo)
    fase = ((datos["t"] - datos["t0"]) / periodo + 0.5) % 1 - 0.5

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for j, nombre in enumerate(datos["nombres"]):
        mascara = datos["indice"] == j
        x = fase[mascara]
        y = valores_fase[mascara]
        # Repetimos un ciclo para ver mejor la continuidad en los bordes.
        ax.plot(np.r_[x, x + 1], np.r_[y, y], ".", ms=2.3,
                alpha=0.50, label=nombre)

    grid_fase = np.linspace(-0.5, 1.5, 900)
    # Cualquier t=t0+P*fase tiene exactamente la fase que se esta dibujando.
    curva = _componente_senal(datos, modelo, indice_senal,
                              datos["t0"] + periodo * grid_fase)
    ax.plot(grid_fase, curva, color="black", lw=2, label="Señal global ajustada")
    ax.set_xlabel("Fase (ciclos)")
    ax.set_ylabel(f"{observable} sin otras componentes")
    ax.set_title(f"Serie completa plegada en fase: P = {periodo:.3f} días")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(alpha=0.12)
    fig.subplots_adjust(bottom=0.25)
    fig.savefig(ruta.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def diagnosticar_residuos(datos, modelo, observable, carpeta):
    """Buscar periodos que aun mejoren el modelo elegido.

    Reutilizamos buscar_periodos(): las nuevas sinusoides se prueban
    sobre la serie GLOBAL, DESPUES de proyectar las componentes del
    modelo actual. Esto equivale a estudiar los residuos condicionados
    al ajuste y no a reestimar periodos por instrumento.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    limite_largo = min(10000., np.ptp(datos["t"]))
    rangos = (("corto", 5., 30.), ("largo", 200., limite_largo))
    resultados = []
    for etiqueta, pmin, pmax in rangos:
        if pmax <= pmin:
            continue
        # Se reduce moderadamente el numero de frecuencias para hacer que
        # los diagnosticos puedan ejecutarse en un ordenador domestico.
        nfreq = 3800 if etiqueta == "corto" else 1100
        busc = buscar_periodos(datos, modelo, pmin, pmax, nfrecuencias=nfreq)
        graficar_periodograma(busc, f"{observable}: señal residual {etiqueta}",
                              carpeta / f"residuos_{etiqueta}")
        resultados.append((etiqueta, float(busc["periodo"]),
                           float(np.max(busc["potencia"]))))

    with (carpeta / "picos_residuales.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["escala", "periodo_dias", "mejora_chi2_fraccional_maxima"])
        w.writerows(resultados)
    return resultados


def _semiamplitud(datos, modelo, indice_senal=1):
    """Calcular sqrt(a^2+b^2) de una señal de un solo armónico."""
    if indice_senal >= len(modelo["seniales"]):
        return float("nan")
    if modelo["seniales"][indice_senal][1] != 1:
        return float("nan")
    pos = datos["ninst"] + int(modelo["tendencia"])
    for k, (_, armonicos) in enumerate(modelo["seniales"]):
        if k == indice_senal:
            return float(np.hypot(modelo["beta"][pos], modelo["beta"][pos + 1]))
        pos += 2 * armonicos
    return float("nan")


def robustez_instrumentos_rv(datos_originales, periodo_corto, periodo_largo,
                              carpeta):
    """Repetir el ajuste global quitando un instrumento cada vez (LOIO).

    Comparamos dentro de CADA subconjunto un modelo ROT2 frente a ROT2+
    LARGO. Los BIC de subconjuntos de distinto tamaño NO deben compararse
    directamente. También evaluamos la señal en el periodo largo FIJO
    para comprobar si su amplitud depende excesivamente de un instrumento.

    No ajustamos un planeta distinto por instrumento: cada fila utiliza
    todas las mediciones de los otros seis instrumentos conjuntamente.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for excluido in ("NINGUNO", *datos_originales["nombres"]):
        conservar = np.ones(datos_originales["n"], dtype=bool)
        if excluido != "NINGUNO":
            conservar = datos_originales["indice"] != np.where(
                datos_originales["nombres"] == excluido)[0][0]
        sub = preparar_datos(*(datos_originales[k][conservar]
                              for k in ("t", "y", "e")),
                              datos_originales["nombres"][datos_originales["indice"][conservar]])

        # Se mantiene el periodo corto determinado externamente con S_MW.
        rot2 = ajustar_modelo(sub, "ROT2", tendencia=True,
                              seniales=((periodo_corto, 2),))
        # Hipotesis A: la señal larga tiene el periodo global conocido.
        fijo = ajustar_modelo(sub, "ROT2+LARGO_FIJO", tendencia=True,
                              seniales=((periodo_corto, 2), (periodo_largo, 1)))
        # Hipotesis B: estimamos otra vez libremente el periodo largo.
        # Un cambio brusco al retirar un instrumento seria una alerta.
        busc = buscar_periodos(sub, rot2, 200., min(10000., np.ptp(sub["t"])),
                               nfrecuencias=1500)
        p_libre = float(busc["periodo"])
        libre = ajustar_modelo(sub, "ROT2+LARGO_LIBRE", tendencia=True,
                               seniales=((periodo_corto, 2), (p_libre, 1)),
                               periodos_estimados=1)
        filas.append((excluido, sub["n"], sub["ninst"],
                      periodo_largo, _semiamplitud(sub, fijo),
                      rot2["bic"] - fijo["bic"],
                      p_libre, _semiamplitud(sub, libre),
                      rot2["bic"] - libre["bic"]))
        print(f"  RV quitando {excluido}: P_largo_libre={p_libre:.1f} d, "
              f"K_fijo={_semiamplitud(sub, fijo):.2f} m/s")

    with (carpeta / "robustez_RV.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instrumento_excluido", "N", "num_instrumentos",
                    "periodo_global_fijo_d", "K_fijo_m_s", "delta_BIC_fijo",
                    "periodo_libre_d", "K_libre_m_s", "delta_BIC_libre"])
        w.writerows(filas)

    # Representacion sencilla: si todas las estimaciones del periodo son
    # parecidas, esperariamos puntos alrededor de la linea discontinua.
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    etiquetas = [v[0] for v in filas]
    pos = np.arange(len(filas))
    a.scatter(pos, [v[6] for v in filas])
    a.axhline(periodo_largo, ls="--", color="black", label="Periodo global")
    a.set_ylabel("Mejor periodo largo (días)")
    a.legend()
    b.scatter(pos, [v[4] for v in filas], label="P fijado al global")
    b.scatter(pos, [v[7] for v in filas], marker="x", label="P reestimado")
    b.set_ylabel("Semiamplitud K (m/s)")
    b.set_xticks(pos)
    b.set_xticklabels(etiquetas, rotation=45, ha="right")
    b.legend()
    fig.suptitle("Sensibilidad de la señal larga a cada conjunto instrumental")
    fig.tight_layout()
    fig.savefig(carpeta / "robustez_RV.png", dpi=220)
    fig.savefig(carpeta / "robustez_RV.pdf")
    plt.close(fig)
    return filas


def robustez_instrumentos_actividad(datos_originales, carpeta):
    """Comprobar periodos de actividad al omitir uno de sus 4 instrumentos.

    Solo se usan modelos fisicos GLOBALMENTE ajustados a las otras series;
    las etiquetas identifican el conjunto excluido, no ajustes individuales.
    """
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for excluido in ("NINGUNO", *datos_originales["nombres"]):
        conservar = np.ones(datos_originales["n"], dtype=bool)
        if excluido != "NINGUNO":
            j = np.where(datos_originales["nombres"] == excluido)[0][0]
            conservar = datos_originales["indice"] != j
        sub = preparar_datos(*(datos_originales[k][conservar]
                              for k in ("t", "y", "e")),
                              datos_originales["nombres"][datos_originales["indice"][conservar]])
        base = ajustar_modelo(sub, "BASE", tendencia=True)
        corto = buscar_periodos(sub, base, 5., 30., nfrecuencias=2400)
        p_corto = float(corto["periodo"])
        rot2 = ajustar_modelo(sub, "ROT2", tendencia=True,
                              seniales=((p_corto, 2),), periodos_estimados=1)
        largo = buscar_periodos(sub, rot2, 200., min(5000., np.ptp(sub["t"])),
                                nfrecuencias=1100)
        p_largo = float(largo["periodo"])
        ciclo = ajustar_modelo(sub, "ROT2+CICLO", tendencia=True,
                               seniales=((p_corto, 2), (p_largo, 1)),
                               periodos_estimados=2)
        filas.append((excluido, sub["n"], p_corto, p_largo,
                      rot2["bic"] - ciclo["bic"]))
        print(f"  S_MW quitando {excluido}: P_corto={p_corto:.3f} d, "
              f"P_largo={p_largo:.1f} d")

    with (carpeta / "robustez_SMW.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instrumento_excluido", "N", "periodo_corto_d",
                    "periodo_largo_d", "delta_BIC_ciclo"])
        w.writerows(filas)
    return filas


def comprobar_alias_anual(periodo1, periodo2, carpeta):
    """Cuantificar la coincidencia entre dos picos y un alias anual.

    Los dos picos PODRIAN corresponder a la misma señal si su diferencia
    de frecuencia coincide con un maximo de la ventana a 1/año.
    La igualdad numerica es una alerta, no la demostracion de un alias.
    """
    f1 = 1 / float(periodo1)
    f2 = 1 / float(periodo2)
    dif = abs(f1 - f2)
    anual = 1 / DIAS_POR_ANIO
    filas = [("periodo_actividad_d", periodo1),
             ("periodo_rv_d", periodo2),
             ("diferencia_frecuencia_d_inv", dif),
             ("frecuencia_anual_d_inv", anual),
             ("diferencia_relativa_f_anual", abs(dif - anual) / anual),
             ("alias_esperado_mas_d", 1 / (f1 + anual)),
             ("alias_esperado_menos_d", 1 / abs(f1 - anual))]
    with (Path(carpeta) / "alias_anual.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cantidad", "valor"])
        w.writerows(filas)
    print("\nComparacion de alias anual:")
    print(f"  Delta f = {dif:.8f} /dia ; f_anio = {anual:.8f} /dia")
    if abs(dif - anual) / anual < 0.05:
        print("  Diferencia compatible con un alias anual; verificar la ventana.")
    else:
        print("  La diferencia NO coincide con un alias anual dentro del 5%.")
    return filas


def main_diagnosticos():
    """Realizar la bateria de pruebas sobre las dos series REALES completas.

    Desde la raiz: python -m src.model_data --diagnosticos
    O ejecutar ejecutar_diagnosticos.py desde Spyder.
    Las salidas nuevas se guardan en results/modelado_global/diagnosticos.
    """
    # Crear y mostrar la ruta ANTES de los ajustes, que pueden tardar.
    carpeta = DIRECTORIO_RESULTADOS / "diagnosticos"
    carpeta.mkdir(parents=True, exist_ok=True)
    print("\nModo DIAGNOSTICOS activado", flush=True)
    print(f"Guardando resultados en: {carpeta.resolve()}", flush=True)

    rjd, rv, srv, inst_rv = leer_csv(
        "data/Epsilon_Eridani_RVdata.csv", "RV (m/s)", "err RV (m/s)")
    ts, s, ss, inst_s = leer_csv(
        "data/Epsilon_Eridani_SMW.csv", "S_MW x1000", "sig S_MW x1000",
        factor=1 / 1000.)

    # Empezamos por S_MW: el periodo corto de actividad se utiliza como
    # hipotesis de rotacion, independientemente de los maximos RV.
    actividad = fit_activity_models(ts, s, ss, inst_s)
    rv_ajuste = fit_rv_sine(rjd, rv, srv, inst_rv,
                           period_rot=actividad["periodo"],
                           period_cycle=actividad["periodo_ciclo"])
    datos_s = preparar_datos(ts, s, ss, inst_s)
    datos_rv = preparar_datos(rjd, rv, srv, inst_rv)

    # 1. Las ventanas indican posibles frecuencias espurias por muestreo.
    print("\n[1/4] Calculando las ventanas espectrales...")
    guardar_ventana(datos_rv, datos_s, carpeta / "ventana")

    # 2. Las curvas de fase permiten ver si ambas señales estan realmente
    # soportadas por observaciones repartidas por todo el ciclo.
    print("[2/4] Generando curvas plegadas en fase...")
    for datos, ajuste, etiqueta in ((datos_rv, rv_ajuste, "RV"),
                                    (datos_s, actividad, "SMW")):
        modelo = ajuste["mejor"]
        for indice_senal, nombre in enumerate(("corto", "largo")):
            if indice_senal < len(modelo["seniales"]):
                graficar_fase(datos, modelo, indice_senal, etiqueta,
                             carpeta / "fases" / f"{etiqueta}_{nombre}")

    # 3. Un periodo fuerte en los residuos advierte de un modelo incompleto.
    print("[3/4] Buscando periodicidades en los residuos...")
    diagnosticar_residuos(datos_rv, rv_ajuste["mejor"], "RV", carpeta / "residuos_RV")
    diagnosticar_residuos(datos_s, actividad["mejor"], "S_MW",
                          carpeta / "residuos_SMW")

    # 4. Retirar uno de los instrumentos debe permitir comprobar si una
    # señal esta dominada por un conjunto concreto de observaciones.
    print("[4/4] Comprobando estabilidad al retirar instrumentos...")
    if rv_ajuste["periodo_largo"] is not None:
        robustez_instrumentos_rv(datos_rv, actividad["periodo"],
                                rv_ajuste["periodo_largo"], carpeta / "robustez")
    robustez_instrumentos_actividad(datos_s, carpeta / "robustez")

    # La comparacion correcta: el pico RV independiente frente al de S_MW.
    # El ajuste final de RV utiliza el corto de actividad deliberadamente.
    base_rv = next(m for m in rv_ajuste["resultados"]
                   if m["nombre"] == "RV_OFFSETS+LINEAL")
    corto_rv = buscar_periodos(datos_rv, base_rv, 5., 30., nfrecuencias=7500)
    comprobar_alias_anual(actividad["periodo"], corto_rv["periodo"], carpeta)

    print(f"\nDiagnosticos guardados en: {carpeta}")
    print("Nota: los periodogramas de mejora de chi2 no incorporan FAP.")
    return {"actividad": actividad, "rv": rv_ajuste}


# -----------------------------------------------------------------------------
# 11. MODELOS FISICOS BAYESIANOS M0, M1 Y M2 CON RUIDO CORRELACIONADO
# -----------------------------------------------------------------------------
# Esta sección es independiente de --bayes (modelos sinusoidales anteriores).
# Todos los modelos contienen el mismo GP y los mismos parámetros instrumentales.
# M0 = actividad (GP) + offsets + tendencia.
# M1 = M0 + un planeta con órbita circular.
# M2 = M0 + un planeta con órbita kepleriana excéntrica.
#
# El GP contiene dos escalas: rotación cuasiperiódica (RotationTerm) y variación
# lenta estocástica (Matern32Term). NO se interpreta la segunda como una órbita.
# La likelihood integra exactamente los coeficientes lineales con priors normales
# propios, y dynesty muestrea los hiperparámetros y la órbita, si existe.


def velocidad_kepleriana(t, periodo, amplitud, excentricidad, omega, fase, t0):
    """Velocidad radial kepleriana K[cos(nu+omega) + e cos(omega)].

    t, t0 y periodo están en días; K en m/s; omega en radianes.
    fase es la fracción del periodo transcurrida hasta el siguiente periastro
    desde t0: M(t)=2pi[(t-t0)/P - fase]. Se resuelve la ecuación
    M=E-e sin(E) mediante Newton y después se calcula la anomalía verdadera.
    
    Para e=0 el resultado debe ser una sinusoide. Ninguna componente orbital
    se ajusta por instrumento: solo los offsets y jitters dependen de él.
    """
    t = np.asarray(t, dtype=float)
    if periodo <= 0 or amplitud < 0 or not 0 <= excentricidad < 1:
        raise ValueError("Parámetros orbitales fuera de rango físico")
    M = 2.0 * np.pi * np.remainder((t - t0) / periodo - fase, 1.0)
    E = M.copy()
    if excentricidad > 0:
        # La aproximación E=M+e sin(M) acelera el método para órbitas moderadas.
        E += excentricidad * np.sin(M)
        for _ in range(35):
            paso = (E - excentricidad * np.sin(E) - M) / (
                1.0 - excentricidad * np.cos(E))
            E -= paso
            if np.max(np.abs(paso)) < 1e-12:
                break
    nu = np.arctan2(np.sqrt(1.0 - excentricidad**2) * np.sin(E),
                    np.cos(E) - excentricidad)
    return amplitud * (np.cos(nu + omega) + excentricidad * np.cos(omega))


def _datos_para_gp(datos):
    """Restar medianas por instrumento para estabilizar la álgebra numérica.

    Es una traslación determinista de los datos, no un ajuste adicional ni un
    borrado de medidas. Se preservan todas las épocas, errores y etiquetas.
    Los offsets absolutos se reconstruyen sumando estas referencias.
    """
    centros = np.array([np.median(datos["y"][datos["indice"] == i])
                        for i in range(datos["ninst"])])
    y_centrado = datos["y"] - centros[datos["indice"]]
    # Design matrix de parámetros lineales comunes a M0, M1 y M2.
    X = np.column_stack(
        [(datos["indice"] == i).astype(float) for i in range(datos["ninst"])]
        + [datos["t_anios"]]
    )
    # Priors propios: N(0, 300 m/s) para offsets relativos y N(0,20 m/s/año)
    # para pendiente. Su centro se define usando medianas observadas:
    # es una elección parcialmente informada por los datos (empirical Bayes).
    # Se aplica idéntica a los tres modelos, pero al interpretar los Bayes
    # factors debe indicarse que quedan condicionados a este centrado.
    sig_beta = np.r_[np.repeat(300.0, datos["ninst"]), 20.0]
    return {"datos": datos, "centros": centros, "y_centrado": y_centrado,
            "X": X, "sig_beta": sig_beta}


def _especificacion_prior_gp(nombres, modelo, pmin, pmax, ajustes_priors=None):
    """Priors con soporte EXPLÍCITO para hacer comparable la evidencia.

    Un prior 'log' es uniforme en el logaritmo del parámetro positivo.
    Los límites deben declararse antes de ver la evidencia y NO se deben
    reajustar al máximo del periodograma para favorecer un modelo.
    """
    if modelo not in ("M0", "M1", "M2"):
        raise ValueError("Modelo inválido: elige M0, M1 o M2")
    if not 0 < pmin < pmax:
        raise ValueError("Debe cumplirse 0 < periodo_min < periodo_max")
    priors = [(f"jitter_{n}", "log", 0.01, 100.0) for n in nombres]
    priors += [
        ("gp_sigma_rot_m_s", "log", 0.05, 100.0),
        ("gp_periodo_rot_d", "lin", 8.0, 16.0),
        ("gp_Q0", "log", 0.2, 20.0),
        ("gp_dQ", "log", 0.2, 50.0),
        ("gp_frac_armonico", "lin", 0.02, 1.0),
        ("gp_sigma_lenta_m_s", "log", 0.05, 100.0),
        ("gp_escala_lenta_d", "log", 150.0, 5000.0),
    ]
    # Para las pruebas de sensibilidad solo se permiten los dos hiperpriors
    # identificados en la revisión de las posteriores originales. Cambiar el
    # prior obliga a utilizar una carpeta de resultados diferente.
    if ajustes_priors is not None:
        permitidos = {"gp_Q0", "gp_escala_lenta_d"}
        if set(ajustes_priors) - permitidos:
            raise ValueError("Solo se permite variar gp_Q0 y gp_escala_lenta_d")
        nuevos = []
        for nombre, tipo, minimo, maximo in priors:
            if nombre in ajustes_priors:
                limites = ajustes_priors[nombre]
                if len(limites) != 2:
                    raise ValueError(f"Limites invalidos de {nombre}")
                minimo, maximo = map(float, limites)
                if not (np.isfinite(minimo) and np.isfinite(maximo)
                        and 0 < minimo < maximo):
                    raise ValueError(f"Limites invalidos del prior {nombre}")
            nuevos.append((nombre, tipo, minimo, maximo))
        priors = nuevos

    if modelo != "M0":
        priors += [
            ("planeta_periodo_d", "log", float(pmin), float(pmax)),
            ("planeta_K_m_s", "lin", 0.0, 80.0),
            ("planeta_fase", "lin", 0.0, 1.0),
        ]
    if modelo == "M2":
        priors += [
            ("planeta_e", "lin", 0.0, 0.8),
            ("planeta_omega_rad", "lin", 0.0, 2.0 * np.pi),
        ]
    return priors


def _transformar_cubo_gp(cubo, priors):
    """Transformación del hipercubo unitario exigida por Nested Sampling."""
    cubo = np.asarray(cubo, dtype=float)
    if len(cubo) != len(priors):
        raise ValueError("La dimensión del cubo no coincide con los priors")
    valores = []
    for u, (nombre, tipo, minimo, maximo) in zip(cubo, priors):
        if tipo == "lin":
            valores.append(minimo + u * (maximo - minimo))
        elif tipo == "log":
            valores.append(np.exp(np.log(minimo) + u * np.log(maximo / minimo)))
        else:
            raise ValueError(f"Tipo de prior desconocido: {nombre}, {tipo}")
    return np.asarray(valores)


def _vector_orbita_gp(parametros, modelo, t, t0):
    """Construir RV orbital compartida: no hay parámetros planetarios por instrumento."""
    if modelo == "M0":
        return np.zeros_like(t)
    P = parametros["planeta_periodo_d"]
    K = parametros["planeta_K_m_s"]
    fase = parametros["planeta_fase"]
    if modelo == "M1":
        return velocidad_kepleriana(t, P, K, 0.0, 0.0, fase, t0)
    return velocidad_kepleriana(
        t, P, K, parametros["planeta_e"], parametros["planeta_omega_rad"],
        fase, t0)


def _evaluar_loglike_gp(theta, modelo, priors, contexto,
                        componentes=False):
    """log p(D|theta,M), integrando niveles cero y pendiente temporal.

    Si β ~ N(0,D) e y|β ~ N(Xβ+RV_orb,C_GP+diag(var_i)), se calcula

      log p(y|theta) = log N(y-RV_orb|0,C)
          + 1/2 b.T A^-1 b - 1/2(log|D| + log|A|)

    con A = X.T C^-1 X + D^-1, b = X.T C^-1(y-RV_orb).
    La normalización de los errores y el determinante GP se mantienen.
    No fijamos parámetros instrumentales usando los residuos de un planeta.
    """
    try:
        from celerite2 import GaussianProcess, terms
    except ImportError as exc:
        raise ImportError("Se necesita 'celerite2': pip install celerite2") from exc
    from scipy.linalg import cho_factor, cho_solve

    datos = contexto["datos"]
    pars = dict(zip((p[0] for p in priors), theta))
    jit = np.asarray(theta[:datos["ninst"]], dtype=float)
    variancia = datos["e"]**2 + jit[datos["indice"]]**2
    if not np.all(np.isfinite(variancia)) or np.any(variancia <= 0):
        return -np.inf if not componentes else None

    # Los dos kernels y sus priors son idénticos en M0, M1 y M2.
    rotacion = terms.RotationTerm(
        sigma=pars["gp_sigma_rot_m_s"],
        period=pars["gp_periodo_rot_d"],
        Q0=pars["gp_Q0"], dQ=pars["gp_dQ"],
        f=pars["gp_frac_armonico"])
    ciclo_estocastico = terms.Matern32Term(
        sigma=pars["gp_sigma_lenta_m_s"], rho=pars["gp_escala_lenta_d"])
    gp = GaussianProcess(rotacion + ciclo_estocastico, mean=0.0)
    orbita = _vector_orbita_gp(pars, modelo, datos["t"], datos["t0"])
    r = contexto["y_centrado"] - orbita

    try:
        gp.compute(datos["t"], diag=variancia)
        # apply_inverse resuelve C^-1 sobre una matriz completa sin invertir C.
        X = contexto["X"]
        resuelto = gp.apply_inverse(np.column_stack((r, X)))
        Cr, CX = resuelto[:, 0], resuelto[:, 1:]
        sig_beta = contexto["sig_beta"]
        Dinv = np.diag(1.0 / sig_beta**2)
        A = X.T @ CX + Dinv
        A = 0.5 * (A + A.T)  # limitar error de redondeo de simetría
        b = X.T @ Cr
        factor = cho_factor(A, lower=True, check_finite=False)
        beta = cho_solve(factor, b, check_finite=False)
        logdet_A = 2.0 * np.sum(np.log(np.diag(factor[0])))
        ajuste = 0.5 * np.dot(b, beta) - 0.5 * (
            np.sum(np.log(sig_beta**2)) + logdet_A)
        lnL = float(gp.log_likelihood(r) + ajuste)
        if not np.isfinite(lnL):
            return -np.inf if not componentes else None
    except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError):
        return -np.inf if not componentes else None

    if not componentes:
        return lnL
    # La predicción GP se condiciona a los datos, NO es una validación externa.
    r_gp = r - X @ beta
    gp_media = gp.predict(r_gp, return_var=False)
    return {
        "lnL": lnL, "coeficientes": beta, "orbita": orbita,
        "gp_media": gp_media,
        "residuos": r_gp - gp_media,
        "desviacion_efectiva": np.sqrt(variancia),
        "niveles_ajustados": beta[:datos["ninst"]] + contexto["centros"],
    }


def _prueba_funciones_kepler():
    """Pruebas deterministas sin librerías opcionales ni muestreo."""
    import scipy
    tiempo = np.linspace(0.0, 120.0, 1000)
    y0 = velocidad_kepleriana(tiempo, 30.0, 5.0, 0.0, 0.0, 0.23, 0.0)
    esperado = 5.0 * np.cos(2.0*np.pi*(tiempo/30.0-0.23))
    assert np.allclose(y0, esperado, atol=1e-10)
    # Ecuación en periastro: ν=0 => v=K(1+e) para omega=0.
    yperi = velocidad_kepleriana(np.array([0.0]), 30.0, 5.0, 0.55, 0., 0., 0.)
    assert np.allclose(yperi, [7.75], atol=1e-10)
    pr = [("p", "log", 1.0, 100.0), ("k", "lin", 0.0, 10.0)]
    assert np.allclose(_transformar_cubo_gp([0.5, 0.25], pr), [10.0, 2.5])
    return True


def _representar_posterior_gp(muestras, priors, modelo, salida):
    """Histogramas que ayudan a reconocer multimodalidad y límites del prior."""
    nombres = [p[0] for p in priors]
    elegidos = ["gp_periodo_rot_d", "gp_escala_lenta_d"]
    if modelo != "M0":
        elegidos += ["planeta_periodo_d", "planeta_K_m_s"]
    if modelo == "M2":
        elegidos += ["planeta_e", "planeta_omega_rad"]
    fig, axes = plt.subplots(len(elegidos), 1,
                             figsize=(8, 2.8 * len(elegidos)))
    for ax, n in zip(np.atleast_1d(axes), elegidos):
        valores = muestras[:, nombres.index(n)]
        ax.hist(valores, bins=45, histtype="stepfilled", alpha=0.6)
        p16, p50, p84 = np.percentile(valores, [16, 50, 84])
        ax.axvline(p50, ls="--", color="black")
        ax.set_xlabel(n.replace("_", " "))
        ax.set_title(f"P16={p16:.3g}, mediana={p50:.3g}, P84={p84:.3g}", fontsize=10)
    fig.suptitle(f"Posteriores bayesianas {modelo}: comprobar multimodalidad", fontsize=13)
    fig.tight_layout()
    fig.savefig(salida / "posteriores.png", dpi=200, bbox_inches="tight")
    fig.savefig(salida / "posteriores.pdf", bbox_inches="tight")
    plt.close(fig)


def _representar_componentes_gp(contexto, modelo, resultado, destino):
    """Mostrar contribuciones orbital y GP en los mismos datos de RV.

    El GP condicionado reproduce parte del ruido observado. No debe
    confundirse el pequeño residuo condicionado con un residuo predictivo.
    """
    datos = contexto["datos"]
    indice = datos["indice"]
    tiempo = datos["t"]
    serie = datos["y"] - resultado["niveles_ajustados"][indice]
    tendencia = resultado["coeficientes"][-1] * datos["t_anios"]
    orbita = resultado["orbita"]
    pred = tendencia + orbita + resultado["gp_media"]
    fig, ejes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for j, nombre in enumerate(datos["nombres"]):
        m = indice == j
        ejes[0].scatter(tiempo[m], serie[m], s=9, alpha=0.5, label=nombre)
    ejes[0].plot(tiempo, pred, ls="none", marker=".", ms=2,
                 color="black", label="Predicción condicional GP + órbita")
    ejes[0].plot(tiempo, tendencia + orbita, ls="none", marker=".", ms=2,
                 color="tomato", label="Tendencia + órbita (sin GP)")
    ejes[0].set_ylabel("RV sin offsets (m/s)")
    ejes[0].set_title(f"{modelo}: componentes del ajuste global")
    ejes[0].legend(ncol=3, fontsize=7, loc="upper left")
    ejes[1].scatter(tiempo, resultado["residuos"], s=7, alpha=0.55)
    ejes[1].axhline(0, color="black", ls="--", lw=0.8)
    ejes[1].set_xlabel("BJD - 2450000 (días)")
    ejes[1].set_ylabel("Residuos condicionales (m/s)")
    fig.tight_layout()
    fig.savefig(destino / "componentes.png", dpi=200, bbox_inches="tight")
    fig.savefig(destino / "componentes.pdf", bbox_inches="tight")
    plt.close(fig)


def _escribir_csv_dict(ruta, columnas, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(columnas)
        escritor.writerows(filas)


class LogLikeGP:
    """Objeto invocable serializable para los checkpoints de dynesty.

    Una función definida dentro de otra función puede no serializarse en
    Windows/Spyder. Esta clase guarda el contexto y usa funciones de módulo.
    """
    def __init__(self, modelo, priors, contexto):
        self.modelo = modelo
        self.priors = priors
        self.contexto = contexto

    def __call__(self, theta):
        return _evaluar_loglike_gp(theta, self.modelo, self.priors, self.contexto)


class PriorGP:
    """Transformación del hipercubo, serializable con el muestreador."""
    def __init__(self, priors):
        self.priors = priors

    def __call__(self, cubo):
        return _transformar_cubo_gp(cubo, self.priors)


def ajustar_gp_bayes_modelo(datos, nombre, salida, nlive=300, dlogz=0.1,
                            pmin=200.0, pmax=6000.0, semilla=42,
                            ajustes_priors=None):
    """Nested sampling para un único modelo; checkpoint y metadatos del prior.

    Evita comparar evidencias de configuraciones de priors no idénticas.
    Las muestras se resamplean con igual peso SOLO para tablas y gráficas.
    """
    import json
    import hashlib
    try:
        import celerite2  # noqa: F401
        import dynesty
        from dynesty.utils import resample_equal
    except ImportError as exc:
        raise ImportError("Instala primero 'dynesty' y 'celerite2': "
                          "pip install dynesty celerite2") from exc
    from scipy.special import logsumexp

    _prueba_funciones_kepler()
    contexto = _datos_para_gp(datos)
    priors = _especificacion_prior_gp(
        datos["nombres"], nombre, pmin, pmax,
        ajustes_priors=ajustes_priors)
    ndim = len(priors)
    if nlive < 2 * ndim:
        raise ValueError(f"Se necesitan como mínimo {2*ndim} live points")
    salida = Path(salida) / nombre
    salida.mkdir(parents=True, exist_ok=True)
    # Guardar el prior exacto y la lista de instrumentos: esencial para lnZ.
    h = hashlib.sha256()
    for campo in ("t", "y", "e", "indice"):
        h.update(np.ascontiguousarray(datos[campo]).tobytes())
    config = {"modelo": nombre, "nlive": int(nlive), "dlogz": float(dlogz),
              "sha256_datos": h.hexdigest(),
              # JSON convierte tuplas a listas. Al normalizar ambos lados,
              # una ejecucion interrumpida puede reanudarse con seguridad.
              "semilla": int(semilla), "priors": [list(p) for p in priors],
              "instrumentos": list(map(str, datos["nombres"])),
              "n_observaciones": datos["n"],
              "marginalizacion": "offsets y pendiente: normales propias",
              "kernel": "RotationTerm + Matern32Term"}
    archivo_cfg = salida / "configuracion.json"
    if archivo_cfg.exists():
        antigua = json.loads(archivo_cfg.read_text(encoding="utf-8"))
        if antigua != config:
            raise ValueError("Ya hay resultados con otros priors/configuración en "
                             f"{salida}. No reutilices su checkpoint: utiliza otra carpeta.")
    else:
        archivo_cfg.write_text(json.dumps(config, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    # Si el modelo ya terminó y su configuración coincide, lo reutilizamos.
    evidencia_terminada = salida / "evidencia.csv"
    if evidencia_terminada.exists():
        with evidencia_terminada.open(newline="", encoding="utf-8") as f:
            fila = next(csv.DictReader(f))
        print(f"{nombre}: evidencia existente y priors compatibles. Se reutiliza.")
        return {"modelo": nombre, "lnZ": float(fila["lnZ"]),
                "lnZerr": float(fila["lnZerr"])}

    logl = LogLikeGP(nombre, priors, contexto)
    prior_transform = PriorGP(priors)

    checkpoint = salida / "checkpoint.dynesty"
    if checkpoint.exists():
        print(f"Reanudando checkpoint de {nombre}: {checkpoint}", flush=True)
        sampler = dynesty.NestedSampler.restore(str(checkpoint))
        reanudar = True
    else:
        sampler = dynesty.NestedSampler(
            logl, prior_transform, ndim=ndim, nlive=nlive,
            sample="rwalk", bound="multi", rstate=np.random.default_rng(semilla))
        reanudar = False

    print(f"Ajuste {nombre}: n={datos['n']}, {ndim} dimensiones, nlive={nlive}", flush=True)
    sampler.run_nested(dlogz=dlogz, checkpoint_file=str(checkpoint),
                       checkpoint_every=120, resume=reanudar, print_progress=True)
    res = sampler.results
    logz = float(res.logz[-1]); errz = float(res.logzerr[-1])
    pesos = np.exp(res.logwt - logsumexp(res.logwt))
    muestras = resample_equal(res.samples, pesos,
                              rstate=np.random.default_rng(semilla+1))
    np.savez_compressed(salida / "resultado_pesado.npz", theta=res.samples,
                        logwt=res.logwt, logl=res.logl, logz=res.logz)
    nombres = [p[0] for p in priors]
    np.savetxt(salida / "muestras_posteriores.csv", muestras, delimiter=",",
               header=",".join(nombres), comments="")
    percentiles = np.percentile(muestras, [16, 50, 84], axis=0)
    _escribir_csv_dict(salida / "parametros.csv", ["parametro", "p16", "p50", "p84"],
                       [[n, *percentiles[:, i]] for i, n in enumerate(nombres)])
    _escribir_csv_dict(salida / "evidencia.csv", ["modelo", "lnZ", "lnZerr"],
                       [[nombre, logz, errz]])
    _representar_posterior_gp(muestras, priors, nombre, salida)
    theta_medio = np.median(muestras, axis=0)
    detalle = _evaluar_loglike_gp(theta_medio, nombre, priors, contexto,
                                componentes=True)
    if detalle is not None:
        _representar_componentes_gp(contexto, nombre, detalle, salida)
        _escribir_csv_dict(salida / "offsets_y_tendencia.csv",
                           ["parametro", "valor_estimado_condicional"],
                           [["offset_"+str(n), float(x)] for n, x in
                            zip(datos["nombres"], detalle["niveles_ajustados"])]
                           + [["pendiente_m_s_anio", float(detalle["coeficientes"][-1])]])
    print(f"{nombre}: lnZ={logz:.3f} ± {errz:.3f}", flush=True)
    return {"modelo": nombre, "lnZ": logz, "lnZerr": errz}



def probar_gp_reales():
    """Prueba corta de la API real de celerite2, sin ejecutar Nested Sampling.

    NO es un ajuste ni una evidencia; sirve para detectar fallos de entorno,
    rutas y evaluación matemática antes de iniciar un muestreo costoso.
    """
    try:
        import celerite2  # noqa: F401
        import dynesty  # noqa: F401
    except ImportError as exc:
        raise ImportError("Para la prueba instala: pip install dynesty celerite2") from exc
    t, rv, err, inst = leer_csv(
        RAIZ_PROYECTO / "data" / "Epsilon_Eridani_RVdata.csv",
        "RV (m/s)", "err RV (m/s)")
    datos = preparar_datos(t, rv, err, inst)
    contexto = _datos_para_gp(datos)
    print(f"Prueba de GP: {datos['n']} medidas de RV, {datos['ninst']} instrumentos")
    for modelo in ("M0", "M1", "M2"):
        priors = _especificacion_prior_gp(datos["nombres"], modelo, 200., 6000.)
        theta = _transformar_cubo_gp(np.full(len(priors), 0.5), priors)
        ll = _evaluar_loglike_gp(theta, modelo, priors, contexto)
        if not np.isfinite(ll):
            raise RuntimeError(f"La prueba de {modelo} dio lnL no finita: {ll}")
        print(f"  {modelo}: lnL de prueba = {ll:.3f} [NO evidencia]")
    print("Comprobación de las tres likelihoods: correcta.")
    return True


def main_gp_bayes(modelo="TODOS", nlive=300, dlogz=0.1,
                  periodo_min=200.0, periodo_max=6000.0):
    """Ejecutar M0/M1/M2 en las RV reales y comparar evidencias al terminar.

    TODOS ejecuta secuencialmente para poder reanudar tras una interrupción.
    La estrella se trata como UNA serie con siete etiquetas instrumentales.
    El ajuste de S_MW es una validación EXTERNA: no es una likelihood conjunta.
    """
    import json
    _prueba_funciones_kepler()
    modelo = modelo.upper()
    if modelo not in ("TODOS", "M0", "M1", "M2"):
        raise ValueError("Usa modelo='TODOS', 'M0', 'M1' o 'M2'")
    try:
        t, rv, err, inst = leer_csv(
            RAIZ_PROYECTO / "data" / "Epsilon_Eridani_RVdata.csv",
            "RV (m/s)", "err RV (m/s)")
    except FileNotFoundError as exc:
        raise FileNotFoundError("Falta data/Epsilon_Eridani_RVdata.csv") from exc
    datos = preparar_datos(t, rv, err, inst)
    destino = DIRECTORIO_RESULTADOS / "bayes_GP_kepler"
    destino.mkdir(parents=True, exist_ok=True)
    print("\nBAYES GP: datos RV globales de Epsilon Eridani", flush=True)
    print(f"n={datos['n']} | instrumentos={list(datos['nombres'])}", flush=True)
    print(f"Resultados: {destino.resolve()}", flush=True)
    print("El modelo RV no está ajustando simultáneamente la serie S_MW.", flush=True)
    lista = ["M0", "M1", "M2"] if modelo == "TODOS" else [modelo]
    resultado = []
    for nombre in lista:
        resultado.append(ajustar_gp_bayes_modelo(
            datos, nombre, destino, nlive=nlive, dlogz=dlogz,
            pmin=periodo_min, pmax=periodo_max))
    # También permite comparar si se ejecutaron M0, M1 y M2 por separado.
    resultados_todos = []
    for nombre in ("M0", "M1", "M2"):
        evidencia = destino / nombre / "evidencia.csv"
        cfg = destino / nombre / "configuracion.json"
        if evidencia.exists() and cfg.exists():
            with evidencia.open(newline="", encoding="utf-8") as f:
                lista_csv = list(csv.DictReader(f))
            c = json.loads(cfg.read_text(encoding="utf-8"))
            resultados_todos.append({"modelo": nombre,
                 "lnZ": float(lista_csv[0]["lnZ"]),
                 "lnZerr": float(lista_csv[0]["lnZerr"]),
                 "prior": c})
    # Si se hicieron ejecuciones con priors distintos, NO comparar.
    comparar = len(resultados_todos) == 3
    if comparar:
        ref = resultados_todos[0]["prior"]
        for z in resultados_todos[1:]:
            if (z["prior"]["sha256_datos"] != ref["sha256_datos"] or
                z["prior"]["n_observaciones"] != ref["n_observaciones"] or
                z["prior"]["instrumentos"] != ref["instrumentos"] or
                z["prior"]["nlive"] != ref["nlive"] or
                z["prior"]["dlogz"] != ref["dlogz"] or
                z["prior"]["semilla"] != ref["semilla"] or
                z["prior"]["kernel"] != ref["kernel"] or
                z["prior"]["marginalizacion"] != ref["marginalizacion"] or
                z["prior"]["priors"][:datos["ninst"]+7]
                    != ref["priors"][:datos["ninst"]+7]):
                comparar = False
        if comparar:
            priors_m1 = resultados_todos[1]["prior"]["priors"]
            priors_m2 = resultados_todos[2]["prior"]["priors"]
            if priors_m1[datos["ninst"]+7:datos["ninst"]+10] != priors_m2[datos["ninst"]+7:datos["ninst"]+10]:
                comparar = False
        if comparar:
            _escribir_csv_dict(destino / "comparacion_evidencias.csv",
                               ["modelo", "lnZ", "lnZerr", "delta_lnZ_vs_M0",
                                "sigma_delta_aprox"],
                               [[z["modelo"], z["lnZ"], z["lnZerr"],
                                 z["lnZ"]-resultados_todos[0]["lnZ"],
                                 np.hypot(z["lnZerr"], resultados_todos[0]["lnZerr"])]
                                for z in resultados_todos])
            print("\nComparación de lnZ guardada: comparacion_evidencias.csv", flush=True)
        else:
            print("Configuraciones de ejecución incompatibles: no comparo evidencias.")
    else:
        print("Aún faltan M0/M1/M2 completos para comparar evidencias.", flush=True)
    return resultado


# -----------------------------------------------------------------------------
# 12. PRUEBAS BAYESIANAS DE ROBUSTEZ Y SENSIBILIDAD (M0 FRENTE A M1)
# -----------------------------------------------------------------------------
# Estas pruebas NO reemplazan las ejecuciones principales M0/M1/M2.
# Sus resultados viven en otra carpeta para proteger los archivos antiguos.
# Cada escenario se ajusta GLOBALMENTE a todas las RV disponibles: el caso
# SIN_APF usa seis instrumentos a la vez, nunca seis ajustes por separado.

ESCENARIOS_ROBUSTEZ_GP = {
    "BASE_REPETICION": {"excluir": None, "ajustes_priors": {}, "semilla_diferente": False},
    "SIN_APF": {"excluir": "APF_1", "ajustes_priors": {}, "semilla_diferente": False},
    "Q0_AMPLIADO": {"excluir": None, "ajustes_priors": {"gp_Q0": (0.03, 20.0)},
                    "semilla_diferente": False},
    "ESCALA_AMPLIADA": {"excluir": None, "ajustes_priors": {"gp_escala_lenta_d": (30.0, 5000.0)},
                       "semilla_diferente": False},
    "SIN_APF_GP_AMPLIADO": {"excluir": "APF_1",
                           "ajustes_priors": {"gp_Q0": (0.03, 20.0),
                                              "gp_escala_lenta_d": (30.0, 5000.0)},
                           "semilla_diferente": False},
    "BASE_SEMILLA_2": {"excluir": None, "ajustes_priors": {},
                       "semilla_diferente": True},
}


def _preparar_escenario_robustez(escenario):
    """Cargar SOLO RV originales, opcionalmente excluyendo APF.

    El orden y todas las incertidumbres se preservan. Se mantiene la misma
    matriz de offsets y tendencia usada en los ajustes originales.
    """
    clave = str(escenario).upper()
    if clave not in ESCENARIOS_ROBUSTEZ_GP:
        raise ValueError("Escenario desconocido: " + clave + ". Disponibles: "
                         + ", ".join(ESCENARIOS_ROBUSTEZ_GP))
    configuracion = ESCENARIOS_ROBUSTEZ_GP[clave]
    t, y, sy, inst = leer_csv(
        RAIZ_PROYECTO / "data" / "Epsilon_Eridani_RVdata.csv",
        "RV (m/s)", "err RV (m/s)")
    if configuracion["excluir"] is not None:
        conservar = np.asarray(inst) != configuracion["excluir"]
        if np.all(conservar):
            raise ValueError("El instrumento a excluir no figura en el CSV")
        t, y, sy, inst = (np.asarray(a)[conservar] for a in (t, y, sy, inst))
    datos = preparar_datos(t, y, sy, inst)
    return clave, datos, configuracion


def _leer_una_fila_csv(ruta):
    """Lectura segura de una tabla de resultados ya finalizada."""
    with Path(ruta).open(newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    if not filas:
        raise ValueError(f"Archivo de resultados vacio: {ruta}")
    return filas[0]


def _resumir_escenario_robustez(carpeta, escenario):
    """Comparar evidencias SOLO entre M0 y M1 del MISMO escenario.

    No se comparan lnZ de distintos datasets (por ejemplo, SIN_APF frente a
    BASE_REPETICION); al retirar datos cambia la probabilidad de los datos.
    """
    import json
    carpeta = Path(carpeta)
    if not all((carpeta / m / "evidencia.csv").exists() for m in ("M0", "M1")):
        return None
    configs = {m: json.loads((carpeta / m / "configuracion.json").read_text(
        encoding="utf-8")) for m in ("M0", "M1")}
    a, b = configs["M0"], configs["M1"]
    # M0 y M1 deben tener las mismas RV y los mismos priors sobre
    # hiperparametros comunes. La parte orbital solo existe en M1.
    comprobaciones = (
        a["sha256_datos"] == b["sha256_datos"],
        a["instrumentos"] == b["instrumentos"],
        a["n_observaciones"] == b["n_observaciones"],
        a["nlive"] == b["nlive"],
        a["dlogz"] == b["dlogz"],
        a["semilla"] == b["semilla"],
        a["kernel"] == b["kernel"],
        a["marginalizacion"] == b["marginalizacion"],
        a["priors"] == b["priors"][:len(a["priors"])],
    )
    if not all(comprobaciones):
        raise RuntimeError(f"M0 y M1 no son comparables en {carpeta}")
    ev0 = _leer_una_fila_csv(carpeta / "M0" / "evidencia.csv")
    ev1 = _leer_una_fila_csv(carpeta / "M1" / "evidencia.csv")
    z0, z1 = float(ev0["lnZ"]), float(ev1["lnZ"])
    dz0, dz1 = float(ev0["lnZerr"]), float(ev1["lnZerr"])
    param = {}
    with (carpeta / "M1" / "parametros.csv").open(newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            param[fila["parametro"]] = fila
    if not {"planeta_periodo_d", "planeta_K_m_s"}.issubset(param):
        raise ValueError("Faltan parametros planetarios en M1")
    def valor(nombre, sufijo):
        return float(param[nombre][sufijo])
    fila = {"escenario": escenario,
            "N_RV": a["n_observaciones"], "instrumentos": ";".join(a["instrumentos"]),
            "lnZ_M0": z0, "lnZ_M0_error": dz0,
            "lnZ_M1": z1, "lnZ_M1_error": dz1,
            "delta_lnZ_M1_menos_M0": z1-z0,
            "error_delta_aprox": float(np.hypot(dz1, dz0)),
            "P16_d": valor("planeta_periodo_d", "p16"),
            "P50_d": valor("planeta_periodo_d", "p50"),
            "P84_d": valor("planeta_periodo_d", "p84"),
            "K16_m_s": valor("planeta_K_m_s", "p16"),
            "K50_m_s": valor("planeta_K_m_s", "p50"),
            "K84_m_s": valor("planeta_K_m_s", "p84"),
            "nlive": a["nlive"], "dlogz": a["dlogz"],
            "semilla": a["semilla"],
            "gp_Q0_prior": str(next(p[2:] for p in a["priors"] if p[0] == "gp_Q0")),
            "gp_escala_lenta_prior": str(next(p[2:] for p in a["priors"]
                                               if p[0] == "gp_escala_lenta_d")),
            }
    _escribir_csv_dict(carpeta / "comparacion_M1_vs_M0.csv", list(fila), [list(fila.values())])
    return fila


def resumen_robustez_bayes():
    """Actualizar la tabla global sin confundir evidencias entre conjuntos."""
    base = DIRECTORIO_RESULTADOS / "robustez_bayes"
    base.mkdir(parents=True, exist_ok=True)
    filas = []
    for escenario in ESCENARIOS_ROBUSTEZ_GP:
        carpeta = base / escenario
        if not carpeta.exists():
            continue
        fila = _resumir_escenario_robustez(carpeta, escenario)
        if fila is not None:
            filas.append(fila)
    if not filas:
        print("No hay pares M0 y M1 terminados todavía. No se crea tabla vacia.")
        return []
    _escribir_csv_dict(base / "resumen_robustez.csv", list(filas[0]),
                       [list(f.values()) for f in filas])
    # Figura descriptiva (no concluyente) para comparar la solidez de la
    # preferencia por M1 y la mediana del periodo estimado.
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(filas))
    ax[0].errorbar(x, [f["delta_lnZ_M1_menos_M0"] for f in filas],
                   yerr=[f["error_delta_aprox"] for f in filas],
                   fmt="o", capsize=3)
    ax[0].axhline(0, ls="--", color="black")
    ax[0].set_ylabel(r"$\Delta\ln Z=\ln Z_{M1}-\ln Z_{M0}$")
    ax[0].set_title("Comparaciones M1 vs M0 dentro de cada escenario")
    for k, f in enumerate(filas):
        ax[1].errorbar(k, f["P50_d"],
                       yerr=[[f["P50_d"]-f["P16_d"]],
                             [f["P84_d"]-f["P50_d"]]],
                       fmt="o", capsize=3)
    ax[1].set_ylabel("Periodo orbital (dias)")
    ax[1].set_xticks(x, [f["escenario"] for f in filas], rotation=35,
                       ha="right")
    fig.tight_layout()
    fig.savefig(base / "resumen_robustez.png", dpi=200)
    fig.savefig(base / "resumen_robustez.pdf")
    plt.close(fig)
    print("Tabla actualizada:", base / "resumen_robustez.csv")
    return filas


def main_robustez_bayes(escenario="SIN_APF", modo="PRUEBA", nlive=400,
                        dlogz=0.05, semilla=31415, semilla_alternativa=27182,
                        periodo_min=200.0, periodo_max=6000.0):
    """Ejecutar o probar M0 y M1 manteniendo los archivos originales intactos.

    modo: PRUEBA evalua la likelihood en parámetros internos; M0, M1, AMBOS
    ejecutan Nested Sampling; RESUMEN solo lee los resultados existentes.
    """
    modo = str(modo).upper()
    if modo == "RESUMEN":
        return resumen_robustez_bayes()
    if modo not in ("PRUEBA", "M0", "M1", "AMBOS"):
        raise ValueError("Modo invalido: PRUEBA, M0, M1, AMBOS o RESUMEN")
    escenario, datos, config = _preparar_escenario_robustez(escenario)
    semilla_elegida = (semilla_alternativa if config["semilla_diferente"] else semilla)
    carpeta = DIRECTORIO_RESULTADOS / "robustez_bayes" / escenario
    print("Escenario:", escenario, "| modo:", modo)
    print("RV incluidas:", datos["n"], "| instrumentos:", list(datos["nombres"]))
    print("Priors modificados:", config["ajustes_priors"] or "NINGUNO")
    print("Destino exclusivo:", carpeta)
    if modo == "PRUEBA":
        # Prueba real de la verosimilitud para ambos modelos.
        # No produce evidencia y no modifica los resultados previos.
        contexto = _datos_para_gp(datos)
        for m in ("M0", "M1"):
            priors = _especificacion_prior_gp(
                datos["nombres"], m, periodo_min, periodo_max,
                config["ajustes_priors"])
            theta = _transformar_cubo_gp(np.full(len(priors), 0.5), priors)
            ll = _evaluar_loglike_gp(theta, m, priors, contexto)
            if not np.isfinite(ll):
                raise RuntimeError(f"lnL no finita al probar {m}: {ll}")
            print(f"  {m}: lnL de prueba={ll:.3f} (NO es evidencia)")
        print("Las dos likelihoods funcionan. El modo PRUEBA no ha hecho ajustes.")
        return True
    carpeta.mkdir(parents=True, exist_ok=True)
    resultados = []
    for m in (("M0", "M1") if modo == "AMBOS" else (modo,)):
        resultado = ajustar_gp_bayes_modelo(
            datos, m, carpeta, nlive=nlive, dlogz=dlogz,
            pmin=periodo_min, pmax=periodo_max,
            semilla=semilla_elegida,
            ajustes_priors=config["ajustes_priors"])
        resultados.append(resultado)
    resumen_robustez_bayes()
    return resultados


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Modelos globales de Epsilon Eridani")
    parser.add_argument("--probar-gp", action="store_true",
                        help="Verificar las likelihoods con celerite2 sin ejecutar dynesty")
    parser.add_argument("--bayes-gp", action="store_true",
                        help="Comparar M0 (actividad GP), M1 (órbita circular) y M2 (kepleriana)")
    parser.add_argument("--modelo-gp", default="TODOS", choices=("TODOS","M0","M1","M2"),
                        help="Modelo para el modo --bayes-gp")
    parser.add_argument("--periodo-min", type=float, default=200.0,
                        help="Límite inferior del periodo planetario, días")
    parser.add_argument("--periodo-max", type=float, default=6000.0,
                        help="Límite superior del periodo planetario, días")
    parser.add_argument("--diagnosticos", action="store_true",
                        help="Ventanas espectrales, fases, residuos y robustez por instrumento")
    parser.add_argument("--bayes", action="store_true",
                        help="Realizar Nested Sampling de RV en vez de ajustes exploratorios")
    parser.add_argument("--periodo-rot", type=float, default=None,
                        help="Periodo inicial de actividad en dias; obligatorio con --bayes")
    parser.add_argument("--periodo-largo", type=float, default=None,
                        help="Candidato largo en dias para comparar con --bayes")
    parser.add_argument("--nlive", type=int, default=160,
                        help="Live points de dynesty (por defecto: 160)")
    parser.add_argument("--dlogz", type=float, default=0.15,
                        help="Criterio de parada Nested Sampling (por defecto: 0.15)")
    args = parser.parse_args()
    if sum([args.bayes, args.bayes_gp, args.diagnosticos, args.probar_gp]) > 1:
        parser.error("Elige solo uno: --bayes, --bayes-gp, --probar-gp o --diagnosticos")
    if args.probar_gp:
        probar_gp_reales()
    elif args.bayes_gp:
        main_gp_bayes(modelo=args.modelo_gp, nlive=args.nlive,
                      dlogz=args.dlogz, periodo_min=args.periodo_min,
                      periodo_max=args.periodo_max)
    elif args.diagnosticos:
        main_diagnosticos()
    elif args.bayes:
        if args.periodo_rot is None or args.periodo_rot <= 0:
            parser.error("Con --bayes proporciona --periodo-rot obtenido de actividad")
        if args.periodo_largo is not None and args.periodo_largo <= 0:
            parser.error("--periodo-largo debe ser positivo")
        main_bayes(args.periodo_rot, args.periodo_largo, args.nlive, args.dlogz)
    else:
        main()
