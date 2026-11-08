"""Analisis exploratorio global de RV y S_MW, independiente de dynesty.

Los instrumentos aportan distintos offsets, pero una tendencia y senales
periodicas comunes. Los periodogramas miden reduccion de chi2 y NO dan FAP.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.linalg import qr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def leer_serie(ruta, actividad=False, excluir=None):
    """Lee CSV y estandariza tiempo, observable, incertidumbre e instrumento."""
    with Path(ruta).open(newline="", encoding="utf-8-sig") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise ValueError(f"CSV vacio: {ruta}")
    factor = 1e-3 if actividad else 1.0
    campo = "S_MW x1000" if actividad else "RV (m/s)"
    error_campo = "sig S_MW x1000" if actividad else "err RV (m/s)"
    t = np.array([float(f["BJD"]) - 2450000.0 for f in filas])
    y = factor * np.array([float(f[campo]) for f in filas])
    err = factor * np.array([float(f[error_campo]) for f in filas])
    inst = np.array([f["Inst"] for f in filas])
    if not (np.isfinite(t).all() and np.isfinite(y).all()
            and np.isfinite(err).all() and (err > 0).all()):
        raise ValueError("Valores no finitos o incertidumbres invalidas")
    orden = np.argsort(t)
    t, y, err, inst = t[orden], y[orden], err[orden], inst[orden]
    if excluir is not None:
        seleccion = inst != excluir
        t, y, err, inst = t[seleccion], y[seleccion], err[seleccion], inst[seleccion]
    nombres, indice = np.unique(inst, return_inverse=True)
    # Piso empirico de ruido descriptivo: las incertidumbres formales del
    # indicador S pueden ser mucho menores que su variabilidad estelar.
    # NO se utiliza este piso en el posterior GP de RV.
    pisos = []
    for j in range(len(nombres)):
        valores = y[indice == j]
        mediana = np.median(valores)
        mad = 1.4826 * np.median(np.abs(valores - mediana))
        pisos.append(max(mad, np.std(valores) * 0.05, 1e-10))
    sigma = np.sqrt(err**2 + np.asarray(pisos)[indice] ** 2)
    return dict(t=t, y=y, err=err, sigma=sigma, nombres=nombres,
                indice=indice, t0=float(np.median(t)), pisos=np.array(pisos))


def matriz_diseno(datos, periodos=()):
    """Una columna offset por instrumento, una tendencia comun y senos."""
    t = datos["t"] - datos["t0"]
    I = datos["indice"]
    n = len(datos["nombres"])
    columnas = [(I == k).astype(float) for k in range(n)]
    columnas.append(t / 365.25)  # pendiente en unidades por anio
    for periodo in periodos:
        fase = 2 * np.pi * t / periodo
        columnas += [np.sin(fase), np.cos(fase)]
    return np.column_stack(columnas)


def ajustar_lineal(datos, periodos=()):
    """Minimos cuadrados ponderados de toda la serie, no por instrumento."""
    X = matriz_diseno(datos, periodos)
    w = 1.0 / datos["sigma"]
    beta = np.linalg.lstsq(X * w[:, None], datos["y"] * w, rcond=None)[0]
    modelo = X @ beta
    residuos = datos["y"] - modelo
    chi2 = float(np.sum((residuos * w) ** 2))
    return dict(beta=beta, modelo=modelo, residuos=residuos,
                chi2=chi2, rms=float(np.sqrt(np.mean(residuos**2))))


def periodograma(datos, minimo, maximo, previos=()):
    """Potencia como mejora fraccional de chi2 al agregar seno y coseno.

    QR proyecta fuera offsets, tendencia y cualquier periodicidad previa.
    Calculamos varios periodos de prueba por lote para ahorrar tiempo.
    """
    t = datos["t"] - datos["t0"]
    sigma = datos["sigma"]
    X0 = matriz_diseno(datos, previos) / sigma[:, None]
    Q, _ = qr(X0, mode="economic")
    yw = datos["y"] / sigma
    r = yw - Q @ (Q.T @ yw)
    chi0 = float(r @ r)
    duracion = float(np.ptp(datos["t"]))
    fmin, fmax = 1.0 / maximo, 1.0 / minimo
    nfreq = int(np.clip(np.ceil(5 * duracion * (fmax - fmin)), 800, 8500))
    frecuencias = np.linspace(fmin, fmax, nfreq)
    potencias = np.empty(nfreq)
    for desde in range(0, nfreq, 160):
        hasta = min(desde + 160, nfreq)
        fase = 2 * np.pi * t[:, None] * frecuencias[None, desde:hasta]
        S, C = np.sin(fase) / sigma[:, None], np.cos(fase) / sigma[:, None]
        S -= Q @ (Q.T @ S)
        C -= Q @ (Q.T @ C)
        ss, cc, sc = np.sum(S*S, axis=0), np.sum(C*C, axis=0), np.sum(S*C, axis=0)
        ys, yc = r @ S, r @ C
        det = ss * cc - sc**2
        mejora = np.divide(cc * ys**2 + ss * yc**2 - 2 * sc * ys * yc,
                           det, out=np.zeros_like(det), where=det > 1e-14)
        potencias[desde:hasta] = np.clip(mejora / max(chi0, 1e-15), 0, 1)
    i = int(np.argmax(potencias))
    return 1 / frecuencias, potencias, float(1 / frecuencias[i])


def ventana_anual(datos):
    """Potencia de la ventana temporal: revela periodicidad del muestreo."""
    t = datos["t"] - datos["t0"]
    f = np.linspace(1 / 1600, 1 / 100, 700)
    ventana = np.empty_like(f)
    for i, freq in enumerate(f):
        fase = 2 * np.pi * freq * t
        ventana[i] = np.mean(np.cos(fase))**2 + np.mean(np.sin(fase))**2
    return 1 / f, ventana


def guardar_tabla(ruta, encabezado, filas):
    with ruta.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


def analizar(ruta_datos, salida, sin_apf=False):
    """Analiza ambas series completas y guarda resultados reproducibles.

    El GP bayesiano, calculado aparte, usa incertidumbres formales + jitter
    ajustable. Los pesos descriptivos de este modulo no modifican sus priors.
    """
    raiz = Path(ruta_datos)
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    rv = leer_serie(raiz / "Epsilon_Eridani_RVdata.csv",
                    excluir="APF_1" if sin_apf else None)
    actividad = leer_serie(raiz / "Epsilon_Eridani_SMW.csv", actividad=True)
    series = {"RV": rv, "S_MW": actividad}
    estadisticas = []
    for nombre, datos in series.items():
        for j, inst in enumerate(datos["nombres"]):
            m = datos["indice"] == j
            estadisticas.append([nombre, inst, int(m.sum()),
                float(np.min(datos["t"][m])), float(np.max(datos["t"][m])),
                float(np.median(datos["y"][m])), float(np.std(datos["y"][m], ddof=1)),
                float(np.median(datos["err"][m]))])
    guardar_tabla(salida / "estadisticas_instrumentos.csv",
        ["serie", "instrumento", "N", "RJD_min", "RJD_max",
         "mediana", "std_muestral", "mediana_error_formal"], estadisticas)
    periodos, modelos, espectros, picos_libres = {}, {}, {}, {}
    # Ajustar primero la escala larga y despues explorar la corta tras retirarla.
    for nombre, datos in series.items():
        pl, powl, P_largo = periodograma(datos, 200, 6000)
        pc, powc, P_corto_libre = periodograma(datos, 5, 25, (P_largo,))
        picos_libres[nombre] = P_corto_libre
        # En actividad guardamos tambien el maximo global (a menudo ~8 d).
        # Para el ajuste ilustrativo seleccionamos el mejor pico en 10-15 d:
        # ventana motivada por los candidatos cercanos a 12 d de RV.
        # No debe presentarse como una deteccion independiente de rotacion.
        if nombre == "S_MW":
            mask = (pc >= 10) & (pc <= 15)
            P_corto = float(pc[mask][np.argmax(powc[mask])])
        else:
            P_corto = P_corto_libre
        periodos[nombre] = (P_largo, P_corto)
        modelos[nombre] = ajustar_lineal(datos, (P_largo, P_corto))
        espectros[nombre] = (pl, powl, pc, powc)
    guardar_tabla(salida / "periodos_exploratorios.csv",
        ["serie", "periodo_largo_d", "periodo_corto_ajustado_d",
         "maximo_corto_libre_d", "RMS_residual"],
        [[nombre, *periodos[nombre], picos_libres[nombre],
          modelos[nombre]["rms"]] for nombre in series])

    fig, ejes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for eje, (nombre, datos) in zip(ejes, series.items()):
        for j, inst in enumerate(datos["nombres"]):
            m = datos["indice"] == j
            eje.scatter(datos["t"][m], datos["y"][m] - np.median(datos["y"][m]),
                        s=7, alpha=.5, label=inst)
        eje.set_ylabel("RV - mediana (m/s)" if nombre == "RV" else "S_MW - mediana")
        eje.legend(fontsize=7, ncol=4)
    ejes[1].set_xlabel("BJD - 2450000 (dias)")
    fig.suptitle("Series completas; centrado instrumental solo para visualizar")
    fig.tight_layout()
    fig.savefig(salida / "01_series.png", dpi=170)
    plt.close(fig)

    fig, ejes = plt.subplots(2, 2, figsize=(11, 7))
    for fila, nombre in enumerate(series):
        pl, powl, pc, powc = espectros[nombre]
        for col, (periodos_grid, potencia, titulo) in enumerate(
                ((pl, powl, "200-6000 d"), (pc, powc, "5-25 d; tras quitar periodo largo"))):
            eje = ejes[fila, col]
            eje.plot(periodos_grid, potencia, lw=1)
            eje.axvline(periodos[nombre][col], color="tab:red", ls="--", lw=1,
                        label="Periodo utilizado")
            if col == 1 and abs(picos_libres[nombre] - periodos[nombre][1]) > 0.05:
                eje.axvline(picos_libres[nombre], color="gray", ls=":", lw=1,
                            label="Maximo libre")
                eje.legend(fontsize=8)
            eje.set(xlabel="Periodo (dias)", ylabel="Reduccion fraccional de chi2",
                    title=f"{nombre}: {titulo}")
            if col == 0:
                eje.set_xscale("log")
    fig.tight_layout()
    fig.savefig(salida / "02_periodogramas.png", dpi=170)
    plt.close(fig)

    fig, ejes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for eje, (nombre, datos) in zip(ejes, series.items()):
        p, w = ventana_anual(datos)
        eje.plot(p, w, lw=1)
        eje.axvline(365.25, color="tab:red", ls="--", label="1 anio")
        eje.set(xlabel="Periodo de la ventana (dias)", ylabel="Potencia", title=nombre)
        eje.legend()
    fig.tight_layout()
    fig.savefig(salida / "03_ventana_anual.png", dpi=170)
    plt.close(fig)

    # Ajuste de S_MW con periodos compartidos y offsets instrumentales distintos.
    d = actividad
    ajuste = modelos["S_MW"]
    ninst = len(d["nombres"])
    s_centrada = d["y"] - ajuste["beta"][d["indice"]]
    modelo_fisico = ajuste["modelo"] - ajuste["beta"][d["indice"]]
    fig, ejes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, inst in enumerate(d["nombres"]):
        m = d["indice"] == j
        ejes[0].scatter(d["t"][m], s_centrada[m], s=8, alpha=.45, label=inst)
    ejes[0].plot(d["t"], modelo_fisico, '.', ms=2, color="tab:red", label="Modelo comun")
    ejes[0].set_ylabel("S_MW - offset")
    ejes[0].legend(fontsize=7)
    ejes[1].scatter(d["t"], ajuste["residuos"], s=8, alpha=.5)
    ejes[1].axhline(0, c="black", ls="--", lw=.8)
    ejes[1].set(xlabel="BJD - 2450000 (dias)", ylabel="Residuos S_MW")
    fig.suptitle("Actividad: ajustes sinusoidales globales (no GP)")
    fig.tight_layout()
    fig.savefig(salida / "04_actividad_ajuste.png", dpi=170)
    plt.close(fig)
    print("Analisis global RV y S_MW terminado.")
    for nombre in series:
        largo, corto = periodos[nombre]
        print(f"  {nombre}: P_largo={largo:.2f} d, P_corto_ajustado={corto:.4f} d"
              f", maximo_corto_libre={picos_libres[nombre]:.4f} d (exploratorios)")
    print(f"Figuras y tablas: {salida}")
    return periodos
