"""Periodogramas Lomb-Scargle de RV y S_MW de epsilon Eridani.

Resultados independientes de los ajustes bayesianos de model_data.py.
Las FAP analiticas suponen ruido independiente; no prueban planetas.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle


FAP = (0.10, 0.01, 0.001)


def calcular_ls(t, y, dy, frecuencias=None):
    """Calcula el GLS ponderado, sus umbrales FAP y los cinco picos principales."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy = np.asarray(dy, dtype=float)
    validos = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    t, y, dy = t[validos], y[validos], dy[validos]

    if len(t) < 10 or np.ptp(t) <= 2:
        raise ValueError("La serie necesita >= 10 medidas validas y > 2 dias de cobertura")

    duracion = np.ptp(t)
    # Se exploran hasta 6000 dias sin superar la cobertura propia de la serie.
    pmax = min(6000.0, duracion)
    if frecuencias is None:
        # Diez muestras por resolucion de Fourier (1/duracion).
        nfreq = int(np.ceil(10 * duracion * (1 / 2 - 1 / pmax))) + 1
        frecuencias = np.linspace(1 / pmax, 1 / 2, nfreq)

    # dy son los errores originales: no se confunde la senal con el ruido.
    ls = LombScargle(t - t.min(), y, dy, fit_mean=True, center_data=True)
    potencia = ls.power(frecuencias)
    niveles = [ls.false_alarm_level(
        fap, method="baluev",
        minimum_frequency=frecuencias[0],
        maximum_frequency=frecuencias[-1]
    ) for fap in FAP]

    # Maximos locales, evitando repetir el mismo pico por sobremuestreo.
    candidatos = np.where(
        (potencia[1:-1] > potencia[:-2]) &
        (potencia[1:-1] >= potencia[2:])
    )[0] + 1
    candidatos = np.unique(np.r_[candidatos, np.argmax(potencia)])
    candidatos = candidatos[np.argsort(potencia[candidatos])[::-1]]
    mejores = []
    for i in candidatos:
        if all(abs(frecuencias[i] - frecuencias[j]) >= 1 / duracion
               for j in mejores):
            mejores.append(int(i))
        if len(mejores) == 5:
            break

    return {
        "f": frecuencias, "potencia": potencia, "niveles": niveles,
        "picos": [(1 / frecuencias[i], potencia[i]) for i in mejores],
        "n": len(t), "periodo_limite": pmax,
        "pico_en_borde": bool(np.argmax(potencia) in (0, len(potencia) - 1))
    }


def guardar_figura(fig, ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(ruta.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dibujar_periodograma(resultado, titulo, ruta):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(1 / resultado["f"], resultado["potencia"], color="steelblue")
    for nivel, fap, color in zip(resultado["niveles"], FAP,
                                  ("gray", "darkgreen", "purple")):
        ax.axhline(nivel, ls="--", lw=1, color=color, label=f"FAP {100*fap:g} %")
    if resultado["picos"]:
        p = resultado["picos"][0][0]
        ax.axvline(p, color="darkorange", ls=":", label=f"Pico: {p:.2f} d")
    ax.set(xscale="log", xlim=(2, resultado["periodo_limite"]),
           xlabel="Periodo (dias)", ylabel="Potencia Lomb-Scargle", title=titulo)
    ax.legend(fontsize=8)
    fig.tight_layout()
    guardar_figura(fig, ruta)


def dibujar_comparacion(original, limpio, titulo, ruta):
    fig, ax = plt.subplots(figsize=(8, 4))
    for res, etiqueta, color in ((original, "Original", "steelblue"),
                                (limpio, "Filtrada a 5 sigma", "darkorange")):
        ax.plot(1 / res["f"], res["potencia"], label=etiqueta, color=color)
        ax.axhline(res["niveles"][1], ls="--", lw=1, color=color,
                   label=f"FAP 1 % ({etiqueta})")
    ax.set(xscale="log", xlim=(2, original["periodo_limite"]),
           xlabel="Periodo (dias)", ylabel="Potencia Lomb-Scargle", title=titulo)
    ax.legend(fontsize=8)
    fig.tight_layout()
    guardar_figura(fig, ruta)


def ventana_espectral(t, frecuencias):
    """Ventana de muestreo: |media(exp(2*pi*i*f*t))|^2."""
    t = np.asarray(t) - np.min(t)
    potencia = np.empty(len(frecuencias))
    for inicio in range(0, len(frecuencias), 200):
        bloque = frecuencias[inicio:inicio + 200]
        fases = 2 * np.pi * bloque[:, None] * t[None, :]
        potencia[inicio:inicio + 200] = (
            np.mean(np.cos(fases), axis=1)**2 +
            np.mean(np.sin(fases), axis=1)**2
        )
    return potencia


def dibujar_ventana(t, titulo, ruta):
    f_lenta = np.linspace(1 / 2000, 0.02, 1600)
    f_rapida = np.linspace(0.02, 1.05, 4000)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
    ejes[0].plot(f_lenta, ventana_espectral(t, f_lenta), color="black")
    ejes[0].axvline(1 / 365.25, color="darkorange", ls="--", label="1 ano")
    ejes[0].set(xlabel="Frecuencia (1/d)", ylabel="Potencia ventana",
                title="Bajas frecuencias")
    ejes[0].legend(fontsize=8)
    ejes[1].plot(f_rapida, ventana_espectral(t, f_rapida), color="black")
    ejes[1].axvline(1, color="darkorange", ls="--", label="1 dia")
    ejes[1].set(xlabel="Frecuencia (1/d)", title="Muestreo diario")
    ejes[1].legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    guardar_figura(fig, ruta)


def analizar_lomb_scargle(rjd_rv, rv, srv, inst_rv,
                            rjd_s, s, ss, inst_s,
                            rjd_s_limpio, s_limpio, ss_limpio, inst_s_limpio):
    """Analiza cada instrumento, compara RV/actividad y exporta tablas y figuras."""
    salida = Path("results/lomb_scargle")
    salida.mkdir(parents=True, exist_ok=True)
    resumen, picos = [], []
    resultados = {}

    series = [
        ("RV", "Original", rjd_rv, rv, srv, inst_rv),
        ("S_MW", "Original", rjd_s, s, ss, inst_s),
        ("S_MW", "Filtrada", rjd_s_limpio, s_limpio, ss_limpio, inst_s_limpio),
    ]

    for observable, seleccion, t, y, dy, instrumentos in series:
        for nombre in np.unique(instrumentos):
            mascara = instrumentos == nombre
            res = calcular_ls(t[mascara], y[mascara], dy[mascara])
            resultados[(observable, seleccion, nombre)] = res
            periodo, potencia = res["picos"][0]
            significativo = potencia > res["niveles"][1]
            resumen.append([observable, seleccion, nombre, res["n"],
                            periodo, potencia, *res["niveles"],
                            significativo, res["pico_en_borde"]])
            for rango, (p, a) in enumerate(res["picos"], start=1):
                picos.append([observable, seleccion, nombre, rango, p, a,
                              a > res["niveles"][1]])
            print(f"{observable:4} {nombre:23} {seleccion:9} "
                  f"P = {periodo:9.3f} d | supera FAP 1 %: {significativo}")

            if observable == "RV":
                dibujar_periodograma(res, f"RV: {nombre}",
                    salida / "periodogramas_RV" / nombre)
            elif seleccion == "Filtrada":
                original = resultados[("S_MW", "Original", nombre)]
                dibujar_comparacion(original, res, f"Actividad: {nombre}",
                    salida / "periodogramas_actividad" / nombre)

    # La ventana depende solo de los instantes, no de los valores de RV/S_MW.
    for observable, _, t, _, _, instrumentos in series[:2]:
        for nombre in np.unique(instrumentos):
            dibujar_ventana(t[instrumentos == nombre], f"Ventana: {observable} - {nombre}",
                            salida / "ventanas" / observable / nombre)

    # Comparacion de periodicidades en los instrumentos que miden ambas series.
    comunes = np.intersect1d(np.unique(inst_rv), np.unique(inst_s_limpio))
    for nombre in comunes:
        a = resultados[("RV", "Original", nombre)]
        b = resultados[("S_MW", "Filtrada", nombre)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(1 / a["f"], a["potencia"], label="RV", color="steelblue")
        ax.plot(1 / b["f"], b["potencia"], label="Actividad filtrada", color="darkorange")
        ax.axvspan(11.35, 11.55, color="gray", alpha=0.2, label="Rotacion (literatura)")
        ax.set(xscale="log", xlim=(2, min(a["periodo_limite"], b["periodo_limite"])),
               xlabel="Periodo (dias)", ylabel="Potencia Lomb-Scargle",
               title=f"RV frente a actividad: {nombre}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        guardar_figura(fig, salida / "comparacion_RV_actividad" / nombre)

    with (salida / "resumen_ls.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["serie", "seleccion", "instrumento", "N",
                          "periodo_maximo_d", "potencia_maxima", "FAP_10",
                          "FAP_1", "FAP_0_1", "supera_FAP_1", "maximo_en_borde"])
        escritor.writerows(resumen)
    with (salida / "picos_ls.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["serie", "seleccion", "instrumento", "rango",
                          "periodo_d", "potencia", "supera_umbral_FAP_1"])
        escritor.writerows(picos)
    print(f"\nPeriodogramas, ventanas y tablas guardados en {salida}/")
