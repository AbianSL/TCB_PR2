"""DCF de RV y S_MW con test de permutacion (Epsilon Eridani).

Colocar junto a main.py y ejecutar con F5 en Spyder.
Reutiliza la lectura y el ajuste global de src/exploratorio.py.
Las bandas y los valores p contrastan ausencia de correlacion temporal
residual; NO son significancias frente a ruido estelar correlacionado.
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from src.exploratorio import leer_serie, ajustar_lineal, periodograma

# Configuracion
RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "results" / "acf"
ANCHO_BIN = 2.0             # Separaciones temporales agrupadas cada 2 dias
LAG_MAX = 100.0             # Retardos de 0 a 100 dias
MIN_PARES = 100             # Minimo de pares para representar una DCF
QUITAR_PERIODO_LARGO = True # Quita una sinusoide lenta antes de buscar rotacion
N_PERMUTACIONES = 1000
SEMILLA = 42


def preparar_pares(t):
    """Agrupa pares i<j por retardo; se reutilizan en las permutaciones."""
    t = np.asarray(t)
    fin = np.searchsorted(t, t + LAG_MAX, side="right")
    n_por_punto = np.maximum(fin - np.arange(len(t)) - 1, 0)
    i = np.repeat(np.arange(len(t), dtype=np.int32), n_por_punto)
    j = np.concatenate([np.arange(k + 1, fin[k], dtype=np.int32)
                        for k in np.flatnonzero(n_por_punto)])
    b = ((t[j] - t[i]) / ANCHO_BIN).astype(np.int32)
    n_bins = int(np.ceil(LAG_MAX / ANCHO_BIN))
    valido = (t[j] > t[i]) & (b < n_bins)
    i, j, b = i[valido], j[valido], b[valido]
    n_pares = np.bincount(b, minlength=n_bins)
    retardos = (np.arange(n_bins) + 0.5) * ANCHO_BIN
    return retardos, i, j, b, n_pares


def calcular_dcf(y, i, j, bin_par, n_pares):
    """Media de productos de desviaciones normalizadas en cada bin."""
    z = y - np.mean(y)
    varianza = np.mean(z ** 2)
    if varianza <= 0:
        raise ValueError("Serie sin variacion: no es posible calcular la DCF")
    sumas = np.bincount(bin_par, weights=z[i] * z[j], minlength=len(n_pares))
    return np.divide(sumas, n_pares * varianza,
                     out=np.full(len(n_pares), np.nan),
                     where=n_pares >= MIN_PARES)


def analizar(datos, rng):
    """Realiza una DCF observada y su test de permutacion con los mismos pares."""
    periodos = ()
    if QUITAR_PERIODO_LARGO:
        p_largo = periodograma(datos, 200, 6000)[-1]
        periodos = (p_largo,)
        print(f"  Periodo largo eliminado: {p_largo:.1f} dias")

    # Un unico ajuste para toda la serie; offsets por instrumento y tendencia.
    residuos = ajustar_lineal(datos, periodos)["residuos"]
    lag, i, j, b, n_pares = preparar_pares(datos["t"])
    observada = calcular_dcf(residuos, i, j, b, n_pares)

    # Bajo la hipotesis nula se conservan fechas e instrumentos; se mezclan
    # solamente los residuos de un mismo instrumento.
    grupos = [np.flatnonzero(datos["indice"] == k)
              for k in range(len(datos["nombres"]))]
    nulos = np.empty((N_PERMUTACIONES, len(lag)))
    for n in range(N_PERMUTACIONES):
        mezclados = residuos.copy()
        for grupo in grupos:
            mezclados[grupo] = residuos[rng.permutation(grupo)]
        nulos[n] = calcular_dcf(mezclados, i, j, b, n_pares)

    valido = n_pares >= MIN_PARES
    inferior = np.full(len(lag), np.nan)
    superior = np.full(len(lag), np.nan)
    inferior[valido], superior[valido] = np.percentile(
        nulos[:, valido], [2.5, 97.5], axis=0)

    # Prueba global: maximo positivo buscado entre 5 y 100 dias.
    region = valido & (lag >= 5) & (lag < LAG_MAX)
    max_observado = np.nanmax(observada[region])
    max_nulo = np.nanmax(nulos[:, region], axis=1)
    p_global = (1 + np.count_nonzero(max_nulo >= max_observado)) / (N_PERMUTACIONES + 1)
    pico = lag[region][np.nanargmax(observada[region])]
    return lag, observada, n_pares, inferior, superior, pico, p_global


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    series = (("RV", "Epsilon_Eridani_RVdata.csv", False),
              ("S_MW", "Epsilon_Eridani_SMW.csv", True))
    fig, ejes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    resumen = []

    with (SALIDA / "acf.csv").open("w", newline="", encoding="utf-8") as archivo:
        tabla = csv.writer(archivo)
        tabla.writerow(["serie", "lag_d", "dcf", "n_pares", "banda_2_5", "banda_97_5"])
        for eje, (nombre, fichero, es_actividad) in zip(ejes, series):
            print(f"{nombre}: calculando DCF y {N_PERMUTACIONES} permutaciones")
            datos = leer_serie(RAIZ / "data" / fichero, actividad=es_actividad)
            lag, dcf, pares, bajo, alto, pico, p = analizar(datos, rng)
            resumen.append((nombre, float(pico), float(p), N_PERMUTACIONES))

            # Se anade lag=0 solo como referencia grafica; no entra en el test.
            tabla.writerow([nombre, 0.0, 1.0, len(datos["t"]), "", ""])
            tabla.writerows((nombre, *fila) for fila in zip(lag, dcf, pares, bajo, alto))
            eje.fill_between(lag, bajo, alto, color="gray", alpha=0.25,
                             label="Permutaciones: banda puntual del 95 %")
            eje.plot(np.r_[0, lag], np.r_[1, dcf], ".-", ms=3, lw=1.2,
                     label="DCF observada")
            eje.axhline(0, color="gray", lw=0.8, ls="--")
            eje.set_ylabel(f"DCF {nombre}")
            eje.grid(alpha=0.2)
            eje.legend(fontsize=8)
            print(f"  Pico global: {pico:.1f} dias; p_global = {p:.4f}")

    with (SALIDA / "resumen_permutaciones.csv").open("w", newline="", encoding="utf-8") as archivo:
        tabla = csv.writer(archivo)
        tabla.writerow(["serie", "pico_global_5_100_d", "p_global", "n_permutaciones"])
        tabla.writerows(resumen)

    ejes[-1].set(xlabel="Retardo temporal (dias)", xlim=(0, LAG_MAX))
    fig.suptitle("Autocorrelacion irregular (DCF) y bandas de permutacion")
    fig.tight_layout()
    fig.savefig(SALIDA / "acf.png", dpi=180)
    plt.close(fig)
    print(f"Resultados: {SALIDA / 'acf.png'}, acf.csv y resumen_permutaciones.csv")
    print("Aviso: las permutaciones no conservan el ruido estelar correlacionado.")


if __name__ == "__main__":
    main()
