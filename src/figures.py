import numpy as np
import matplotlib.pyplot as plt
import os


def plot_activity(rjd_s, s, ss, instrumento_s):

    fig, ax = plt.subplots(figsize=(9, 4))

    for nombre in np.unique(instrumento_s):

        mascara = instrumento_s == nombre

        ax.errorbar(
            rjd_s[mascara],
            s[mascara],
            yerr=ss[mascara],
            fmt="o",
            markersize=3,
            label=nombre
        )

    ax.set_xlabel("BJD - 2450000 (días)")
    ax.set_ylabel("Índice de actividad S_MW")
    ax.set_title("Epsilon Eridani")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_rv(rjd_rv, rv, srv, instrumento_rv):

    fig, ax = plt.subplots(figsize=(10, 6))

    for nombre in np.unique(instrumento_rv):

        mascara = instrumento_rv == nombre

        velocidad = rv[mascara]
        velocidad_centrada = velocidad - np.median(velocidad)

        ax.errorbar(
            rjd_rv[mascara],
            velocidad_centrada,
            yerr=srv[mascara],
            fmt="o",
            markersize=3,
            label=nombre
        )

    ax.set_xlabel("BJD - 2450000 (días)")
    ax.set_ylabel("RV - mediana de cada instrumento (m/s)")
    ax.set_title("Epsilon Eridani")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8)

    plt.tight_layout()
    plt.savefig("results/RV_centrada.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/RV_centrada.pdf", bbox_inches="tight")
    plt.show()

def plot_outliers(tiempo, actividad, error, keep, nombre, tmin, tmax):
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.errorbar(
        tiempo[keep], actividad[keep],
        yerr=error[keep],
        fmt="o", markersize=4,
        color="steelblue", label="No marcados"
    )

    ax.errorbar(
        tiempo[~keep], actividad[~keep],
        yerr=error[~keep],
        fmt="o", markersize=5,
        color="red", label="Marcados por sigma-clipping"
    )

    ax.set_xlim(tmin, tmax)
    ax.set_xlabel("BJD - 2450000 (días)")
    ax.set_ylabel("Índice de actividad S_MW")
    ax.set_title(nombre)
    ax.legend()

    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    plt.tight_layout()
    plt.show() 

def plot_periodogram_comparison(
    frecuencias, potencia, potencia_clean,
    nombre, niveles, niveles_clean
):

    fig, ax = plt.subplots(figsize=(10, 6))

    periodos = 1 / frecuencias

    ax.plot(
        periodos, potencia,
        color="steelblue", label="Todos los datos"
    )

    ax.plot(
        periodos, potencia_clean,
        color="darkorange", linestyle="--",
        label="Filtrados a 5 sigma"
    )

    porcentajes = [10, 1, 0.1]
    colores = ["gray", "green", "purple"]

    for i in range(3):

        ax.axhline(
            niveles[i],
            color=colores[i], linestyle=":", linewidth=1,
            label=f"FAP {porcentajes[i]} % original"
        )

        ax.axhline(
            niveles_clean[i],
            color=colores[i], linestyle="--", linewidth=1,
            label=f"FAP {porcentajes[i]} % filtrada"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Periodo (días)")
    ax.set_ylabel("Potencia Lomb–Scargle")
    ax.set_title("Actividad: " + nombre)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=3,
        fontsize=8
    )

    plt.tight_layout()
    plt.show()

def plot_window(frecuencias, ventana, nombre):

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(frecuencias, ventana, color="black")

    ax.set_xlabel("Frecuencia (ciclos/día)")
    ax.set_ylabel("Potencia de la ventana")
    ax.set_title("Muestreo: " + nombre)
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show() 
    
def plot_rv_periodogram(frecuencias, potencia, niveles, nombre):

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        1 / frecuencias,
        potencia,
        color="steelblue",
        label="RV"
    )

    porcentajes = [10, 1, 0.1]
    colores = ["gray", "green", "purple"]

    for i in range(3):

        ax.axhline(
            niveles[i],
            color=colores[i],
            linestyle="--",
            linewidth=1,
            label=f"FAP {porcentajes[i]} %"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Periodo (días)")
    ax.set_ylabel("Potencia Lomb–Scargle")
    ax.set_title("Velocidad radial: " + nombre)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4
    )

    plt.tight_layout()

    os.makedirs("results/periodogramas_RV", exist_ok=True)

    fig.savefig(
        f"results/periodogramas_RV/{nombre}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)
def plot_rv_trends(tiempo, rv, error, constante, lineal, nombre):
    """Guardar los datos y los dos ajustes de tendencia."""

    orden = np.argsort(tiempo)

    # Restar el nivel constante solo para facilitar la representación
    referencia = constante[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        tiempo, rv - referencia, yerr=error,
        fmt=".", alpha=0.6, label="Datos"
    )

    ax.plot(
        tiempo[orden], constante[orden] - referencia,
        color="black", linestyle="--", label="Constante"
    )

    ax.plot(
        tiempo[orden], lineal[orden] - referencia,
        color="red", label="Recta"
    )

    ax.set_xlabel("BJD - 2450000 (días)")
    ax.set_ylabel("RV - nivel constante (m/s)")
    ax.set_title(nombre)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    fig.tight_layout()

    os.makedirs("results/ajustes_RV", exist_ok=True)
    fig.savefig(
        f"results/ajustes_RV/{nombre}.png",
        dpi=300, bbox_inches="tight"
    )

    plt.close(fig)  

def plot_rv_sine(tiempo, y, error, recta,
                 tiempo_curva, curva, residuos, nombre):
    """Guardar el ajuste sinusoidal y sus residuos."""

    orden = np.argsort(tiempo)

    fig, ejes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True
    )

    ejes[0].errorbar(
        tiempo, y, yerr=error,
        fmt=".", alpha=0.6, label="Datos"
    )
    ejes[0].plot(
        tiempo[orden], recta[orden],
        "k--", label="Recta"
    )
    ejes[0].plot(
        tiempo_curva, curva,
        color="red", linewidth=1, label="Recta + seno"
    )
    ejes[0].set_ylabel("RV centrada (m/s)")
    ejes[0].set_title(nombre)
    ejes[0].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3
    )

    ejes[1].errorbar(
        tiempo, residuos, yerr=error,
        fmt=".", alpha=0.6
    )
    ejes[1].axhline(0, color="black", linestyle="--")
    ejes[1].set_ylabel("Residuos (m/s)")
    ejes[1].set_xlabel("BJD - 2450000 (días)")

    fig.tight_layout()

    os.makedirs("results/ajustes_seno_RV", exist_ok=True)
    fig.savefig(
        f"results/ajustes_seno_RV/{nombre}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)