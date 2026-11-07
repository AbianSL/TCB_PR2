import numpy as np
import matplotlib.pyplot as plt


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