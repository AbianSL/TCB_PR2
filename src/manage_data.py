import numpy as np
from astropy.table import Table


def opendata(datapath, observable):

    datos = Table.read(datapath, format="csv")
    datos.sort("BJD")

    rjd = np.array(datos["BJD"]) - 2450000
    instrumento = np.array(datos["Inst"])

    if observable == "RV":
        y = np.array(datos["RV (m/s)"])
        sy = np.array(datos["err RV (m/s)"])

    elif observable == "SMW":
        y = np.array(datos["S_MW x1000"]) / 1000
        sy = np.array(datos["sig S_MW x1000"]) / 1000

    return rjd, y, sy, instrumento

def sigma_clip(y, nsigma=5):

    keep = np.ones(len(y), dtype=bool)

    while True:

        mediana = np.median(y[keep])
        sigma = np.std(y[keep], ddof=1)

        bad = np.abs(y[keep] - mediana) > nsigma * sigma

        if not bad.any():
            return keep

        keep[np.flatnonzero(keep)[bad]] = False

def clean_data(rjd, y, sy, instrumento, nsigma=5):

    keep = np.ones(len(y), dtype=bool)

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre
        keep[mascara] = sigma_clip(y[mascara], nsigma)

    print("\nMedidas originales:", len(y))
    print("Medidas excluidas:", np.sum(~keep))
    print("Medidas conservadas:", np.sum(keep))

    return rjd[keep], y[keep], sy[keep], instrumento[keep]