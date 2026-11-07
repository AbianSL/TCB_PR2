import numpy as np
import src.manage_data as manage_data

def describe_data(rjd, y, sy, instrumento, observable):

    print("\nSerie:", observable)
    print("Número de medidas:", len(y))
    print("Instrumentos:", np.unique(instrumento))

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        valores = y[mascara]
        errores = sy[mascara]

        print("\nInstrumento:", nombre)
        print("Número de medidas:", len(valores))
        print("Duración (días):", tiempo.max() - tiempo.min())
        print("Media:", np.mean(valores))
        print("Dispersión:", np.std(valores))
        print("Error medio:", np.mean(errores))

def inspect_outliers(rjd, y, instrumento, observable):

    print("\nRevisión de puntos atípicos:", observable)

    for nombre in np.unique(instrumento):

        mascara = instrumento == nombre

        tiempo = rjd[mascara]
        valores = y[mascara]

        keep = manage_data.sigma_clip(valores)

        print("\nInstrumento:", nombre)
        print("Puntos marcados:", np.sum(~keep))

        for t, valor in zip(tiempo[~keep], valores[~keep]):
            print("BJD - 2450000:", t, "Valor:", valor)