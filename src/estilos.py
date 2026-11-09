"""Estilos de figura aptos para daltonismo (paleta Okabe-Ito).

Cada instrumento conserva el mismo color y marcador en todas las figuras, de
modo que la distinción nunca depende únicamente del color.
"""

# nombre del instrumento -> (color, marcador)
ESTILOS_INSTRUMENTOS = {
    "APF_1": ("#0072B2", "o"),
    "COUDE-SPECTROGRAPH_1": ("#E69F00", "s"),
    "ESPRESSO18_1": ("#009E73", "^"),
    "HAMILTON_2": ("#CC79A7", "D"), 
    "HARPS03_1": ("#D55E00", "v"),
    "HARPS15_1": ("#56B4E9", "P"),
    "HIRES_1": ("#000000", "X"),
}
ESTILO_INSTRUMENTO_DESCONOCIDO = ("#666666", "o")

# nombre de la serie -> (color, marcador), para figuras que muestran una curva por serie
ESTILOS_SERIES = {
    "RV": ("#0072B2", "o"),
    "S_MW": ("#D55E00", "s"),
}

COLOR_ESPECTRO = "#0072B2"
COLOR_PERIODO_USADO = "#D55E00"
COLOR_PERIODO_LIBRE = "#666666"
COLOR_MODELO = "#000000"
COLOR_BANDA_NULA = "#999999"


def estilo_instrumento(nombre):
    """Devuelve (color, marcador) del instrumento."""
    return ESTILOS_INSTRUMENTOS.get(nombre, ESTILO_INSTRUMENTO_DESCONOCIDO)


def estilo_serie(nombre):
    """Devuelve (color, marcador) de una serie ('RV' o 'S_MW')."""
    return ESTILOS_SERIES.get(nombre, ESTILO_INSTRUMENTO_DESCONOCIDO)
