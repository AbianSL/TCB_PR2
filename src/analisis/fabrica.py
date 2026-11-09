"""Estrategias de análisis. `crear` instancia la elegida sin importar las demás."""

from importlib import import_module

# nombre en la línea de comandos -> (módulo, clase)
ANALISIS = {
    "exploratorio": ("exploratorio", "Exploratorio"),
    "acf": ("acf", "ACF"),
    "lomb": ("lomb_scargle", "PeriodogramaLS"),
    "bayes": ("bayes", "Bayes"),
}


def crear(nombre, args):
    modulo, clase = ANALISIS[nombre]
    return getattr(import_module(f"src.analisis.{modulo}"), clase)(args)
