"""Interfaz común (estrategia) de todos los análisis."""

from abc import ABC, abstractmethod


class Analisis(ABC):
    """Recibe las series ya leídas y escribe sus resultados en `salida`."""

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def ejecutar(self, datos, salida):
        """`datos` = {"RV": Serie, "S_MW": Serie}; `salida` = carpeta (Path)."""
