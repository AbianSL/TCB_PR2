"""Registro de modelos disponibles: nombre en la línea de comandos -> clase."""

from src.modelos.actividad import ModeloActividad
from src.modelos.planeta_circular import ModeloPlanetaCircular
from src.modelos.planeta_kepleriano import ModeloPlanetaKepleriano

MODELOS = {
    "M0": ModeloActividad,
    "M1": ModeloPlanetaCircular,
    "M2": ModeloPlanetaKepleriano,
}
