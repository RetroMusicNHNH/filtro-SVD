"""
Utilidades del proyecto Filtro SVD

Este paquete contiene utilidades y configuraciones generales
que son utilizadas por múltiples módulos del proyecto.

Módulos disponibles:
- config: Configuraciones globales del proyecto
- math_utils: Utilidades matemáticas auxiliares
"""

__version__ = '1.0.0'
__author__ = 'RetroMusicNHNH'

# Importaciones principales
from .config import Config
from .math_utils import MathUtils

__all__ = [
    'Config',
    'MathUtils'
]