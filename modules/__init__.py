"""
Módulos del proyecto Filtro SVD

Este paquete contiene todos los módulos especializados para el filtrado
de señales temporales usando descomposición SVD.

Módulos disponibles:
- signal_generator: Generación de señales ruidosas
- hankel_matrix: Construcción de matrices de Hankel
- svd_processor: Procesamiento SVD y determinación de rango
- reconstruction: Reconstrucción usando anti-diagonales
- qr_decomposition: Implementación propia de QR
- visualization: Visualización y gráficas
"""

__version__ = '1.0.0'
__author__ = 'RetroMusicNHNH'
__email__ = 'estudiante@ejemplo.com'

# Importaciones principales para facilitar el uso
from .signal_generator import SignalGenerator
from .hankel_matrix import HankelMatrix
from .svd_processor import SVDProcessor
from .reconstruction import SignalReconstructor
from .qr_decomposition import QRDecomposition
from .visualization import Visualizer

__all__ = [
    'SignalGenerator',
    'HankelMatrix', 
    'SVDProcessor',
    'SignalReconstructor',
    'QRDecomposition',
    'Visualizer'
]