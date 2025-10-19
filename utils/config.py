"""
Configuraciones globales del proyecto Filtro SVD

Este módulo contiene todas las configuraciones y parámetros
utilizados a lo largo del proyecto de filtrado de señales.

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np
from typing import Tuple


class Config:
    """
    Clase de configuración global del proyecto
    
    Contiene todos los parámetros necesarios para la ejecución
    del filtro SVD según las especificaciones del examen.
    """
    
    # ========================================================================
    # PARÁMETROS DE SEÑAL (Tarea 1)
    # ========================================================================
    
    # Número de puntos para la evaluación de la función
    N_PUNTOS: int = 2000
    
    # Intervalo de evaluación [-3π, 3π]
    INTERVALO: Tuple[float, float] = (-3 * np.pi, 3 * np.pi)
    
    # Rango del parámetro de ruido ξ ∈ [0, 1/10]
    XI_MIN: float = 0.0
    XI_MAX: float = 1.0 / 10.0
    
    # ========================================================================
    # PARÁMETROS DE MATRIZ DE HANKEL (Tarea 2)
    # ========================================================================
    
    # Tamaño de ventana L para la matriz de Hankel
    L: int = 800
    
    # K se calcula automáticamente como K = N - L + 1
    @property
    def K(self) -> int:
        """Calcula K basado en N_PUNTOS y L"""
        return self.N_PUNTOS - self.L + 1
    
    # ========================================================================
    # PARÁMETROS DE SVD Y TRUNCAMIENTO (Tarea 3)
    # ========================================================================
    
    # Criterios para determinación de rango de truncamiento
    # Factor de tolerancia para el criterio de truncamiento
    TOLERANCE_FACTOR: float = 1.0  # Se divide por L en la implementación
    
    # Mínimo número de valores singulares a mantener
    MIN_RANK: int = 1
    
    # Máximo número de valores singulares a considerar (como fracción de L)
    MAX_RANK_FRACTION: float = 0.8
    
    # ========================================================================
    # PARÁMETROS DE VISUALIZACIÓN (Tarea 6)
    # ========================================================================
    
    # Tamaño de figura por defecto
    FIGSIZE: Tuple[int, int] = (12, 8)
    
    # Resolución de las gráficas
    DPI: int = 300
    
    # Estilo de las gráficas
    PLOT_STYLE: str = 'default'
    
    # Colores para las gráficas
    COLOR_ORIGINAL: str = 'blue'
    COLOR_FILTRADA: str = 'red'
    COLOR_DIFERENCIA: str = 'green'
    COLOR_SIGMA: str = 'orange'
    
    # Transparencia por defecto
    ALPHA: float = 0.8
    
    # Ancho de línea por defecto
    LINEWIDTH: float = 1.0
    
    # ========================================================================
    # CONFIGURACIONES DE REPRODUCIBILIDAD
    # ========================================================================
    
    # Semilla aleatoria (None para comportamiento aleatorio)
    SEED: int = None  # Cambiar por un número para reproducibilidad
    
    # Semilla específica para desarrollo/testing
    DEVELOPMENT_SEED: int = 42
    
    # ========================================================================
    # CONFIGURACIONES DE QR (Tarea 5 - Implementación propia)
    # ========================================================================
    
    # Tolerancia para verificación de ortogonalidad en QR
    QR_TOLERANCE: float = 1e-12
    
    # Método QR preferido ('householder' o 'givens')
    QR_METHOD: str = 'householder'
    
    # Tamaño máximo de matriz para demostración QR
    QR_MAX_SIZE: int = 200
    
    # ========================================================================
    # CONFIGURACIONES DE PERFORMANCE
    # ========================================================================
    
    # Usar cálculos vectorizados cuando sea posible
    USE_VECTORIZATION: bool = True
    
    # Mostrar barras de progreso para operaciones largas
    SHOW_PROGRESS: bool = False  # Cambiar a True si se implementa tqdm
    
    # Nivel de verbosidad (0: silencioso, 1: básico, 2: detallado)
    VERBOSITY: int = 1
    
    # ========================================================================
    # VALIDACIONES DE CONFIGURACIÓN
    # ========================================================================
    
    def __post_init__(self):
        """Valida la configuración después de la inicialización"""
        self.validar_configuracion()
    
    def validar_configuracion(self):
        """
        Valida que todos los parámetros de configuración sean válidos
        
        Raises:
            ValueError: Si algún parámetro es inválido
        """
        # Validar parámetros de señal
        if self.N_PUNTOS <= 0:
            raise ValueError(f"N_PUNTOS debe ser positivo, got {self.N_PUNTOS}")
        
        if self.INTERVALO[1] <= self.INTERVALO[0]:
            raise ValueError(f"Intervalo inválido: {self.INTERVALO}")
        
        if not (0 <= self.XI_MIN < self.XI_MAX <= 1):
            raise ValueError(f"Rango de ξ inválido: [{self.XI_MIN}, {self.XI_MAX}]")
        
        # Validar parámetros de Hankel
        if self.L <= 0 or self.L >= self.N_PUNTOS:
            raise ValueError(f"L debe estar entre 1 y {self.N_PUNTOS-1}, got {self.L}")
        
        if self.K <= 0:
            raise ValueError(f"K calculado es inválido: {self.K}")
        
        # Validar parámetros de visualización
        if len(self.FIGSIZE) != 2 or any(x <= 0 for x in self.FIGSIZE):
            raise ValueError(f"FIGSIZE inválido: {self.FIGSIZE}")
        
        if self.DPI <= 0:
            raise ValueError(f"DPI debe ser positivo, got {self.DPI}")
        
        # Validar verbosidad
        if not isinstance(self.VERBOSITY, int) or self.VERBOSITY < 0:
            raise ValueError(f"VERBOSITY debe ser entero no negativo, got {self.VERBOSITY}")
    
    def usar_seed_desarrollo(self):
        """Configura la semilla para desarrollo/testing reproducible"""
        self.SEED = self.DEVELOPMENT_SEED
        np.random.seed(self.DEVELOPMENT_SEED)
    
    def obtener_resumen(self) -> str:
        """
        Genera un resumen legible de la configuración actual
        
        Returns:
            str: Resumen formateado de la configuración
        """
        return f"""
CONFIGURACIÓN DEL PROYECTO FILTRO SVD
=====================================
Parámetros de Señal:
  • Puntos de evaluación: {self.N_PUNTOS}
  • Intervalo: [{self.INTERVALO[0]:.4f}, {self.INTERVALO[1]:.4f}]
  • Rango de ruido ξ: [{self.XI_MIN}, {self.XI_MAX}]

Parámetros de Hankel:
  • Tamaño de ventana L: {self.L}
  • Columnas K: {self.K}
  • Dimensión matriz: {self.L}×{self.K}

Configuración SVD:
  • Factor de tolerancia: {self.TOLERANCE_FACTOR}
  • Rango mínimo: {self.MIN_RANK}
  • Fracción máxima de rango: {self.MAX_RANK_FRACTION}

Visualización:
  • Tamaño de figura: {self.FIGSIZE}
  • DPI: {self.DPI}
  • Estilo: {self.PLOT_STYLE}

Reproducibilidad:
  • Semilla: {self.SEED or 'Aleatoria'}
  • Verbosidad: {self.VERBOSITY}
        """


# Instancia global de configuración por defecto
config = Config()


# Funciones de utilidad para configuración
def crear_config_personalizada(**kwargs) -> Config:
    """
    Crea una configuración personalizada
    
    Args:
        **kwargs: Parámetros a modificar de la configuración por defecto
        
    Returns:
        Config: Nueva instancia de configuración
    """
    nueva_config = Config()
    
    # Aplicar modificaciones personalizadas
    for key, value in kwargs.items():
        if hasattr(nueva_config, key):
            setattr(nueva_config, key, value)
        else:
            raise ValueError(f"Parámetro de configuración desconocido: {key}")
    
    # Validar la nueva configuración
    nueva_config.validar_configuracion()
    
    return nueva_config


def obtener_config_examen() -> Config:
    """
    Retorna configuración específica para el examen
    
    Returns:
        Config: Configuración con parámetros exactos del examen
    """
    return crear_config_personalizada(
        N_PUNTOS=2000,
        INTERVALO=(-3 * np.pi, 3 * np.pi),
        L=800,
        VERBOSITY=2,
        SEED=None  # Aleatorio como especifica el examen
    )


if __name__ == "__main__":
    # Ejemplo de uso del módulo de configuración
    print("Probando módulo de configuración...")
    
    # Configuración por defecto
    config_default = Config()
    print(config_default.obtener_resumen())
    
    # Configuración personalizada
    config_custom = crear_config_personalizada(
        N_PUNTOS=1000,
        L=400,
        VERBOSITY=2
    )
    print("\nConfiguración personalizada:")
    print(f"N_PUNTOS: {config_custom.N_PUNTOS}")
    print(f"L: {config_custom.L}")
    print(f"K: {config_custom.K}")
    
    # Configuración del examen
    config_examen = obtener_config_examen()
    print("\nConfiguración del examen cargada exitosamente ✓")