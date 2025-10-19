"""
Generación de Señales Ruidosas para Filtro SVD

Este módulo implementa la Tarea 1 del examen:
Generación de señales temporales ruidosas usando la función
f(t) = sin(t) + ξ·sin(50t) con parámetro aleatorio ξ ∈ [0, 1/10]

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np
import random
from typing import Tuple, Optional
import warnings


class SignalGenerator:
    """
    Clase para generar señales temporales ruidosas según especificaciones del examen
    
    Esta clase implementa la generación de señales usando la función:
    f(t) = sin(t) + ξ·sin(50t)
    
    donde ξ es un parámetro aleatorio en el intervalo [0, 1/10]
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Inicializa el generador de señales
        
        Args:
            seed: Semilla para reproducibilidad (opcional)
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generar_senal_ruidosa(self, 
                            n_puntos: int = 2000,
                            intervalo: Tuple[float, float] = (-3*np.pi, 3*np.pi),
                            xi_min: float = 0.0,
                            xi_max: float = 1.0/10.0,
                            seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Genera señal ruidosa f(t) = sin(t) + ξ·sin(50t) con ξ ∈ [0, 1/10]
        
        Args:
            n_puntos: Número de puntos de evaluación (default: 2000)
            intervalo: Tupla (inicio, fin) del intervalo (default: (-3π, 3π))
            xi_min: Valor mínimo del parámetro de ruido (default: 0.0)
            xi_max: Valor máximo del parámetro de ruido (default: 1/10)
            seed: Semilla específica para esta generación (opcional)
            
        Returns:
            tuple: (t, y, xi) donde:
                - t: array con los puntos de tiempo
                - y: array con los valores de la señal ruidosa
                - xi: valor del parámetro de ruido utilizado
                
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Validar parámetros de entrada
        self._validar_parametros(n_puntos, intervalo, xi_min, xi_max)
        
        # Configurar semilla si se especifica
        if seed is not None:
            random.seed(seed)
        
        # Generar partición uniforme del intervalo usando numpy.linspace
        t = np.linspace(intervalo[0], intervalo[1], n_puntos)
        
        # Generar parámetro aleatorio ξ en el rango especificado
        xi = random.uniform(xi_min, xi_max)
        
        # Evaluar la función f(t) = sin(t) + ξ·sin(50t)
        y = np.sin(t) + xi * np.sin(50 * t)
        
        # Reportar información de la señal generada
        self._reportar_informacion_senal(n_puntos, intervalo, xi, y)
        
        return t, y, xi
    
    def generar_senal_limpia(self,
                           n_puntos: int = 2000,
                           intervalo: Tuple[float, float] = (-3*np.pi, 3*np.pi)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera la señal limpia sin ruido: f(t) = sin(t)
        
        Útil para comparaciones y análisis de la efectividad del filtro.
        
        Args:
            n_puntos: Número de puntos de evaluación
            intervalo: Tupla (inicio, fin) del intervalo
            
        Returns:
            tuple: (t, y_limpia) donde:
                - t: array con los puntos de tiempo
                - y_limpia: array con los valores de sin(t)
        """
        # Validar parámetros básicos
        if n_puntos <= 0:
            raise ValueError(f"n_puntos debe ser positivo, got {n_puntos}")
        if intervalo[1] <= intervalo[0]:
            raise ValueError(f"Intervalo inválido: {intervalo}")
        
        # Generar partición y evaluar función limpia
        t = np.linspace(intervalo[0], intervalo[1], n_puntos)
        y_limpia = np.sin(t)
        
        return t, y_limpia
    
    def calcular_componente_ruido(self,
                                xi: float,
                                t: np.ndarray) -> np.ndarray:
        """
        Calcula solo la componente de ruido ξ·sin(50t)
        
        Args:
            xi: Parámetro de ruido
            t: Array de puntos de tiempo
            
        Returns:
            np.ndarray: Componente de ruido pura
        """
        return xi * np.sin(50 * t)
    
    def analizar_frecuencias_senal(self, 
                                 y: np.ndarray, 
                                 dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza análisis de frecuencias de la señal usando FFT
        
        Args:
            y: Señal a analizar
            dt: Paso de tiempo entre muestras
            
        Returns:
            tuple: (frecuencias, magnitudes) del espectro
        """
        # Calcular FFT
        Y = np.fft.fft(y)
        frecuencias = np.fft.fftfreq(len(y), dt)
        
        # Tomar solo frecuencias positivas
        idx_pos = frecuencias >= 0
        frecuencias = frecuencias[idx_pos]
        magnitudes = np.abs(Y[idx_pos])
        
        return frecuencias, magnitudes
    
    def _validar_parametros(self, 
                          n_puntos: int,
                          intervalo: Tuple[float, float],
                          xi_min: float,
                          xi_max: float) -> None:
        """
        Valida que todos los parámetros de entrada sean válidos
        
        Args:
            n_puntos: Número de puntos
            intervalo: Intervalo de evaluación
            xi_min: Valor mínimo de xi
            xi_max: Valor máximo de xi
            
        Raises:
            ValueError: Si algún parámetro es inválido
        """
        if n_puntos <= 0:
            raise ValueError(f"n_puntos debe ser positivo, got {n_puntos}")
        
        if len(intervalo) != 2:
            raise ValueError(f"intervalo debe ser una tupla de 2 elementos, got {len(intervalo)}")
        
        if intervalo[1] <= intervalo[0]:
            raise ValueError(f"intervalo[1] debe ser mayor que intervalo[0], got {intervalo}")
        
        if xi_min < 0:
            raise ValueError(f"xi_min debe ser no negativo, got {xi_min}")
        
        if xi_max <= xi_min:
            raise ValueError(f"xi_max debe ser mayor que xi_min, got xi_min={xi_min}, xi_max={xi_max}")
        
        if xi_max > 1:
            warnings.warn(f"xi_max={xi_max} es mayor que 1, lo cual puede generar ruido muy alto")
        
        # Validaciones específicas del examen
        if abs(xi_max - 0.1) > 1e-10:  # Verificar que sea 1/10
            warnings.warn(f"El examen especifica xi_max = 1/10 = 0.1, pero got {xi_max}")
    
    def _reportar_informacion_senal(self,
                                  n_puntos: int,
                                  intervalo: Tuple[float, float],
                                  xi: float,
                                  y: np.ndarray) -> None:
        """
        Reporta información detallada de la señal generada
        
        Args:
            n_puntos: Número de puntos generados
            intervalo: Intervalo de evaluación
            xi: Parámetro de ruido utilizado
            y: Señal generada
        """
        print(f"📊 Señal ruidosa generada exitosamente:")
        print(f"  • Número de puntos: {n_puntos}")
        print(f"  • Intervalo de evaluación: [{intervalo[0]:.4f}, {intervalo[1]:.4f}]")
        print(f"  • Parámetro de ruido ξ: {xi:.6f}")
        print(f"  • Rango de valores y: [{np.min(y):.4f}, {np.max(y):.4f}]")
        print(f"  • Media: {np.mean(y):.4f}")
        print(f"  • Desviación estándar: {np.std(y):.4f}")
        
        # Calcular algunas estadísticas útiles
        amplitud_sin = 1.0  # Amplitud de sin(t)
        amplitud_ruido = xi * 1.0  # Amplitud máxima de xi*sin(50t)
        snr_teorico = 20 * np.log10(amplitud_sin / amplitud_ruido) if xi > 0 else float('inf')
        
        print(f"  • SNR teórico: {snr_teorico:.2f} dB" if xi > 0 else "  • SNR teórico: ∞ dB")
    
    def obtener_estadisticas_senal(self, y: np.ndarray) -> dict:
        """
        Calcula estadísticas completas de una señal
        
        Args:
            y: Señal a analizar
            
        Returns:
            dict: Diccionario con estadísticas de la señal
        """
        return {
            'media': np.mean(y),
            'mediana': np.median(y),
            'std': np.std(y),
            'var': np.var(y),
            'min': np.min(y),
            'max': np.max(y),
            'rms': np.sqrt(np.mean(y**2)),
            'energia': np.sum(y**2),
            'potencia': np.mean(y**2)
        }
    
    def generar_multiples_realizaciones(self,
                                      n_realizaciones: int,
                                      **kwargs) -> list:
        """
        Genera múltiples realizaciones de señales ruidosas
        
        Útil para estudios Monte Carlo o promediado de señales.
        
        Args:
            n_realizaciones: Número de señales a generar
            **kwargs: Parámetros para generar_senal_ruidosa
            
        Returns:
            list: Lista de tuplas (t, y, xi) para cada realización
        """
        realizaciones = []
        
        for i in range(n_realizaciones):
            # Generar nueva semilla para cada realización si no se especifica
            if 'seed' not in kwargs:
                kwargs['seed'] = None
            
            t, y, xi = self.generar_senal_ruidosa(**kwargs)
            realizaciones.append((t, y, xi))
            
            print(f"Realización {i+1}/{n_realizaciones} - ξ = {xi:.6f}")
        
        return realizaciones


# Funciones de utilidad del módulo
def demo_generacion_senal():
    """
    Función de demostración del generador de señales
    """
    print("🎯 Demostración del Generador de Señales")
    print("="*50)
    
    # Crear generador
    generator = SignalGenerator(seed=42)  # Semilla para reproducibilidad
    
    # Generar señal con parámetros del examen
    t, y, xi = generator.generar_senal_ruidosa(
        n_puntos=2000,
        intervalo=(-3*np.pi, 3*np.pi)
    )
    
    # Mostrar estadísticas
    stats = generator.obtener_estadisticas_senal(y)
    print("\n📈 Estadísticas de la señal:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Generar señal limpia para comparación
    t_limpia, y_limpia = generator.generar_senal_limpia(
        n_puntos=2000,
        intervalo=(-3*np.pi, 3*np.pi)
    )
    
    print(f"\n✓ Señal limpia generada con {len(y_limpia)} puntos")
    print(f"✓ Diferencia RMS: {np.sqrt(np.mean((y - y_limpia)**2)):.4f}")


if __name__ == "__main__":
    # Ejecutar demostración
    demo_generacion_senal()