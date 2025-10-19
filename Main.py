"""
FILTRO DE RUIDO PARA SEÑALES TEMPORALES USANDO SVD
Examen Parcial - Álgebra Lineal Numérica (MO0014)
Profesor: Jorge Luis Salazar Chaves

Parte II: Problema programado (40 pts)
Controlador principal del proyecto modular

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

from modules.signal_generator import SignalGenerator
from modules.hankel_matrix import HankelMatrix
from modules.svd_processor import SVDProcessor
from modules.reconstruction import SignalReconstructor
from modules.qr_decomposition import QRDecomposition
from modules.visualization import Visualizer
from utils.config import Config

import numpy as np


class FiltroSVD:
    """
    Clase principal que orquesta todo el proceso de filtrado SVD
    
    Esta clase coordina todos los módulos del proyecto:
    1. Generación de señales ruidosas
    2. Construcción de matriz de Hankel
    3. Descomposición SVD y determinación de rango
    4. Reconstrucción usando anti-diagonales
    5. Visualización de resultados
    6. Implementación de QR propio
    """
    
    def __init__(self, config=None):
        """
        Inicializa el filtro SVD
        
        Args:
            config: Objeto de configuración personalizado (opcional)
        """
        self.config = config or Config()
        self.resultados = {}
        self.setup_modules()
    
    def setup_modules(self):
        """Inicializa todos los módulos del proyecto"""
        self.signal_gen = SignalGenerator()
        self.hankel = HankelMatrix(L=self.config.L)
        self.svd_proc = SVDProcessor()
        self.reconstructor = None  # Se inicializa después de conocer L y K
        self.qr_decomp = QRDecomposition()
        self.visualizer = Visualizer(figsize=self.config.FIGSIZE)
        
        print("Módulos inicializados correctamente")
    
    def ejecutar_tarea_1(self):
        """Ejecuta Tarea 1: Generación de señal ruidosa"""
        print("\n" + "="*70)
        print("EJECUTANDO TAREA 1: GENERACIÓN DE SEÑAL RUIDOSA")
        print("="*70)
        
        # Generar señal ruidosa
        t, y, xi = self.signal_gen.generar_senal_ruidosa(
            n_puntos=self.config.N_PUNTOS,
            intervalo=self.config.INTERVALO,
            seed=self.config.SEED
        )
        
        # Almacenar resultados
        self.resultados['t'] = t
        self.resultados['y_original'] = y
        self.resultados['xi'] = xi
        
        # Visualizar señal original
        self.visualizer.graficar_senal_original(t, y, xi)
        self.visualizer.graficar_zoom_regiones(t, y, None)
        
        print(f"Tarea 1 completada - ξ = {xi:.6f}")
        return t, y, xi
    
    def ejecutar_tarea_2(self):
        """Ejecuta Tarea 2: Construcción de matriz de Hankel"""
        print("\n" + "="*70)
        print("EJECUTANDO TAREA 2: CONSTRUCCIÓN DE MATRIZ DE HANKEL")
        print("="*70)
        
        if 'y_original' not in self.resultados:
            raise ValueError("Debe ejecutar Tarea 1 primero")
        
        # Construir matriz de Hankel
        X = self.hankel.construir_matriz(self.resultados['y_original'])
        
        # Almacenar resultados
        self.resultados['X'] = X
        self.resultados['L'] = self.hankel.L
        self.resultados['K'] = X.shape[1]
        
        print(f"Matriz de Hankel construida: {X.shape[0]}×{X.shape[1]}")
        return X
    
    def ejecutar_tarea_3(self):
        """Ejecuta Tarea 3: Descomposición SVD y determinación de rango"""
        print("\n" + "="*70)
        print("EJECUTANDO TAREA 3: DESCOMPOSICIÓN SVD")
        print("="*70)
        
        if 'X' not in self.resultados:
            raise ValueError("Debe ejecutar Tarea 2 primero")
        
        # Descomposición SVD
        U, sigma, Vt = self.svd_proc.descomponer_svd(self.resultados['X'])
        
        # Determinar rango óptimo
        r = self.svd_proc.determinar_rango(sigma, self.resultados['L'])
        
        # Construir matriz truncada
        Xr = self.svd_proc.construir_matriz_truncada(U, sigma, Vt, r)
        
        # Almacenar resultados
        self.resultados['U'] = U
        self.resultados['sigma'] = sigma
        self.resultados['Vt'] = Vt
        self.resultados['r'] = r
        self.resultados['Xr'] = Xr
        
        # Visualizar valores singulares
        self.visualizer.graficar_valores_singulares(sigma, r)
        
        print(f"SVD completada - Rango r = {r}")
        return U, sigma, Vt, r, Xr
    
    def ejecutar_tarea_4(self):
        """Ejecuta Tarea 4: Reconstrucción usando anti-diagonales"""
        print("\n" + "="*70)
        print("EJECUTANDO TAREA 4: RECONSTRUCCIÓN DE SEÑAL")
        print("="*70)
        
        if 'Xr' not in self.resultados:
            raise ValueError("Debe ejecutar Tarea 3 primero")
        
        # Inicializar reconstructor con dimensiones conocidas
        self.reconstructor = SignalReconstructor(
            self.resultados['L'], 
            self.resultados['K']
        )
        
        # Reconstruir señal filtrada
        y_filtrada = self.reconstructor.reconstruir_senal(self.resultados['Xr'])
        
        # Almacenar resultados
        self.resultados['y_filtrada'] = y_filtrada
        
        # Visualizar comparación
        self.visualizer.graficar_comparacion(
            self.resultados['t'],
            self.resultados['y_original'],
            y_filtrada,
            self.resultados['xi']
        )
        
        print(f"Reconstrucción completada")
        return y_filtrada
    
    def ejecutar_tarea_5(self):
        """Ejecuta Tarea 5: Implementación QR propia"""
        print("\n" + "="*70)
        print("EJECUTANDO TAREA 5: IMPLEMENTACIÓN QR PROPIA")
        print("="*70)
        
        if 'X' not in self.resultados:
            raise ValueError("Debe ejecutar Tarea 2 primero")
        
        # Demostrar QR con una submatriz para eficiencia
        X_sample = self.resultados['X'][:100, :100]  # Muestra pequeña
        
        # Descomposición QR usando Householder
        Q_h, R_h = self.qr_decomp.householder_qr(X_sample)
        
        # Descomposición QR usando Givens
        Q_g, R_g = self.qr_decomp.givens_qr(X_sample)
        
        # Verificar resultados
        error_h = np.linalg.norm(Q_h @ R_h - X_sample)
        error_g = np.linalg.norm(Q_g @ R_g - X_sample)
        
        print(f"QR Householder - Error: {error_h:.2e}")
        print(f"QR Givens - Error: {error_g:.2e}")
        
        return Q_h, R_h, Q_g, R_g
    
    def ejecutar_pipeline_completo(self):
        """Ejecuta el pipeline completo de filtrado SVD"""
        print("\n" + "*"*20)
        print("INICIANDO PIPELINE COMPLETO DE FILTRADO SVD")
        print("*"*20)
        
        try:
            # Ejecutar todas las tareas en secuencia
            self.ejecutar_tarea_1()    # Generación de señal
            self.ejecutar_tarea_2()    # Matriz de Hankel
            self.ejecutar_tarea_3()    # Descomposición SVD
            self.ejecutar_tarea_4()    # Reconstrucción
            self.ejecutar_tarea_5()    # QR propio
            
            # Resumen final
            self.generar_resumen_final()
            
            print("\n")
            print("PIPELINE COMPLETADO EXITOSAMENTE")
            print("-"*20)
            
        except Exception as e:
            print(f"\n Error en el pipeline: {str(e)}")
            raise
    
    def generar_resumen_final(self):
        """Genera un resumen final de todos los resultados"""
        print("\n")
        print("RESUMEN FINAL DE RESULTADOS")
        print("\n")
        
        print(f"• Señal original: {len(self.resultados['y_original'])} puntos")
        print(f"• Parámetro de ruido ξ: {self.resultados['xi']:.6f}")
        print(f"• Matriz de Hankel: {self.resultados['L']}×{self.resultados['K']}")
        print(f"• Rango de truncamiento: {self.resultados['r']}")
        print(f"• Señal filtrada: {len(self.resultados['y_filtrada'])} puntos")
        
        # Calcular métricas de calidad
        error_rms = np.sqrt(np.mean((self.resultados['y_original'] - self.resultados['y_filtrada'])**2))
        print(f"• Error RMS: {error_rms:.4f}")


def main():
    """Función principal para ejecutar el programa"""
    
    # Mostrar información del proyecto
    print("="*70)
    print("FILTRO DE RUIDO PARA SEÑALES TEMPORALES USANDO SVD")
    print("Examen Parcial - Álgebra Lineal Numérica (MO0014)")
    print("="*70)
    
    # Crear y ejecutar el filtro
    config = Config()
    filtro = FiltroSVD(config)
    
    # Ejecutar pipeline completo
    filtro.ejecutar_pipeline_completo()


if __name__ == "__main__":
    main()