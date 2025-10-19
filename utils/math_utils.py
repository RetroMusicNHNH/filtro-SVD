"""
Utilidades Matemáticas para el Proyecto Filtro SVD

Este módulo contiene funciones auxiliares y utilidades matemáticas
que son utilizadas por múltiples módulos del proyecto.

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np
from typing import Tuple, Union, Optional
import warnings


class MathUtils:
    """
    Clase con utilidades matemáticas estáticas para el proyecto
    """
    
    @staticmethod
    def calcular_error_rms(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula el error RMS (Root Mean Square) entre dos señales
        
        Args:
            y_true: Señal de referencia
            y_pred: Señal predicha/filtrada
            
        Returns:
            float: Error RMS
        """
        if len(y_true) != len(y_pred):
            raise ValueError(f"Las señales deben tener la misma longitud: {len(y_true)} != {len(y_pred)}")
        
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    @staticmethod
    def calcular_snr(senal: np.ndarray, ruido: np.ndarray) -> float:
        """
        Calcula la relación señal-ruido (SNR) en dB
        
        Args:
            senal: Señal limpia
            ruido: Componente de ruido
            
        Returns:
            float: SNR en decibelios
        """
        potencia_senal = np.mean(senal**2)
        potencia_ruido = np.mean(ruido**2)
        
        if potencia_ruido == 0:
            return float('inf')
        
        snr_linear = potencia_senal / potencia_ruido
        return 10 * np.log10(snr_linear)
    
    @staticmethod
    def normalizar_matriz(X: np.ndarray, tipo: str = 'frobenius') -> Tuple[np.ndarray, float]:
        """
        Normaliza una matriz según diferentes normas
        
        Args:
            X: Matriz a normalizar
            tipo: Tipo de norma ('frobenius', 'spectral', 'nuclear')
            
        Returns:
            tuple: (X_normalizada, factor_normalizacion)
        """
        if tipo == 'frobenius':
            norma = np.linalg.norm(X, 'fro')
        elif tipo == 'spectral':
            norma = np.linalg.norm(X, 2)
        elif tipo == 'nuclear':
            norma = np.sum(np.linalg.svd(X, compute_uv=False))
        else:
            raise ValueError(f"Tipo de norma no reconocido: {tipo}")
        
        if norma == 0:
            warnings.warn("La matriz tiene norma cero, no se puede normalizar")
            return X, 0
        
        return X / norma, norma
    
    @staticmethod
    def verificar_ortogonalidad(Q: np.ndarray, tolerance: float = 1e-12) -> Tuple[bool, float]:
        """
        Verifica si una matriz es ortogonal
        
        Args:
            Q: Matriz a verificar
            tolerance: Tolerancia para la verificación
            
        Returns:
            tuple: (es_ortogonal, error_ortogonalidad)
        """
        QTQ = Q.T @ Q
        I = np.eye(Q.shape[1])
        error = np.linalg.norm(QTQ - I, 'fro')
        
        return error < tolerance, error
    
    @staticmethod
    def calcular_rango_numerico(X: np.ndarray, tolerance: float = 1e-12) -> int:
        """
        Calcula el rango numérico de una matriz usando SVD
        
        Args:
            X: Matriz de entrada
            tolerance: Tolerancia para considerar valores singulares como cero
            
        Returns:
            int: Rango numérico de la matriz
        """
        sigma = np.linalg.svd(X, compute_uv=False)
        return np.sum(sigma > tolerance)
    
    @staticmethod
    def proyeccion_ortogonal(v: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calcula la proyección ortogonal de v sobre u
        
        Args:
            v: Vector a proyectar
            u: Vector base de la proyección
            
        Returns:
            np.ndarray: Proyección de v sobre u
        """
        if np.linalg.norm(u) == 0:
            return np.zeros_like(v)
        
        return (np.dot(v, u) / np.dot(u, u)) * u
    
    @staticmethod
    def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
        """
        Aplica el proceso de Gram-Schmidt para ortogonalizar vectores
        
        Args:
            vectors: Matriz donde cada columna es un vector
            
        Returns:
            np.ndarray: Matriz con vectores ortonormales
        """
        n, m = vectors.shape
        Q = np.zeros((n, m))
        
        for j in range(m):
            v = vectors[:, j].copy()
            
            # Ortogonalizar contra vectores anteriores
            for i in range(j):
                v = v - MathUtils.proyeccion_ortogonal(v, Q[:, i])
            
            # Normalizar
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-12:
                Q[:, j] = v / norm_v
            else:
                warnings.warn(f"Vector {j} es linealmente dependiente")
                Q[:, j] = 0
        
        return Q
    
    @staticmethod
    def crear_matriz_hankel_indices(N: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea matrices de índices para construir matriz de Hankel eficientemente
        
        Args:
            N: Longitud de la señal
            L: Número de filas de la matriz de Hankel
            
        Returns:
            tuple: (indices_fila, indices_columna) para indexación vectorizada
        """
        K = N - L + 1
        
        # Crear matrices de índices
        i = np.arange(L).reshape(-1, 1)
        j = np.arange(K)
        
        indices = i + j
        
        return indices, (L, K)
    
    @staticmethod
    def calcular_energia_banda_frecuencial(Y: np.ndarray, 
                                         frecuencias: np.ndarray,
                                         banda: Tuple[float, float]) -> float:
        """
        Calcula la energía en una banda de frecuencias específica
        
        Args:
            Y: Espectro de frecuencias (FFT)
            frecuencias: Array de frecuencias correspondientes
            banda: Tupla (f_min, f_max) definiendo la banda
            
        Returns:
            float: Energía en la banda especificada
        """
        mask = (frecuencias >= banda[0]) & (frecuencias <= banda[1])
        return np.sum(np.abs(Y[mask])**2)
    
    @staticmethod
    def encontrar_picos_espectro(magnitudes: np.ndarray,
                               frecuencias: np.ndarray,
                               altura_minima: float = None,
                               distancia_minima: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encuentra picos en el espectro de frecuencias
        
        Args:
            magnitudes: Magnitudes del espectro
            frecuencias: Frecuencias correspondientes
            altura_minima: Altura mínima de los picos
            distancia_minima: Distancia mínima entre picos (en muestras)
            
        Returns:
            tuple: (frecuencias_picos, magnitudes_picos)
        """
        # Implementación simple de detección de picos
        picos = []
        
        for i in range(1, len(magnitudes) - 1):
            if (magnitudes[i] > magnitudes[i-1] and 
                magnitudes[i] > magnitudes[i+1]):
                
                if altura_minima is None or magnitudes[i] >= altura_minima:
                    # Verificar distancia mínima con picos anteriores
                    if not picos or (i - picos[-1]) >= distancia_minima:
                        picos.append(i)
        
        if picos:
            picos = np.array(picos)
            return frecuencias[picos], magnitudes[picos]
        else:
            return np.array([]), np.array([])
    
    @staticmethod
    def interpolacion_lagrange(x: np.ndarray, 
                             y: np.ndarray, 
                             x_interp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Interpolación usando polinomios de Lagrange
        
        Args:
            x: Puntos x conocidos
            y: Valores y conocidos
            x_interp: Puntos donde interpolar
            
        Returns:
            Valores interpolados
        """
        n = len(x)
        
        if isinstance(x_interp, (int, float)):
            x_interp = np.array([x_interp])
            return_scalar = True
        else:
            return_scalar = False
        
        y_interp = np.zeros_like(x_interp, dtype=float)
        
        for i in range(len(x_interp)):
            xi = x_interp[i]
            yi = 0
            
            for j in range(n):
                # Calcular el término j-ésimo de Lagrange
                termino = y[j]
                for k in range(n):
                    if k != j:
                        termino *= (xi - x[k]) / (x[j] - x[k])
                yi += termino
            
            y_interp[i] = yi
        
        return y_interp[0] if return_scalar else y_interp
    
    @staticmethod
    def matriz_condicion_numero(A: np.ndarray) -> float:
        """
        Calcula el número de condición de una matriz
        
        Args:
            A: Matriz de entrada
            
        Returns:
            float: Número de condición
        """
        return np.linalg.cond(A)
    
    @staticmethod
    def descomposicion_valores_singulares_truncada(A: np.ndarray, 
                                                  k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza SVD truncada manteniendo solo los k valores singulares más grandes
        
        Args:
            A: Matriz de entrada
            k: Número de valores singulares a mantener
            
        Returns:
            tuple: (U_k, sigma_k, Vt_k) componentes truncadas
        """
        U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
        
        # Truncar a k componentes
        k = min(k, len(sigma))
        
        U_k = U[:, :k]
        sigma_k = sigma[:k]
        Vt_k = Vt[:k, :]
        
        return U_k, sigma_k, Vt_k


# Funciones de utilidad independientes
def validar_matriz_cuadrada(A: np.ndarray) -> None:
    """
    Valida que una matriz sea cuadrada
    
    Args:
        A: Matriz a validar
        
    Raises:
        ValueError: Si la matriz no es cuadrada
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"La matriz debe ser cuadrada, got shape {A.shape}")


def validar_dimensiones_compatibles(A: np.ndarray, B: np.ndarray, operacion: str = "multiplicación") -> None:
    """
    Valida que dos matrices tengan dimensiones compatibles
    
    Args:
        A: Primera matriz
        B: Segunda matriz
        operacion: Tipo de operación a realizar
        
    Raises:
        ValueError: Si las dimensiones no son compatibles
    """
    if operacion == "multiplicación":
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Dimensiones incompatibles para multiplicación: {A.shape} x {B.shape}")
    elif operacion == "suma":
        if A.shape != B.shape:
            raise ValueError(f"Dimensiones incompatibles para suma: {A.shape} != {B.shape}")


if __name__ == "__main__":
    # Demostración de las utilidades matemáticas
    print("Demostración de Utilidades Matemáticas")
    print("="*50)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    A = np.random.randn(5, 3)
    
    # Demostrar ortogonalización
    Q = MathUtils.gram_schmidt(A)
    es_ortogonal, error = MathUtils.verificar_ortogonalidad(Q)
    
    print(f"✓ Ortogonalización: {es_ortogonal}, Error: {error:.2e}")
    
    # Demostrar cálculo de rango
    rango = MathUtils.calcular_rango_numerico(A)
    print(f"✓ Rango numérico: {rango}")
    
    # Demostrar SVD truncada
    U_k, s_k, Vt_k = MathUtils.descomposicion_valores_singulares_truncada(A, 2)
    print(f"✓ SVD truncada: U{U_k.shape}, σ{s_k.shape}, V^T{Vt_k.shape}")