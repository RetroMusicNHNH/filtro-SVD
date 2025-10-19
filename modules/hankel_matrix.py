"""
Módulo para construcción de Matriz de Hankel

Implementa el mapeo de la señal y = (y1, ..., yN) a una matriz X de tamaño L×K
con estructura de Hankel, según la especificación del examen.

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np
from typing import Tuple


class HankelMatrix:
    """Clase para operaciones con matrices de Hankel"""

    def __init__(self, L: int = 800):
        if L <= 1:
            raise ValueError(f"L debe ser ≥ 2, got {L}")
        self.L = L

    def calcular_K(self, N: int) -> int:
        """Calcula K = N - L + 1"""
        return N - self.L + 1

    def validar_dimensiones(self, y: np.ndarray) -> None:
        """Valida que L cumpla 2 ≤ L ≤ N y que K ≥ 1"""
        if y.ndim != 1:
            raise ValueError("La señal y debe ser un vector 1D")
        N = len(y)
        if not (2 <= self.L <= N):
            raise ValueError(f"Se requiere 2 ≤ L ≤ N, got L={self.L}, N={N}")
        K = self.calcular_K(N)
        if K < 1:
            raise ValueError(f"K debe ser ≥ 1; con N={N} y L={self.L} se tiene K={K}")

    def construir_matriz(self, y: np.ndarray) -> np.ndarray:
        """
        Construye la matriz de Hankel X de tamaño L×K a partir de la señal y.

        X = [x1, x2, ..., xK]
        con columnas xk = [y_k, y_{k+1}, ..., y_{k+L-1}]^T

        Args:
            y: Señal 1D de longitud N

        Returns:
            X: Matriz de Hankel de dimensión (L, K)
        """
        self.validar_dimensiones(y)
        N = len(y)
        K = self.calcular_K(N)

        # Construcción vectorizada usando broadcasting de índices
        i = np.arange(self.L).reshape(-1, 1)  # (L, 1)
        j = np.arange(K).reshape(1, -1)       # (1, K)
        idx = i + j                           # (L, K)
        X = y[idx]
        return X

    def reconstruir_por_antidiagonales(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruye la señal y a partir de una matriz de Hankel X usando promedios
        de anti-diagonales (para validación y pruebas).

        Args:
            X: Matriz de Hankel (L×K)

        Returns:
            y_rec: Vector 1D de longitud N = L + K - 1
        """
        if X.ndim != 2:
            raise ValueError("X debe ser 2D")
        L, K = X.shape
        N = L + K - 1
        y_rec = np.zeros(N)
        counts = np.zeros(N)

        # Recorrer todas las entradas y acumular por anti-diagonales i+j
        for r in range(L):
            for c in range(K):
                s = r + c
                y_rec[s] += X[r, c]
                counts[s] += 1

        # Promediar
        counts[counts == 0] = 1
        y_rec /= counts
        return y_rec
