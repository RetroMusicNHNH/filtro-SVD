"""
Módulo de Reconstrucción de Señales por Anti‑diagonales

Implementa el método de PROMEDIOS DE ANTI‑DIAGONALES para recuperar la señal
filtrada y_r a partir de la matriz de Hankel truncada X_r, siguiendo el enunciado:

  y_r[i] = (1/(i+1)) * sum_{j=0}^{i} X_r[j, i-j],                 si 0 ≤ i ≤ L-1
  y_r[i] = (1/L)    * sum_{j=0}^{L-1} X_r[j, i-j],                si L-1 ≤ i ≤ K-1
  y_r[i] = (1/(N-i)) * sum_{j=i-K+1}^{L-1} X_r[j, i-j],           si K-1 ≤ i ≤ N-1

donde N = L + K − 1.

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np


class SignalReconstructor:
    """Clase para reconstrucción de señales usando promedios de anti‑diagonales"""

    def __init__(self, L: int, K: int):
        if L < 2 or K < 1:
            raise ValueError(f"Parámetros inválidos: L={L}, K={K}")
        self.L = L
        self.K = K
        self.N = L + K - 1

    def reconstruir_senal(self, Xr: np.ndarray) -> np.ndarray:
        """
        Reconstruye la señal y_r a partir de X_r mediante promedios de anti‑diagonales.

        Args:
            Xr: Matriz de Hankel truncada (L×K)
        Returns:
            y_r: Vector de longitud N = L + K − 1
        """
        if Xr.ndim != 2:
            raise ValueError("Xr debe ser 2D")
        L, K = Xr.shape
        if L != self.L or K != self.K:
            raise ValueError(f"Dimensiones incompatibles: Xr es {L}×{K}, esperado {self.L}×{self.K}")

        N = self.N
        y_r = np.zeros(N, dtype=float)

        # Caso 1: 0 ≤ i ≤ L-1
        for i in range(0, self.L):
            s = 0.0
            for j in range(0, i + 1):
                s += Xr[j, i - j]
            y_r[i] = s / (i + 1)

        # Caso 2: L-1 ≤ i ≤ K-1
        for i in range(self.L, self.K):
            # Promedio de L términos: j = 0..L-1, columnas i-j siempre válidas en este rango
            s = 0.0
            for j in range(0, self.L):
                s += Xr[j, i - j]
            y_r[i] = s / self.L

        # Caso 3: K-1 ≤ i ≤ N-1
        for i in range(self.K, N):
            s = 0.0
            # j recorre desde i-K+1 hasta L-1 (inclusive)
            j_ini = i - self.K + 1
            for j in range(j_ini, self.L):
                s += Xr[j, i - j]
            y_r[i] = s / (N - i)

        return y_r

    # Métodos auxiliares (si se quiere exponer los promedios por caso)
    def promedio_antidiagonal(self, Xr: np.ndarray, i: int) -> float:
        """Devuelve el promedio de la anti‑diagonal i‑ésima según el tramo correspondiente."""
        if not (0 <= i < self.N):
            raise ValueError(f"Índice i fuera de rango [0, {self.N-1}]")
        if i < self.L:
            return np.mean([Xr[j, i - j] for j in range(0, i + 1)])
        elif i < self.K:
            return np.mean([Xr[j, i - j] for j in range(0, self.L)])
        else:
            j_ini = i - self.K + 1
            return np.mean([Xr[j, i - j] for j in range(j_ini, self.L)])
