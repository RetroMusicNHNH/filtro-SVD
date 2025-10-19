"""
Módulo para Procesamiento SVD y Determinación de Rango

Implementa:
- Descomposición SVD: X = U · Σ · V^T
- Criterio del examen para determinar r = i - 2
- Construcción de X_r truncando valores singulares de alta frecuencia

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np
from typing import Tuple


class SVDProcessor:
    """Clase para procesamiento de descomposición SVD"""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # SVD completa
    # ------------------------------------------------------------------
    def descomponer_svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza descomposición SVD: X = U · Σ · V^T

        Args:
            X: Matriz de Hankel (L×K)
        Returns:
            U, sigma, Vt: Componentes SVD (full_matrices=False)
        """
        if X.ndim != 2:
            raise ValueError("X debe ser 2D")
        U, sigma, Vt = np.linalg.svd(X, full_matrices=False)
        return U, sigma, Vt

    # ------------------------------------------------------------------
    # Criterio de determinación de rango (del enunciado)
    # ------------------------------------------------------------------
    def determinar_rango(self, sigma: np.ndarray, L: int) -> int:
        """
        Determina r usando el criterio del examen:

        Sea i el menor índice tal que
            |σ_i − σ_{i+1}| ≤ (max σ − min σ) / L
            y también |σ_{i+1} − σ_{i+2}| ≤ (max σ − min σ) / L
        entonces r = i − 2.

        Notas:
        - Indices en el enunciado son 1‑based; aquí usamos 0‑based.
        - Se asegura r ≥ 1.
        """
        if len(sigma) < 3:
            return max(1, len(sigma) - 1)

        smax = np.max(sigma)
        smin = np.min(sigma)
        thresh = (smax - smin) / max(L, 1)

        # Buscar el menor i (0‑based) que cumpla las dos condiciones consecutivas
        found_i = None
        for i in range(len(sigma) - 2):
            cond1 = abs(sigma[i] - sigma[i + 1]) <= thresh
            cond2 = abs(sigma[i + 1] - sigma[i + 2]) <= thresh
            if cond1 and cond2:
                found_i = i
                break

        if found_i is None:
            # Fallback conservador: mantener la mitad de los valores singulares
            r = max(1, len(sigma) // 2)
        else:
            r = max(1, found_i - 1)  # i − 2 con 1‑based equivale a i − 1 con 0‑based

        # No exceder L por seguridad
        r = min(r, L)
        return r

    # ------------------------------------------------------------------
    # Construcción de matriz truncada X_r
    # ------------------------------------------------------------------
    def construir_matriz_truncada(self, U: np.ndarray, sigma: np.ndarray, Vt: np.ndarray, r: int) -> np.ndarray:
        """
        Construye X_r = U_r · Σ_r · V_r^T
        
        Args:
            U, sigma, Vt: Componentes de la SVD
            r: Rango a conservar
        Returns:
            X_r: Aproximación truncada de rango r
        """
        if r < 1:
            raise ValueError("r debe ser ≥ 1")
        r = min(r, len(sigma))

        U_r = U[:, :r]
        Sigma_r = np.diag(sigma[:r])
        Vt_r = Vt[:r, :]
        X_r = U_r @ Sigma_r @ Vt_r
        return X_r
