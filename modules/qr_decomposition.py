"""
Módulo: Implementación propia de descomposición QR

Incluye dos variantes clásicas:
- Householder: estable y eficiente para densas.
- Givens: útil para introducir ceros individualmente y para matrices dispersas/actualizaciones.

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import numpy as np
from typing import Tuple


class QRDecomposition:
    """Descomposición QR mediante Householder y Givens"""

    # ---------------------------------------------------------------
    # Householder QR
    # ---------------------------------------------------------------
    @staticmethod
    def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorización QR usando reflexiones de Householder.
        
        Args:
            A: Matriz m×n (m ≥ n recomendado)
        Returns:
            Q, R con Q ortogonal (m×m) y R triangular superior (m×n)
        """
        A = A.astype(float).copy()
        m, n = A.shape
        Q = np.eye(m)

        for k in range(min(m, n)):
            # Extraer el vector x (columna k desde fila k)
            x = A[k:, k]
            normx = np.linalg.norm(x)
            if normx == 0:
                continue

            # Elegir el signo para evitar cancelación catastrófica
            sign = 1.0 if x[0] >= 0 else -1.0
            u1 = x[0] + sign * normx
            v = x.copy()
            v[0] = u1
            v = v / np.linalg.norm(v)

            # Construir H_k = I - 2 v v^T sobre el subbloque
            Hk_sub = np.eye(m - k) - 2.0 * np.outer(v, v)

            # Aplicar a A: A_k := H_k A_k
            A[k:, k:] = Hk_sub @ A[k:, k:]

            # Acumular Q: Q := Q H_k^T (pero H es simétrica)
            Q[:, k:] = Q[:, k:] @ Hk_sub

        R = A
        return Q, R

    # ---------------------------------------------------------------
    # Givens QR
    # ---------------------------------------------------------------
    @staticmethod
    def givens_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Factorización QR mediante rotaciones de Givens.
        
        Args:
            A: Matriz m×n
        Returns:
            Q, R con Q ortogonal y R triangular superior
        """
        A = A.astype(float).copy()
        m, n = A.shape
        Q = np.eye(m)
        R = A

        for j in range(n):
            for i in range(m - 1, j, -1):
                # Anular R[i, j] usando fila i y j
                a = R[i - 1, j]
                b = R[i, j]
                if abs(b) < 1e-15:
                    continue

                c, s = QRDecomposition._givens_cs(a, b)

                G = np.eye(m)
                G[i - 1, i - 1] = c
                G[i - 1, i] = s
                G[i, i - 1] = -s
                G[i, i] = c

                R = G @ R
                Q = Q @ G.T

        return Q, R

    @staticmethod
    def _givens_cs(a: float, b: float) -> Tuple[float, float]:
        """Calcula c y s para una rotación de Givens que anule b."""
        r = np.hypot(a, b)
        if r == 0:
            return 1.0, 0.0
        c = a / r
        s = b / r
        return c, s

    # ---------------------------------------------------------------
    # Utilidad opcional: resolver mínimos cuadrados con QR
    # ---------------------------------------------------------------
    @staticmethod
    def resolver_minimos_cuadrados(A: np.ndarray, b: np.ndarray, metodo: str = 'householder') -> Tuple[np.ndarray, float]:
        """
        Resuelve min ||Ax - b||_2 via QR.
        
        Args:
            A: Matriz m×n
            b: Vector m
            metodo: 'householder' o 'givens'
        Returns:
            x: Solución de mínimos cuadrados
            residuo: ||Ax - b||_2
        """
        if metodo == 'householder':
            Q, R = QRDecomposition.householder_qr(A)
        elif metodo == 'givens':
            Q, R = QRDecomposition.givens_qr(A)
        else:
            raise ValueError("Método QR no reconocido")

        # Resolver R x = Q^T b (tomar la parte superior de R si m>n)
        m, n = A.shape
        Qtb = Q.T @ b
        R1 = R[:n, :n]
        y = Qtb[:n]

        # Sustitución hacia atrás
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if abs(R1[i, i]) < 1e-15:
                raise np.linalg.LinAlgError("R es singular o mal condicionada")
            x[i] = (y[i] - R1[i, i + 1:] @ x[i + 1:]) / R1[i, i]

        residuo = np.linalg.norm(A @ x - b)
        return x, residuo
