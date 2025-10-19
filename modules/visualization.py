"""
Módulo de Visualización para Filtro SVD

Este módulo centraliza todas las funciones de graficación usadas en el proyecto:
- Señal original ruidosa
- Comparación original vs filtrada
- Valores singulares y punto de truncamiento
- Vistas con zoom en diferentes regiones

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


class Visualizer:
    """Clase para generar todas las visualizaciones del proyecto"""

    def __init__(self, figsize: tuple = (12, 8), dpi: int = 300, style: str = 'default'):
        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')

    # ---------------------------------------------------------------------
    # Señal original ruidosa
    # ---------------------------------------------------------------------
    def graficar_senal_original(self, t: np.ndarray, y: np.ndarray, xi: float,
                                titulo: str = "Señal Original Ruidosa") -> None:
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(t, y, color='tab:blue', linewidth=1.0, alpha=0.85,
                 label=fr'f(t) = sin(t) + {xi:.4f}·sin(50t)')
        plt.xlabel('Tiempo t')
        plt.ylabel('Amplitud')
        plt.title(titulo)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Comparación original vs filtrada
    # ---------------------------------------------------------------------
    def graficar_comparacion(self, t: np.ndarray, y_original: np.ndarray,
                             y_filtrada: np.ndarray, xi: float,
                             titulo: str = "Comparación: Original vs Filtrada") -> None:
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(t, y_original, color='tab:blue', linewidth=0.9, alpha=0.7, label='Original')
        plt.plot(t, y_filtrada, color='tab:red', linewidth=1.0, alpha=0.9, label='Filtrada')
        plt.xlabel('Tiempo t')
        plt.ylabel('Amplitud')
        plt.title(titulo + fr"  (ξ = {xi:.4f})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Diferencia
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(t, y_original - y_filtrada, color='tab:green', linewidth=0.9, alpha=0.9, label='Diferencia')
        plt.xlabel('Tiempo t')
        plt.ylabel('Amplitud')
        plt.title('Diferencia: Original - Filtrada')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Valores singulares y punto de truncamiento
    # ---------------------------------------------------------------------
    def graficar_valores_singulares(self, sigma: np.ndarray, r: Optional[int] = None,
                                    titulo: str = "Valores singulares (Σ)") -> None:
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        idx = np.arange(1, len(sigma) + 1)
        plt.semilogy(idx, sigma, 'o-', color='tab:orange', label='σ_i')
        plt.xlabel('Índice i')
        plt.ylabel('Magnitud (escala log)')
        plt.title(titulo)
        plt.grid(True, which='both', ls='--', alpha=0.3)
        
        if r is not None and 1 <= r <= len(sigma):
            plt.axvline(r, color='tab:red', linestyle='--', label=fr'corte r = {r}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Vistas con zoom
    # ---------------------------------------------------------------------
    def graficar_zoom_regiones(self, t: np.ndarray, y_original: np.ndarray,
                               y_filtrada: Optional[np.ndarray] = None) -> None:
        """Genera múltiples vistas con zoom en regiones de interés"""
        regiones = [
            (-(np.pi), np.pi, 'Región Central'),
            (0, np.pi/2, 'Zoom Detallado 1'),
            (-2*np.pi, -1*np.pi, 'Zoom Lateral 1'),
            (1*np.pi, 2*np.pi, 'Zoom Lateral 2')
        ]
        
        for a, b, nombre in regiones:
            idx = np.where((t >= a) & (t <= b))[0]
            if len(idx) < 2:
                continue
            
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            plt.plot(t[idx], y_original[idx], color='tab:blue', linewidth=0.9, alpha=0.8, label='Original')
            if y_filtrada is not None:
                plt.plot(t[idx], y_filtrada[idx], color='tab:red', linewidth=1.0, alpha=0.9, label='Filtrada')
            
            plt.xlabel('Tiempo t')
            plt.ylabel('Amplitud')
            plt.title(f'{nombre} [{a:.2f}, {b:.2f}]')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
