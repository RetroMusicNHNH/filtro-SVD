"""
Módulo de Visualización Profesional para Filtro SVD

Visualizations with advanced techniques:
- Subplots optimizados para espacios reducidos
- Paletas de colores profesionales (seaborn-inspired)
- Métricas informativas en tiempo real
- Layouts adaptativos y tipografía mejorada

Autor: RetroMusicNHNH
Fecha: Octubre 2025
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


class Visualizer:
    """Clase para generar visualizaciones profesionales del proyecto"""

    def __init__(self, figsize: tuple = (12, 8), dpi: int = 100, style: str = 'seaborn-v0_8-whitegrid'):
        self.figsize = figsize
        self.dpi = dpi
        
        # Configurar estilo profesional
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # Paleta de colores profesional
        self.colors = {
            'primary': '#2E86AB',      # Azul profesional
            'secondary': '#A23B72',    # Rosa elegante
            'accent': '#F18F01',       # Naranja vibrante
            'success': '#C73E1D',      # Rojo elegante
            'neutral': '#6C757D',      # Gris neutro
            'background': '#F8F9FA'    # Gris claro de fondo
        }
        
        # Configuración de fuentes
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 16,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
        })

    # -----------------------------------------------------------------------
    # Señal original ruidosa con dashboard compacto
    # -----------------------------------------------------------------------
    def graficar_senal_original(self, t: np.ndarray, y: np.ndarray, xi: float,
                                titulo: str = "Señal Original Ruidosa") -> None:
        fig = plt.figure(figsize=(14, 8), dpi=self.dpi)
        gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1], width_ratios=[2, 1, 1])
        
        # Gráfica principal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, y, color=self.colors['primary'], linewidth=1.2, alpha=0.8,
                label=fr'$f(t) = \sin(t) + {xi:.4f} \cdot \sin(50t)$')
        ax1.set_xlabel('Tiempo $t$')
        ax1.set_ylabel('Amplitud')
        ax1.set_title(titulo, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', framealpha=0.9)
        
        # Panel de estadísticas
        ax2 = fig.add_subplot(gs[1, 0])
        stats_text = f'Media: {np.mean(y):.3f}\nStd: {np.std(y):.3f}\nRango: [{np.min(y):.2f}, {np.max(y):.2f}]\nParámetro $\\xi$: {xi:.6f}'
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor=self.colors['background'], alpha=0.8))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Histograma de amplitudes
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(y, bins=30, color=self.colors['accent'], alpha=0.7, density=True)
        ax3.set_title('Distribución', fontsize=10)
        ax3.set_xlabel('Amplitud')
        ax3.set_ylabel('Densidad')
        
        # Espectro de potencia (FFT)
        ax4 = fig.add_subplot(gs[1, 2])
        fft_y = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), t[1] - t[0])
        power = np.abs(fft_y)**2
        pos_idx = freqs > 0
        ax4.loglog(freqs[pos_idx], power[pos_idx], color=self.colors['secondary'], alpha=0.7)
        ax4.set_title('Espectro', fontsize=10)
        ax4.set_xlabel('Frecuencia')
        ax4.set_ylabel('Potencia')
        
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # Comparación profesional con métricas
    # -----------------------------------------------------------------------
    def graficar_comparacion(self, t: np.ndarray, y_original: np.ndarray,
                             y_filtrada: np.ndarray, xi: float,
                             titulo: str = "Análisis Comparativo: Original vs Filtrada") -> None:
        
        # Calcular métricas de calidad
        error_rms = np.sqrt(np.mean((y_original - y_filtrada)**2))
        correlacion = np.corrcoef(y_original, y_filtrada)[0, 1]
        snr_mejora = 20 * np.log10(np.std(y_original) / np.std(y_original - y_filtrada))
        
        fig = plt.figure(figsize=(16, 10), dpi=self.dpi)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        # Comparación principal
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, y_original, color=self.colors['primary'], linewidth=1.0, alpha=0.7, label='Original')
        ax1.plot(t, y_filtrada, color=self.colors['secondary'], linewidth=1.2, alpha=0.9, label='Filtrada SVD')
        ax1.set_xlabel('Tiempo $t$')
        ax1.set_ylabel('Amplitud')
        ax1.set_title(f'{titulo} ($\\xi = {xi:.4f}$)', fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Error/diferencia
        ax2 = fig.add_subplot(gs[1, 0])
        diferencia = y_original - y_filtrada
        ax2.plot(t, diferencia, color=self.colors['accent'], linewidth=1.0, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(t, diferencia, alpha=0.3, color=self.colors['accent'])
        ax2.set_xlabel('Tiempo $t$')
        ax2.set_ylabel('Error')
        ax2.set_title('Señal de Error', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Zoom detallado
        ax3 = fig.add_subplot(gs[1, 1])
        zoom_start, zoom_end = len(t)//3, 2*len(t)//3
        t_zoom = t[zoom_start:zoom_end]
        ax3.plot(t_zoom, y_original[zoom_start:zoom_end], color=self.colors['primary'], 
                linewidth=1.5, alpha=0.7, label='Original')
        ax3.plot(t_zoom, y_filtrada[zoom_start:zoom_end], color=self.colors['secondary'], 
                linewidth=1.5, alpha=0.9, label='Filtrada')
        ax3.set_xlabel('Tiempo $t$')
        ax3.set_ylabel('Amplitud')
        ax3.set_title('Vista Detallada (Centro)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Panel de métricas
        ax4 = fig.add_subplot(gs[2, :])
        metricas_texto = f'''
        📊 Métricas de Filtrado:
        • Error RMS: {error_rms:.6f}
        • Correlación: {correlacion:.4f}
        • Mejora SNR: {snr_mejora:.2f} dB
        • Reducción de ruido: {(1 - np.std(diferencia)/np.std(y_original))*100:.1f}%
        '''
        
        ax4.text(0.02, 0.95, metricas_texto, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['background'], alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.suptitle('Resultados del Filtro SVD', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # Valores singulares con análisis detallado
    # -----------------------------------------------------------------------
    def graficar_valores_singulares(self, sigma: np.ndarray, r: Optional[int] = None,
                                    titulo: str = "Análisis de Valores Singulares") -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        
        # Gráfica principal log-scale
        idx = np.arange(1, len(sigma) + 1)
        ax1.semilogy(idx, sigma, 'o-', color=self.colors['primary'], 
                    markersize=4, linewidth=1.5, alpha=0.8, label='$\\sigma_i$')
        
        if r is not None and 1 <= r <= len(sigma):
            ax1.axvline(r, color=self.colors['success'], linestyle='--', linewidth=2, 
                       label=f'Truncamiento $r = {r}$')
            ax1.fill_between(idx[r:], sigma[r:], alpha=0.3, color=self.colors['success'], 
                           label=f'Descartados ({len(sigma)-r})')
        
        ax1.set_xlabel('Índice $i$')
        ax1.set_ylabel('Magnitud $\\sigma_i$ (escala log)')
        ax1.set_title('Decaimiento de Valores Singulares', fontweight='bold')
        ax1.grid(True, which='both', ls='--', alpha=0.3)
        ax1.legend()
        
        # Análisis de diferencias consecutivas
        if len(sigma) > 1:
            diff_sigma = np.abs(np.diff(sigma))
            ax2.semilogy(idx[1:], diff_sigma, 's-', color=self.colors['accent'], 
                        markersize=3, linewidth=1.2, alpha=0.8)
            
            if r is not None and r > 1:
                ax2.axvline(r, color=self.colors['success'], linestyle='--', linewidth=2)
                
            ax2.set_xlabel('Índice $i$')
            ax2.set_ylabel('$|\\sigma_i - \\sigma_{i+1}|$')
            ax2.set_title('Diferencias Consecutivas', fontweight='bold')
            ax2.grid(True, which='both', ls='--', alpha=0.3)
        
        plt.suptitle(titulo, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # Zoom con subplots optimizados
    # -----------------------------------------------------------------------
    def graficar_zoom_regiones(self, t: np.ndarray, y_original: np.ndarray,
                               y_filtrada: Optional[np.ndarray] = None) -> None:
        """Vista de regiones con layout optimizado tipo dashboard"""
        regiones = [
            (-np.pi, np.pi, 'Región Central'),
            (0, np.pi/2, 'Alta Frecuencia'),
            (-2*np.pi, -np.pi, 'Lateral Izq.'),
            (np.pi, 2*np.pi, 'Lateral Der.')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=self.dpi)
        fig.suptitle('Vistas Regionales Detalladas', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, (a, b, nombre) in enumerate(regiones):
            idx = np.where((t >= a) & (t <= b))[0]
            if len(idx) < 2:
                axes[i].text(0.5, 0.5, 'Sin datos\nen región', ha='center', va='center')
                axes[i].set_title(nombre, fontweight='bold')
                continue
            
            # Plot con mejores colores y estilos
            axes[i].plot(t[idx], y_original[idx], color=self.colors['primary'], 
                        linewidth=1.2, alpha=0.8, label='Original')
            
            if y_filtrada is not None:
                axes[i].plot(t[idx], y_filtrada[idx], color=self.colors['secondary'], 
                           linewidth=1.4, alpha=0.9, label='Filtrada')
            
            axes[i].set_xlabel('Tiempo $t$')
            axes[i].set_ylabel('Amplitud')
            axes[i].set_title(f'{nombre}  $t \\in [{a:.2f}, {b:.2f}]$', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:  # Solo leyenda en el primer subplot
                axes[i].legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------------
    # Método adicional: resumen ejecutivo
    # -----------------------------------------------------------------------
    def generar_resumen_ejecutivo(self, t: np.ndarray, y_original: np.ndarray, 
                                 y_filtrada: np.ndarray, xi: float, r: int, 
                                 sigma: np.ndarray) -> None:
        """Genera un dashboard completo de una sola página"""
        fig = plt.figure(figsize=(16, 12), dpi=self.dpi)
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1.5, 1.5, 1], hspace=0.4, wspace=0.3)
        
        # Título principal
        fig.suptitle('Filtro SVD - Resumen Ejecutivo', fontsize=20, fontweight='bold', y=0.96)
        
        # Señales principales (arriba)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(t, y_original, color=self.colors['primary'], linewidth=1.0, alpha=0.7, label='Original')
        ax1.plot(t, y_filtrada, color=self.colors['secondary'], linewidth=1.2, alpha=0.9, label='Filtrada')
        ax1.set_title('Señales Completas', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Valores singulares (arriba derecha)
        ax2 = fig.add_subplot(gs[0, 2:])
        idx_sigma = np.arange(1, len(sigma) + 1)
        ax2.semilogy(idx_sigma, sigma, 'o-', color=self.colors['accent'], markersize=3)
        ax2.axvline(r, color=self.colors['success'], linestyle='--', linewidth=2)
        ax2.set_title(f'Valores Singulares (r={r})', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Error detallado (medio)
        ax3 = fig.add_subplot(gs[1, :2])
        error = y_original - y_filtrada
        ax3.plot(t, error, color=self.colors['accent'], linewidth=1.0)
        ax3.fill_between(t, error, alpha=0.3, color=self.colors['accent'])
        ax3.set_title('Error de Reconstrucción', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Estadísticas numéricas
        ax4 = fig.add_subplot(gs[1, 2:])
        stats = {
            'Parámetro ξ': f'{xi:.6f}',
            'Puntos N': f'{len(t):,}',
            'Matriz L×K': f'{800}×{len(t)-799}',
            'Rango r': f'{r}',
            'Error RMS': f'{np.sqrt(np.mean(error**2)):.2e}',
            'Correlación': f'{np.corrcoef(y_original, y_filtrada)[0,1]:.4f}'
        }
        
        y_pos = 0.9
        for key, val in stats.items():
            ax4.text(0.05, y_pos, f'{key}:', fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.55, y_pos, val, transform=ax4.transAxes, fontfamily='monospace')
            y_pos -= 0.14
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Métricas Clave', fontweight='bold')
        
        # Distribuciones (abajo)
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.hist(y_original, bins=40, alpha=0.6, color=self.colors['primary'], label='Original', density=True)
        ax5.hist(y_filtrada, bins=40, alpha=0.6, color=self.colors['secondary'], label='Filtrada', density=True)
        ax5.set_title('Distribuciones de Amplitud', fontweight='bold')
        ax5.legend()
        
        # Espectro comparativo
        ax6 = fig.add_subplot(gs[2, 2:])
        fft_orig = np.abs(np.fft.fft(y_original))
        fft_filt = np.abs(np.fft.fft(y_filtrada))
        freqs = np.fft.fftfreq(len(t), t[1]-t[0])
        pos_idx = (freqs > 0) & (freqs < 100)  # Limitar frecuencias
        
        ax6.loglog(freqs[pos_idx], fft_orig[pos_idx], color=self.colors['primary'], alpha=0.7, label='Original')
        ax6.loglog(freqs[pos_idx], fft_filt[pos_idx], color=self.colors['secondary'], alpha=0.9, label='Filtrada')
        ax6.set_title('Espectros de Frecuencia', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()