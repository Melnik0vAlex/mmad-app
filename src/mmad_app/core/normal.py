# -*- coding: utf-8 -*-
# src/mmad_app/core/normal.py
"""
Стандартное нормальное распределение без SciPy.

Содержит:
- normal_cdf(z)  : Φ(z) через функцию ошибок erf
- normal_ppf(p)  : Φ^{-1}(p) через рациональную аппроксимацию Acklam

Примечание
----------
Аппроксимация Acklam широко используется для вычисления inverse-normal
в статистике и финансовых приложениях.

Ограничения:
-----------
- p должен быть в (0, 1). Значения 0 и 1 обрезаются к eps.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

_SQRT2 = math.sqrt(2.0)


def normal_cdf(z: np.ndarray | Iterable[float]) -> np.ndarray:
    """
    CDF стандартного нормального распределения Φ(z).

    Формула:
        Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))

    Parameters
    ----------
    z:
        Значения z (скаляр или массив).

    Returns
    -------
    np.ndarray
        Значения Φ(z) в диапазоне (0, 1).
    """
    z_arr = np.asarray(z, dtype=float)

    # np.vectorize удобен и достаточно быстрый для малых массивов (у нас 5–10 точек)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z_arr / _SQRT2))


def normal_ppf(p: np.ndarray | Iterable[float], *, eps: float = 1e-12) -> np.ndarray:
    """
    Inverse CDF стандартного нормального распределения Φ^{-1}(p).

    Используется аппроксимация Peter John Acklam (рациональные функции).

    Parameters
    ----------
    p:
        Вероятности (0..1). Значения 0 и 1 будут "подрезаны" к eps.
    eps:
        Малое число для защиты от ±inf.

    Returns
    -------
    np.ndarray
        Значения z такие, что Φ(z) = p.
    """
    p_arr = np.asarray(p, dtype=float)
    p_arr = np.clip(p_arr, eps, 1.0 - eps)

    # Векторизуем скалярную реализацию
    ppf_vec = np.vectorize(_normal_ppf_scalar)
    return ppf_vec(p_arr)


def _normal_ppf_scalar(p: float) -> float:
    """
    Скалярная Φ^{-1}(p) по аппроксимации Acklam.

    Алгоритм разделяет область вероятностей на 3 зоны:
    - нижний хвост p < plow
    - центральная область plow <= p <= phigh
    - верхний хвост p > phigh
    """
    # Коэффициенты Acklam
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        # Нижний хвост
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den

    if p > phigh:
        # Верхний хвост (симметрия)
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -(num / den)

    # Центральная область
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num / den
