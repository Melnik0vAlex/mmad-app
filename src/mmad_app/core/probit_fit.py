# -*- coding: utf-8 -*-
"""
Пробит-аппроксимация кумулятивной кривой (S-кривая) для APSD.

Идея (логнормальная модель):
    F(d) = Phi((ln d - mu) / sigma)

Пробит-линейнизация:
    z = Phi^{-1}(F) = a*log(d) + b
Иногда используют шкалу probit = z + 5 (сдвиг на 5).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mmad_app.core.models import ProbitFitResult


def _clip_fraction(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Ограничивает вероятности, чтобы избежать бесконечностей norm.ppf(0) и norm.ppf(1).
    """
    return np.clip(p, eps, 1.0 - eps)


def fit_probit_line(
    diam_um: np.ndarray,
    cum_pct: np.ndarray,
) -> ProbitFitResult:
    """
    Строит пробит-линейную регрессию:
        z = a*log10(d) + b

    Возвращает коэффициенты (a, b) и R^2.
    """
    x = np.asarray(diam_um, dtype=float)
    y_pct = np.asarray(cum_pct, dtype=float)

    # Используем только положительные диаметры и валидные проценты
    mask = (x > 0.0) & np.isfinite(x) & np.isfinite(y_pct)
    x = x[mask]
    y_pct = y_pct[mask]

    # Переводим проценты в долю (0..1)
    p = _clip_fraction(y_pct / 100.0)

    # Пробит: z = Phi^{-1}(p)
    z = norm.ppf(p)

    # Регрессия по X = log10(d)
    lx = np.log10(x)

    # Линейная аппроксимация z = a*lx + b
    a, b = np.polyfit(lx, z, deg=1)

    # R^2 для качества подгонки
    z_hat = a * lx + b
    ss_res = float(np.sum((z - z_hat) ** 2))
    ss_tot = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ProbitFitResult(a=float(a), b=float(b), r2=float(r2))


def predict_cumulative_from_probit(
    fit: ProbitFitResult,
    d_um: np.ndarray,
) -> np.ndarray:
    """
    По пробит-модели возвращает cumulative undersize в процентах для заданных диаметров.
    """
    d = np.asarray(d_um, dtype=float)
    mask = d > 0.0
    out = np.full_like(d, fill_value=np.nan, dtype=float)

    lx = np.log10(d[mask])
    z = fit.a * lx + fit.b

    # Возвращаемся из пробита в вероятность
    p = norm.cdf(z)
    out[mask] = 100.0 * p
    return out


def probit_scale(pct: np.ndarray) -> np.ndarray:
    """
    Перевод cumulative (%) в шкалу probit (z + 5).
    Удобно для "пробит-графика", где ось Y от 0 до 10.
    """
    p = _clip_fraction(np.asarray(pct, dtype=float) / 100.0)
    return norm.ppf(p) + 5.0