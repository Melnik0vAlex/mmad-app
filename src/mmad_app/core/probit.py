# -*- coding: utf-8 -*-
# src/mmad_app/core/probit.py
"""Пробит-аппроксимация кумулятивной кривой (S-кривая) для APSD."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from mmad_app.core.models import ProbitLine


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ограничение вероятностей, чтобы norm.ppf не дал +/-inf."""
    return np.clip(p, eps, 1.0 - eps)


def fit_probit(diam_um: np.ndarray, cum_pct: np.ndarray) -> ProbitLine:
    """Строит пробит-линейную регрессию:
        p = cum_pct / 100
        z = Phi^{-1}(p)
        probit = z + 5
        probit = a*log10(d) + b
    """
    x = np.asarray(diam_um, dtype=float)
    y_pct = np.asarray(cum_pct, dtype=float)

    # Используются только положительные диаметры и валидные проценты
    mask = (x > 0.0) & np.isfinite(x) & np.isfinite(y_pct)
    x = x[mask]
    y_pct = y_pct[mask]

    # Перевод проценты в долю (0..1)
    p = clip_prob(y_pct / 100.0)

    # Пробит: z = Phi^{-1}(p) + 5
    z = norm.ppf(p)
    probit = z + 5.0

    # Регрессия по X = log10(d)
    lx = np.log10(x)

    # Линейная аппроксимация probit = a*lx + b
    a, b = np.polyfit(lx, probit, deg=1)

    # R^2 для качества подгонки
    z_hat = a * lx + b
    ss_res = float(np.sum((z - z_hat) ** 2))
    ss_tot = float(np.sum((z - float(np.mean(z))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ProbitLine(a=float(a), b=float(b), r2=float(r2))


def predict_cumulative_from_probit(
    fit: ProbitLine,
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
    """
    p = clip_prob(np.asarray(pct, dtype=float) / 100.0)
    return norm.ppf(p) + 5.0
