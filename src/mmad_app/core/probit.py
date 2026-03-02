# -*- coding: utf-8 -*-
# src/mmad_app/core/probit.py
"""Пробит-аппроксимация кумулятивной кривой (S-кривая) для APSD."""

from __future__ import annotations

import numpy as np
from mmad_app.core.normal import normal_cdf, normal_ppf

from mmad_app.core.models import ProbitLine


def clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ограничение вероятностей, чтобы norm.ppf не дал +/-inf."""
    return np.clip(p, eps, 1.0 - eps)


def fit_probit(diam: np.ndarray, cum_pct: np.ndarray) -> ProbitLine:
    """
    Строит пробит-линейную регрессию (log–probit).

    Модель:
        probit = a * log10(d) + b

    Где:
        p = cum_pct / 100
        z = Φ⁻¹(p)
        probit = z + 5

    Параметры
    ---------
    diam:
        Диаметры (мкм).
    cum_pct:
        Кумулятивная доля undersize (%).

    Возвращает
    ----------
    ProbitLine
        Параметры линии (a, b) и RMSE в пробит-единицах.
    """
    x = np.asarray(diam, dtype=float)
    y_pct = np.asarray(cum_pct, dtype=float)

    # Используем только:
    # - конечные значения
    # - положительные диаметры
    # - проценты строго внутри (0, 100), чтобы исключить 0 и 100
    mask = (
        np.isfinite(x)
        & np.isfinite(y_pct)
        & (x > 0.0)
        & (y_pct > 0.0)
        & (y_pct < 100.0)
    )

    x = x[mask]
    y_pct = y_pct[mask]

    if x.size < 2:
        return ProbitLine(a=float("nan"), b=float("nan"), rmse=float("nan"), r2=float("nan"))

    # Перевод проценты в долю (0..1)
    p = clip_prob(y_pct / 100.0)

    # Пробит: z = Phi^{-1}(p) + 5
    z = normal_ppf(p)
    probit = z + 5.0

    # Регрессия по X = log10(d)
    lx = np.log10(x)

    # Линейная аппроксимация probit = a*lx + b
    a, b = np.polyfit(lx, probit, deg=1)

    # RMSE для качества подгонки
    probit_hat = a * lx + b

    # RMSE
    rmse = float(np.sqrt(np.mean((probit - probit_hat) ** 2)))

    # R²
    ss_res = float(np.sum((probit - probit_hat) ** 2))
    ss_tot = float(np.sum((probit - np.mean(probit)) ** 2))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ProbitLine(a=float(a), b=float(b), rmse=float(rmse), r2=float(r2))


def predict_cumulative_from_probit(
    fit: ProbitLine,
    diam: np.ndarray,
) -> np.ndarray:
    """
    Предсказывает накопленную долю массы частиц с диаметром меньше заданного
    (cumulative undersize) по пробит-модели.

    Параметры
    ---------
    fit:
        Параметры пробит-линии.
    diam:
        Диаметры (мкм).

    Возвращает
    ----------
    np.ndarray
        Накопленная доля массы частиц с диаметром < d, выраженная в процентах.
    """
    d = np.asarray(diam, dtype=float)

    # Для d <= 0 логарифм не определён
    mask = np.isfinite(d) & (d > 0.0)
    out = np.full_like(d, fill_value=np.nan, dtype=float)

    if not np.any(mask):
        return out

    # Вычисляем probit по модели
    lx = np.log10(d[mask])
    probit = fit.a * lx + fit.b

    # norm.cdf ожидает z = Φ^{-1}(p), а не probit (z + 5)
    z = probit - 5.0
    p = normal_cdf(z)

    out[mask] = 100.0 * p
    return out


def probit_scale(pct: np.ndarray) -> np.ndarray:
    """
    Преобразует накопленную долю массы частиц (< d), выраженную в процентах,
    в шкалу probit.

    Преобразование:
        p = F(d) / 100
        z = Φ⁻¹(p)
        probit = z + 5

    Параметры
    ---------
    pct:
        Накопленная доля массы частиц с диаметром меньше d,
        выраженная в процентах (0..100).

    Возвращает
    ----------
    np.ndarray
        Значения в шкале probit (z + 5), соответствующие заданным процентам.
    """
    p = clip_prob(np.asarray(pct, dtype=float) / 100.0)
    return normal_ppf(p) + 5.0
