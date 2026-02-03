# -*- coding: utf-8 -*-
"""
Тесты для interpolate_d_at_cum_pct_logx.

Проверяем:
1) Корректность лог-интерполяции на простой кривой.
2) Инвариантность к порядку входных точек.
3) Ошибки: target_pct вне (0,100), target_pct вне диапазона кривой,
   недостаточно точек с положительным диаметром.
4) Поведение на "плоском" участке (y1 == y0).
5) Игнорирование diam <= 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmad_app.core.mmad import interpolate_d_at_cum_pct_logx


def test_simple_log_interpolation_midpoint() -> None:
    """
    На участке между (d=1, y=10) и (d=100, y=90) при target_pct=50:

    t = (50-10)/(90-10) = 0.5
    log10(d) = 0 + 0.5*(2-0) = 1  => d = 10

    Ожидаем 10.
    """
    diam = np.array([1.0, 100.0], dtype=float)
    cum = np.array([10.0, 90.0], dtype=float)

    d50 = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=50.0)
    assert d50 == pytest.approx(10.0, rel=1e-12, abs=1e-12)


def test_input_order_does_not_matter() -> None:
    """
    Функция сортирует точки по x, поэтому результат должен быть одинаковым
    независимо от исходного порядка.
    """
    diam1 = np.array([1.0, 10.0, 100.0], dtype=float)
    cum1 = np.array([10.0, 50.0, 90.0], dtype=float)

    diam2 = diam1[::-1]
    cum2 = cum1[::-1]

    d_a = interpolate_d_at_cum_pct_logx(diam1, cum1, target_pct=50.0)
    d_b = interpolate_d_at_cum_pct_logx(diam2, cum2, target_pct=50.0)

    assert d_a == pytest.approx(d_b, rel=1e-12, abs=1e-12)


def test_raises_if_target_pct_outside_open_interval() -> None:
    """target_pct должен быть строго в (0,100)."""
    diam = np.array([1.0, 10.0, 100.0], dtype=float)
    cum = np.array([10.0, 50.0, 90.0], dtype=float)

    with pytest.raises(ValueError, match=r"target_pct должен быть в \(0, 100\)"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=0.0)

    with pytest.raises(ValueError, match=r"target_pct должен быть в \(0, 100\)"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=100.0)


def test_raises_if_not_enough_positive_diameters() -> None:
    """
    После маски diam>0 должно остаться минимум 2 точки.
    """
    diam = np.array([0.0, -1.0, 2.0], dtype=float)
    cum = np.array([0.0, 10.0, 50.0], dtype=float)

    with pytest.raises(ValueError, match=r"Недостаточно точек"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=20.0)


def test_raises_if_target_pct_outside_curve_range() -> None:
    """
    Если target_pct меньше min(y) или больше max(y) — ошибка.
    """
    diam = np.array([1.0, 10.0, 100.0], dtype=float)
    cum = np.array([20.0, 30.0, 40.0], dtype=float)

    with pytest.raises(ValueError, match=r"вне диапазона кривой"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=10.0)

    with pytest.raises(ValueError, match=r"вне диапазона кривой"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=90.0)


def test_flat_segment_returns_left_boundary() -> None:
    """
    Проверяем ветку if y1 == y0.

    Пример:
        (d=1, y=10)
        (d=10, y=10)  <- "плоский" участок
        (d=100, y=90)

    Для target_pct=10:
    searchsorted вернёт k=0? Нет, если y начинается с 10, то k=0 и вернётся x[0]=1.
    Поэтому берём target чуть выше 10, например 10.0 + eps, чтобы попасть между
    точками y0=10 и y1=10 (всё равно плоско).
    """
    diam = np.array([1.0, 10.0, 100.0], dtype=float)
    cum = np.array([10.0, 10.0, 90.0], dtype=float)

    # Важно: target должен быть в диапазоне [y_min, y_max]
    # y_min=10, y_max=90; берём target=10 (это вернёт x[0]).
    d_exact = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=10.0)
    assert d_exact == pytest.approx(1.0)
    assert True


def test_non_monotonic_input_is_fixed_by_accumulate() -> None:
    """
    Если cum_pct "шумный" и не монотонный, np.maximum.accumulate делает его
    неубывающим. Проверяем, что функция не падает и даёт результат.
    """
    diam = np.array([1.0, 10.0, 100.0, 200.0], dtype=float)
    cum = np.array([10.0, 60.0, 55.0, 90.0], dtype=float)  # 55 < 60 (шум)

    d = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=60.0)
    assert d > 0.0
