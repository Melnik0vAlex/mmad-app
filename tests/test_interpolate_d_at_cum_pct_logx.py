# -*- coding: utf-8 -*-
# test/test_interpolate_d_at_cum_pct_logx.py
"""
Тесты для `interpolate_d_at_cum_pct_logx`.

Цель — проверить, что интерполяция по кумулятивной кривой работает устойчиво и
предсказуемо в типичных и граничных ситуациях.

Покрываем следующие сценарии:
- лог-интерполяция на простом примере с аналитически известным ответом;
- независимость результата от порядка входных точек;
- корректные исключения при неверных параметрах и недостатке данных;
- поведение на «плоском» участке (y1 == y0);
- игнорирование неположительных диаметров (diam <= 0);
- устойчивость к немонотонному шуму кумулятивы (исправление через accumulate).
"""

from __future__ import annotations

import numpy as np
import pytest

from mmad_app.core.mmad import interpolate_d_at_cum_pct_logx


def test_simple_log_interpolation_midpoint() -> None:
    """
    Проверка логарифмической интерполяции на одном интервале.

    Между точками:
        (d=1,   y=10)
        (d=100, y=90)
    при target_pct=50 получаем t=0.5 и log10(d)=1, то есть d=10.
    """
    diam = np.array([1.0, 100.0], dtype=float)
    cum = np.array([10.0, 90.0], dtype=float)

    d50 = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=50.0)
    assert d50 == pytest.approx(10.0, rel=1e-12, abs=1e-12)


def test_input_order_does_not_matter() -> None:
    """
    Функция сортирует точки по диаметру, поэтому исходный порядок не должен
    влиять на результат.
    """
    diam1 = np.array([1.0, 10.0, 100.0], dtype=float)
    cum1 = np.array([10.0, 50.0, 90.0], dtype=float)

    diam2 = diam1[::-1]
    cum2 = cum1[::-1]

    d_a = interpolate_d_at_cum_pct_logx(diam1, cum1, target_pct=50.0)
    d_b = interpolate_d_at_cum_pct_logx(diam2, cum2, target_pct=50.0)

    assert d_a == pytest.approx(d_b, rel=1e-12, abs=1e-12)


def test_raises_if_target_pct_outside_open_interval() -> None:
    """
    target_pct задаётся в процентах и по контракту должен лежать строго в (0, 100).
    """
    diam = np.array([1.0, 10.0, 100.0], dtype=float)
    cum = np.array([10.0, 50.0, 90.0], dtype=float)

    with pytest.raises(ValueError, match=r"target_pct должен быть в \(0, 100\)"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=0.0)

    with pytest.raises(ValueError, match=r"target_pct должен быть в \(0, 100\)"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=100.0)


def test_raises_if_not_enough_positive_diameters() -> None:
    """
    Если после фильтра diam > 0 остаётся меньше двух точек, интерполяция невозможна.
    """
    diam = np.array([0.0, -1.0, 2.0], dtype=float)
    cum = np.array([0.0, 10.0, 50.0], dtype=float)

    with pytest.raises(ValueError, match=r"Недостаточно точек"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=20.0)


def test_raises_if_target_pct_outside_curve_range() -> None:
    """
    target_pct должен попадать в диапазон значений кумулятивной кривой.
    Иначе функция обязана сообщить об ошибке.
    """
    diam = np.array([1.0, 10.0, 100.0], dtype=float)
    cum = np.array([20.0, 30.0, 40.0], dtype=float)

    with pytest.raises(ValueError, match=r"вне диапазона кривой"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=10.0)

    with pytest.raises(ValueError, match=r"вне диапазона кривой"):
        _ = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=90.0)


def test_flat_segment_returns_left_boundary() -> None:
    """
    Ветвь y1 == y0: если кумулятива на интервале по диаметру не меняется,
    возвращаем левую границу (x0), так как «куда двигаться по Y» не определено.

    Здесь target_pct совпадает с первым значением y=10, поэтому попадём в k==0
    и получим x[0] (это тоже корректное граничное поведение).
    """
    diam = np.array([1.0, 10.0, 100.0], dtype=float)
    cum = np.array([10.0, 10.0, 90.0], dtype=float)

    d_exact = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=10.0)
    assert d_exact == pytest.approx(1.0)


def test_non_monotonic_input_is_fixed_by_accumulate() -> None:
    """
    Кумулятива в исходных данных может быть «шумной» и локально убывать.
    Внутри функции применяется np.maximum.accumulate, поэтому расчёт должен
    оставаться устойчивым и не приводить к исключению.
    """
    diam = np.array([1.0, 10.0, 100.0, 200.0], dtype=float)
    cum = np.array([10.0, 60.0, 55.0, 90.0], dtype=float)  # 55 < 60 (шум)

    d = interpolate_d_at_cum_pct_logx(diam, cum, target_pct=60.0)
    assert d > 0.0
