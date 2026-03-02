# -*- coding: utf-8 -*-
# test/test_computate_mmad.py
"""
Тесты для `compute_mmad`.

Здесь мы проверяем поведение расчёта на небольших, полностью контролируемых наборах:

- «Здравый смысл» результата на типичном наборе ступеней (порядок квантилей,
  конечность значений, разумные диапазоны).
- Корректную обработку ошибок (например, нулевая суммарная масса).
- Интервальные метрики (log_mean / mass_mean / modal):
  * считаются и конечны, если интервалы заданы корректно,
  * становятся NaN, если интервалы не позволяют корректно вычислить эти величины.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmad_app.core.mmad import compute_mmad
from mmad_app.core.models import StageRecord


def _make_andersen_records() -> list[StageRecord]:
    """Возвращает компактный тестовый набор ступеней Andersen (массы > 0)."""
    return [
        StageRecord(name="0", d_low=9.0, d_high=10.0, mass=0.10),
        StageRecord(name="1", d_low=5.8, d_high=9.0, mass=0.15),
        StageRecord(name="2", d_low=4.7, d_high=5.8, mass=0.20),
        StageRecord(name="3", d_low=3.3, d_high=4.7, mass=0.25),
        StageRecord(name="4", d_low=2.1, d_high=3.3, mass=0.15),
        StageRecord(name="5", d_low=1.1, d_high=2.1, mass=0.10),
        StageRecord(name="6", d_low=0.65, d_high=1.1, mass=0.04),
        StageRecord(name="7", d_low=0.43, d_high=0.65, mass=0.01),
    ]


def test_compute_mmad_basic_sanity() -> None:
    """
    Базовая проверка корректности результата на «типичных» данных.

    Не тестируем точные числа (они зависят от деталей интерполяции/округления),
    а проверяем свойства, которые должны выполняться всегда.
    """
    records = _make_andersen_records()
    res = compute_mmad(records)

    # Суммарная масса и ключевые метрики должны быть положительными.
    assert res.total_mass > 0.0
    assert res.mmad > 0.0

    # Для реального аэрозоля ожидаем GSD > 1.
    assert res.gsd > 1.0

    # Квантили должны идти строго по возрастанию.
    assert res.d10 < res.d16 < res.mmad < res.d84 < res.d90

    # Span по определению не должен быть отрицательным.
    assert res.span >= 0.0

    # FPF — это доля, выраженная в процентах.
    assert 0.0 <= res.fpf_pct <= 100.0

    # Для построения графиков возвращаются согласованные массивы.
    assert isinstance(res.diam_um, np.ndarray)
    assert isinstance(res.cum_undersize_pct, np.ndarray)
    assert res.diam_um.shape == res.cum_undersize_pct.shape
    assert res.diam_um.size >= 2


def test_compute_mmad_interval_means_and_mode_are_finite() -> None:
    """
    Если у записей заданы корректные интервалы (d_low < d_high),
    интервальные метрики должны быть вычислены и быть конечными числами.
    """
    records = _make_andersen_records()
    res = compute_mmad(records)

    assert math.isfinite(res.log_mean)
    assert math.isfinite(res.mass_mean)
    assert math.isfinite(res.modal)

    assert res.log_mean > 0.0
    assert res.mass_mean > 0.0
    assert res.modal > 0.0


def test_compute_mmad_raises_if_total_mass_non_positive() -> None:
    """
    Нулевая (или отрицательная) суммарная масса — это некорректные входные данные.

    Важно: оставляем минимум 3 записи, чтобы ошибка была именно по массе, а не
    из-за валидации «слишком мало ступеней».
    """
    records = [
        StageRecord(name="0", d_low=9.0, d_high=10.0, mass=0.0),
        StageRecord(name="1", d_low=5.8, d_high=9.0, mass=0.0),
        StageRecord(name="2", d_low=4.7, d_high=5.8, mass=0.0),
    ]

    with pytest.raises(ValueError, match=r"Суммарная масса должна быть > 0"):
        _ = compute_mmad(records)


def test_compute_mmad_interval_metrics_nan_if_no_valid_bins() -> None:
    """
    Если интервальные метрики посчитать нельзя (нет валидных интервалов),
    функция должна вернуть NaN для log_mean / mass_mean / modal.

    Здесь у всех записей отсутствует d_low, поэтому ни один корректный интервал
    для расчёта репрезентативного диаметра не формируется.
    """
    records = [
        StageRecord(name="Ступень 1", d_low=None, d_high=1.0, mass=1.0),
        StageRecord(name="Ступень 2", d_low=None, d_high=3.0, mass=2.0),
        StageRecord(name="Ступень 3", d_low=None, d_high=10.0, mass=3.0),
    ]

    res = compute_mmad(records)

    assert math.isnan(res.log_mean)
    assert math.isnan(res.mass_mean)
    assert math.isnan(res.modal)
