# -*- coding: utf-8 -*-
"""
Тесты для compute_mmad.

Проверяем:
1) Базовую корректность результата на данных Andersen:
   - значения конечны и > 0 где должны быть
   - квантили возрастают
   - gsd > 1
   - fpf_pct в диапазоне 0..100
2) Ошибки:
   - суммарная масса <= 0
3) Интервальные метрики:
   - log_mean, mass_mean, modal конечны и > 0 при наличии корректных интервалов.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmad_app.core.mmad import compute_mmad
from mmad_app.core.models import StageRecord


def _make_andersen_records() -> list[StageRecord]:
    """Тестовый набор Andersen (массы положительные)."""
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
    Интеграционный sanity-тест:
    - метрики конечны
    - физически разумные отношения квантилей
    """
    records = _make_andersen_records()
    res = compute_mmad(records)

    assert res.total_mass > 0.0
    assert res.mmad > 0.0
    assert res.gsd > 1.0

    # Возрастание квантилей
    assert res.d10 < res.d16 < res.mmad < res.d84 < res.d90

    # Span неотрицателен
    assert res.span >= 0.0

    # FPF в процентах
    assert 0.0 <= res.fpf_pct <= 100.0

    # Кумулятива и diam возвращаются
    assert isinstance(res.diam_um, np.ndarray)
    assert isinstance(res.cum_undersize_pct, np.ndarray)
    assert res.diam_um.shape == res.cum_undersize_pct.shape
    assert res.diam_um.size >= 2


def test_compute_mmad_interval_means_and_mode_are_finite() -> None:
    """
    Проверяем, что интервальные метрики считаются и конечны.
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
    Если суммарная масса <= 0 -> ошибка.
    Важно: нужно минимум 3 ступени, иначе ошибка на _validate_records.
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
    Если интервальные метрики невозможно посчитать (например, все массы 0 кроме одной,
    или интервалы некорректны), то log_mean/mass_mean/modal должны остаться NaN.

    Здесь делаем: масса есть, но интервалы сломаны (d_high <= d_low) -> bins пустые.
    """
    records = [
        StageRecord(name="0", d_low=1.0, d_high=1.0, mass=1.0),  # некорректный интервал
        StageRecord(name="1", d_low=2.0, d_high=2.0, mass=1.0),  # некорректный интервал
        StageRecord(name="2", d_low=3.0, d_high=3.0, mass=1.0),  # некорректный интервал
    ]

    res = compute_mmad(records)

    assert math.isnan(res.log_mean)
    assert math.isnan(res.mass_mean)
    assert math.isnan(res.modal)
