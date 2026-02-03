# -*- coding: utf-8 -*-
"""
Тесты для compute_cumulative_undersize.

Проверяем:
1) Возврат массивов одинаковой длины и корректные значения.
2) Кумулятива не убывает после сортировки по диаметру.
3) Границы: 0..100 и последняя точка = 100%.
4) Инвариант к порядку входных records.
5) Обработка дублей d_low.
6) Ошибка при total_mass <= 0.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import compute_cumulative_undersize


def _make_records_andersen() -> list[StageRecord]:
    """
    Тестовый набор для импактора Андерсона.
    """
    return [
        StageRecord("0", 9.0, 10.0, 0.0034),
        StageRecord("1", 5.8, 9.0, 0.0115),
        StageRecord("2", 4.7, 5.8, 0.0233),
        StageRecord("3", 3.3, 4.7, 0.0582),
        StageRecord("4", 2.1, 3.3, 0.0408),
        StageRecord("5", 1.1, 2.1, 0.0140),
        StageRecord("6", 0.65, 1.1, 0.0015),
        StageRecord("7", 0.43, 0.65, 0.0),
    ]


def test_shapes_monotonic_and_bounds() -> None:
    """
    Базовый тест:
    - длины равны
    - diam по возрастанию
    - cum не убывает
    - cum в [0, 100]
    - последняя точка cum = 100
    """
    records = _make_records_andersen()
    diam, cum = compute_cumulative_undersize(records)

    assert isinstance(diam, np.ndarray)
    assert isinstance(cum, np.ndarray)
    assert diam.shape == cum.shape
    assert diam.size >= 2

    # Диаметры должны быть по возрастанию
    assert np.all(np.diff(diam) > -1e-12)

    # Кумулятива (undersize) должна быть неубывающей
    assert np.all(np.diff(cum) >= -1e-9)

    # Границы
    assert float(np.min(cum)) >= -1e-9
    assert float(np.max(cum)) <= 100.0 + 1e-9

    # Последняя точка должна быть 100%
    assert float(cum[-1]) == pytest.approx(100.0)


def test_order_invariance() -> None:
    """
    Результат не должен зависеть от порядка входных records.
    """
    records = _make_records_andersen()

    diam1, cum1 = compute_cumulative_undersize(records)
    diam2, cum2 = compute_cumulative_undersize(list(reversed(records)))

    assert np.allclose(diam1, diam2)
    assert np.allclose(cum1, cum2)


def test_raises_if_total_mass_non_positive() -> None:
    """
    Если суммарная масса <= 0, функция должна упасть.
    Важно: нужно минимум 3 записи, иначе упадём на _validate_records().
    """
    records = [
        StageRecord("0", 9.0, 10.0, 0.0),
        StageRecord("1", 5.8, 9.0, 0.0),
        StageRecord("2", 4.7, 5.8, 0.0),
    ]

    with pytest.raises(ValueError, match=r"Суммарная масса должна быть > 0"):
        _ = compute_cumulative_undersize(records)


def test_duplicate_diameters_are_merged() -> None:
    """
    Если два интервала имеют одинаковый d_low, должен остаться один диаметр.
    Кумулятив для дубля должен стать max из двух значений.
    """
    records = [
        StageRecord("A", 2.0, 3.0, 1.0),
        StageRecord("B", 2.0, 2.5, 2.0),  # дубликат d_low
        StageRecord("C", 1.0, 2.0, 3.0),
    ]

    diam, cum = compute_cumulative_undersize(records)

    # Проверим, что диаметр 2.0 встречается ровно один раз
    assert int(np.sum(np.isclose(diam, 2.0))) == 1

    # Кумулятива всё равно должна быть неубывающей
    assert np.all(np.diff(cum) >= -1e-9)
