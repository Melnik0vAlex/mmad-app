# -*- coding: utf-8 -*-
# test/test_cumulative_undersize.py
"""
Тесты для `compute_cumulative_undersize`.

Проверяем свойства, которые должны выполняться для корректной кумулятивной кривой:

- Функция возвращает два массива одинаковой длины.
- Диаметры упорядочены по возрастанию, а cumulative undersize не убывает.
- Значения кумулятивы ограничены диапазоном 0..100, а последняя точка равна 100%.
- Результат не зависит от порядка входных записей.
- Дубликаты по d_low объединяются в один узел.
- Нулевая суммарная масса приводит к понятной ошибке.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import compute_cumulative_undersize


def _make_records_andersen() -> list[StageRecord]:
    """Возвращает небольшой тестовый набор ступеней Andersen (массы неотрицательные)."""
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
    Базовые свойства результата:
    - массивы согласованы по размеру
    - diam отсортирован по возрастанию
    - cum не убывает
    - cum лежит в 0..100
    - последняя точка кумулятивы равна 100
    """
    records = _make_records_andersen()
    diam, cum = compute_cumulative_undersize(records)

    assert isinstance(diam, np.ndarray)
    assert isinstance(cum, np.ndarray)
    assert diam.shape == cum.shape
    assert diam.size >= 2

    # Диаметры должны идти по возрастанию.
    assert np.all(np.diff(diam) > -1e-12)

    # Кумулятива должна быть неубывающей (допускаем микрошумы).
    assert np.all(np.diff(cum) >= -1e-9)

    # Границы по процентам.
    assert float(np.min(cum)) >= -1e-9
    assert float(np.max(cum)) <= 100.0 + 1e-9

    # Верхняя граничная точка.
    assert float(cum[-1]) == pytest.approx(100.0)


def test_order_invariance() -> None:
    """
    Порядок записей на входе не должен влиять на результат:
    функция сама сортирует точки по диаметру.
    """
    records = _make_records_andersen()

    diam1, cum1 = compute_cumulative_undersize(records)
    diam2, cum2 = compute_cumulative_undersize(list(reversed(records)))

    assert np.allclose(diam1, diam2)
    assert np.allclose(cum1, cum2)


def test_raises_if_total_mass_non_positive() -> None:
    """
    Если входные массы нулевые, после фильтрации (m<=0) данных не остаётся,
    поэтому ожидаем ошибку о недостаточности данных для кумулятивы.
    """
    records = [
        StageRecord("0", 9.0, 10.0, 0.0),
        StageRecord("1", 5.8, 9.0, 0.0),
        StageRecord("2", 4.7, 5.8, 0.0),
    ]

    with pytest.raises(ValueError, match=r"Недостаточно данных для кумулятивы"):
        _ = compute_cumulative_undersize(records)


def test_duplicate_diameters_are_merged() -> None:
    """
    Если два узла имеют одинаковый d_low, в результате должен остаться один диаметр.

    Для объединённого диаметра cumulative undersize выбирается максимальное значение,
    после чего кривая должна остаться неубывающей.
    """
    records = [
        StageRecord("A", 2.0, 3.0, 1.0),
        StageRecord("B", 2.0, 2.5, 2.0),  # дубликат d_low
        StageRecord("C", 1.0, 2.0, 3.0),
    ]

    diam, cum = compute_cumulative_undersize(records)

    # Диаметр 2.0 должен встретиться ровно один раз.
    assert int(np.sum(np.isclose(diam, 2.0))) == 1

    # Кумулятива должна оставаться неубывающей.
    assert np.all(np.diff(cum) >= -1e-9)
