# -*- coding: utf-8 -*-
"""
Тесты для ядра расчёта MMAD.

Запуск:
    pytest
"""

from __future__ import annotations

import numpy as np
import pytest

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import compute_cumulative_undersize, compute_mmad


def test_compute_mmad_known_case() -> None:
    """
    Контрольный пример:

    D50 и массы подобраны так, что 50% находится между точками:
    (d=2 µm, cum=40%) и (d=3 µm, cum=65%),
    а интерполяция идёт в log10(d).
    Ожидаем MMAD ≈ 2.352158 µm.
    """
    records = [
        # Запишем намеренно "вперемешку" — функция должна отсортировать сама
        StageRecord("Stage 3", 2.0, 25.0),
        StageRecord("Stage 2", 3.0, 20.0),
        StageRecord("Stage 1", 5.0, 10.0),
        StageRecord("Stage 0", 9.0, 5.0),
        StageRecord("Stage 4", 1.0, 20.0),
        StageRecord("Stage 5", 0.5, 10.0),
        StageRecord("Stage 6", 0.3, 5.0),
        StageRecord("Stage 7", 0.2, 3.0),
        StageRecord("Filter", 0.0, 2.0),
    ]

    result = compute_mmad(records)
    assert np.isfinite(result.mmad_um)
    assert abs(result.mmad_um - 2.352158045049347) < 1e-6


def test_validation_negative_mass() -> None:
    records = [
        StageRecord("Stage 0", 9.0, 1.0),
        StageRecord("Stage 1", 5.0, -1.0),
        StageRecord("Filter", 0.0, 1.0),
    ]
    with pytest.raises(ValueError):
        compute_mmad(records)


def test_compute_cumulative_returns_monotonic() -> None:
    """
    Кумулятивная кривая должна быть неубывающей.
    """
    records = [
        StageRecord("Stage 0", 9.0, 1.0),
        StageRecord("Stage 1", 5.0, 1.0),
        StageRecord("Stage 2", 3.0, 1.0),
        StageRecord("Filter", 0.0, 1.0),
    ]
    diam_um, cum_pct = compute_cumulative_undersize(records)
    diffs = np.diff(cum_pct)
    assert np.all(diffs >= -1e-12)
