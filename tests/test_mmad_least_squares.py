# -*- coding: utf-8 -*-
# tests/test_mmad_least_squares.py
"""
Тесты для `compute_mmad_least_squares`.

Здесь проверяется второй метод расчёта MMAD — через линейную регрессию в координатах
ln(d) vs arcerf(2p-1) (эквивалент Excel LINEST/ЛИНЕЙН с включённой статистикой).

Покрываем:
- восстановление параметров регрессии на синтетике без шума (идеальная линейность);
- корректное исключение при недостатке валидных интервалов (mask_intervals < 3);
- корректное исключение, если после отбраковки p=0 и p=1 остаётся слишком мало точек.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmad_app.core.mmad import compute_mmad_least_squares
from mmad_app.core.models import StageRecord
from mmad_app.core.normal import normal_ppf


def _make_andersen_11_records(
    *,
    d_low: list[float],
    d_high: list[float],
    masses_stage_7_to_0: list[float],
    mass_filter: float = 0.05,
    mass_preseparator: float = 0.05,
    mass_inlet: float = 0.0,
) -> list[StageRecord]:
    """
    Сборка фикстуры из 11 записей в том порядке, который ожидает
    `compute_mmad_least_squares`.

    Порядок:
      0) Фильтр            (d_low=None,  d_high=0.43)
      1..8) Ступени 7..0   (все с конечными d_low/d_high)
      9) Пресепаратор      (d_low=10.0,  d_high=None)
      10) Входной канал    (d_low=None,  d_high=None)

    Параметры `d_low`, `d_high`, `masses_stage_7_to_0` должны быть длины 8 и
    соответствовать ступеням 7..0 (в этом же порядке).
    """
    if len(d_low) != 8 or len(d_high) != 8 or len(masses_stage_7_to_0) != 8:
        raise ValueError("Ожидаются массивы длины 8 для ступеней 7..0.")

    records: list[StageRecord] = [
        StageRecord(name="Фильтр", d_low=None, d_high=0.43, mass=float(mass_filter)),
    ]

    # Ступени 7..0
    for i in range(8):
        stage_num = 7 - i
        records.append(
            StageRecord(
                name=f"Ступень {stage_num}",
                d_low=float(d_low[i]),
                d_high=float(d_high[i]),
                mass=float(masses_stage_7_to_0[i]),
            )
        )

    # Пресепаратор и входной канал
    records.append(
        StageRecord(
            name="Пресепаратор",
            d_low=10.0,
            d_high=None,
            mass=float(mass_preseparator),
        )
    )
    records.append(
        StageRecord(
            name="Входной канал",
            d_low=None,
            d_high=None,
            mass=float(mass_inlet),
        )
    )

    return records


def test_compute_mmad_least_squares_recovers_parameters_on_ideal_data() -> None:
    """
    Идеальный случай без шума: зависимость ln(d) от arcerf(2p-1) строго линейна.

    Логика синтетики:
    1) задаём массы -> получаем p так же, как это делает функция;
    2) считаем x = normal_ppf(p) / sqrt(2);
    3) задаём истинные slope/intercept и строим y = slope*x + intercept;
    4) подбираем интервалы d_low/d_high так, чтобы геометрический центр
       sqrt(d_low*d_high) совпадал с exp(y).
    """
    slope_true = 1.35
    intercept_true = 0.25  # mu = ln(D50)

    # Массы: все положительные, чтобы p строго попадали в (0, 1)
    masses_stages = [0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.10, 0.12]
    mass_filter = 0.05
    mass_pre = 0.05
    mass_inlet = 0.00

    mass_vector = np.asarray(
        [mass_filter, *masses_stages, mass_pre, mass_inlet], dtype=float
    )
    p_all = np.cumsum(mass_vector) / float(np.sum(mass_vector))

    # В compute_mmad_least_squares используется p = p[1:9] (ступени 7..0)
    p = p_all[1:9]
    assert np.all((p > 0.0) & (p < 1.0))

    x_erfinv = normal_ppf(p) / math.sqrt(2.0)

    # Идеальная линейная зависимость
    y_ln_d = slope_true * x_erfinv + intercept_true
    d_g = np.exp(y_ln_d)

    # Интервалы делаем симметрично вокруг d_g: sqrt(d_low*d_high) = d_g
    f = 1.15
    d_low = (d_g / f).tolist()
    d_high = (d_g * f).tolist()

    records = _make_andersen_11_records(
        d_low=d_low,
        d_high=d_high,
        masses_stage_7_to_0=masses_stages,
        mass_filter=mass_filter,
        mass_preseparator=mass_pre,
        mass_inlet=mass_inlet,
    )

    res = compute_mmad_least_squares(records)

    # Параметры регрессии
    assert res.slope == pytest.approx(slope_true, abs=1e-10)
    assert res.intercept == pytest.approx(intercept_true, abs=1e-10)

    # Производные величины
    d50_true = float(np.exp(intercept_true))
    sigma_true = float(slope_true / math.sqrt(2.0))
    kor_k_true = float(1.0 / slope_true)

    assert res.mmad == pytest.approx(d50_true, rel=0.0, abs=1e-10)
    assert res.sigma == pytest.approx(sigma_true, rel=0.0, abs=1e-10)
    assert res.kor_k == pytest.approx(kor_k_true, rel=0.0, abs=1e-10)

    # Для идеальной линейности ожидаем R²≈1 и r≈1
    assert res.r2 == pytest.approx(1.0, abs=1e-12)
    assert res.r == pytest.approx(1.0, abs=1e-12)

    # Точки для графика должны вернуться той же длины, что и p после фильтрации
    assert res.x_erfinv.shape == p.shape
    assert res.y_ln_d.shape == p.shape


def test_compute_mmad_least_squares_raises_if_not_enough_valid_intervals() -> None:
    """
    Если корректных интервалов (по mask_intervals) меньше трёх, регрессию со
    статистикой оценивать нельзя — ожидается ValueError.
    """
    # Делаем только 2 корректных интервала, остальные портим через None
    d_low = [0.43, 0.65, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    d_high = [0.65, 1.10, 0.9, None, None, None, None, None]  # type: ignore[list-item]
    masses = [0.1] * 8

    records: list[StageRecord] = [
        StageRecord("Фильтр", None, 0.43, 0.05),
        StageRecord("Ступень 7", d_low[0], d_high[0], masses[0]),
        StageRecord("Ступень 6", d_low[1], d_high[1], masses[1]),
        StageRecord(
            "Ступень 5", d_low[2], d_high[2], masses[2]
        ),  # некорректный: d_high < d_low
        StageRecord("Ступень 4", d_low[3], d_high[3], masses[3]),
        StageRecord("Ступень 3", d_low[4], d_high[4], masses[4]),
        StageRecord("Ступень 2", d_low[5], d_high[5], masses[5]),
        StageRecord("Ступень 1", d_low[6], d_high[6], masses[6]),
        StageRecord("Ступень 0", d_low[7], d_high[7], masses[7]),
        StageRecord("Пресепаратор", 10.0, None, 0.05),
        StageRecord("Входной канал", None, None, 0.0),
    ]

    with pytest.raises(ValueError, match="Недостаточно корректных интервалов"):
        _ = compute_mmad_least_squares(records)


def test_compute_mmad_least_squares_raises_if_too_few_points_after_excluding_p_0_1() -> (
    None
):
    """
    Если после исключения точек с p=0 и p=1 остаётся меньше трёх значений,
    регрессия не должна выполняться — ожидается ValueError.

    Конструкция:
    - фильтр = 0
    - ступени 7..1 = 0
    - ступень 0 = 1
    - пресепаратор = 0
    - входной канал = 0

    Тогда в p[1:9] получаем [0, 0, 0, 0, 0, 0, 0, 1] и после маски остаётся 0 точек.
    """
    masses_stages = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    mass_filter = 0.0
    mass_pre = 0.0
    mass_inlet = 0.0

    # Интервалы делаем корректными, чтобы ошибка была именно на этапе p
    d_low = [0.43, 0.65, 1.1, 2.1, 3.3, 4.7, 5.8, 9.0]
    d_high = [0.65, 1.1, 2.1, 3.3, 4.7, 5.8, 9.0, 10.0]

    records = _make_andersen_11_records(
        d_low=d_low,
        d_high=d_high,
        masses_stage_7_to_0=masses_stages,
        mass_filter=mass_filter,
        mass_preseparator=mass_pre,
        mass_inlet=mass_inlet,
    )

    with pytest.raises(
        ValueError, match="После исключения p=0 и p=1 осталось < 3 точек"
    ):
        _ = compute_mmad_least_squares(records)
