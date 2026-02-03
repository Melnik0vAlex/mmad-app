# -*- coding: utf-8 -*-
"""
Тесты для probit-аналитики:
- fit_probit
- predict_cumulative_from_probit

Проверяем:
1) Фильтрацию точек (d<=0, NaN/inf, cum_pct=0/100)
2) Восстановление параметров a,b на синтетических данных
3) RMSE на идеальных данных ~ 0
4) Предсказание cumulative (%) по модели
5) Поведение на невалидных диаметрах при predict (NaN)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmad_app.core.models import ProbitLine
from mmad_app.core.normal import normal_cdf
from mmad_app.core.probit import fit_probit, predict_cumulative_from_probit


def _make_synthetic_probit_data(
    *,
    a: float,
    b: float,
    diam_um: np.ndarray,
) -> np.ndarray:
    """
    Генерирует идеальные кумулятивы cum_pct (%) для заданных параметров
    probit = a*log10(d) + b.

    Схема:
        probit = z + 5
        z = probit - 5
        p = Φ(z)
        cum_pct = 100*p
    """
    lx = np.log10(np.asarray(diam_um, dtype=float))
    probit = a * lx + b
    z = probit - 5.0
    p = normal_cdf(z)
    return 100.0 * p


def test_fit_probit_returns_nan_if_less_than_2_points_after_filter() -> None:
    """
    Если после фильтрации остаётся < 2 точек,
    fit_probit должен вернуть NaN-параметры.
    """
    diam = np.array([1.0, 2.0, np.nan], dtype=float)

    # Все точки будут отброшены, потому что 0% и 100% исключаются строго
    cum = np.array([0.0, 100.0, 50.0], dtype=float)

    fit = fit_probit(diam, cum)

    assert math.isnan(fit.a)
    assert math.isnan(fit.b)
    assert math.isnan(fit.rmse)


def test_fit_probit_filters_0_and_100_percent() -> None:
    """
    Проверяем, что точки с cum_pct = 0 и 100 не участвуют в фиттинге.

    Строим синтетику на 3 точках, но добавляем "краевые" 0 и 100:
    если бы они учитывались, параметры могли бы исказиться из-за клиппинга.
    """
    a_true = 2.0
    b_true = 5.0

    diam = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)
    cum = _make_synthetic_probit_data(a=a_true, b=b_true, diam_um=diam)

    # Принудительно делаем две крайние точки ровно 0 и 100
    cum[0] = 0.0
    cum[-1] = 100.0

    fit = fit_probit(diam, cum)

    assert math.isfinite(fit.a)
    assert math.isfinite(fit.b)
    assert math.isfinite(fit.rmse)

    # Ожидаем, что параметры восстановятся примерно по двум оставшимся точкам
    assert fit.a == pytest.approx(a_true, abs=1e-6)
    assert fit.b == pytest.approx(b_true, abs=1e-6)


def test_fit_probit_recovers_parameters_on_ideal_data() -> None:
    """
    На идеальных данных probit-линейная регрессия должна восстановить a,b,
    а RMSE быть близким к 0.
    """
    a_true = 1.7
    b_true = 4.3

    diam = np.array([0.43, 0.65, 1.1, 2.1, 3.3, 4.7, 5.8, 9.0], dtype=float)
    cum = _make_synthetic_probit_data(a=a_true, b=b_true, diam_um=diam)

    # Важно: исключаем ровно 0 и 100 (на синтетике их почти не будет,
    # но на всякий случай "подрежем" к допустимому диапазону)
    cum = np.clip(cum, 1e-6, 100.0 - 1e-6)

    fit = fit_probit(diam, cum)

    assert fit.a == pytest.approx(a_true, abs=1e-6)
    assert fit.b == pytest.approx(b_true, abs=1e-6)
    assert fit.rmse == pytest.approx(0.0, abs=1e-8)


def test_predict_cumulative_from_probit_matches_synthetic_curve() -> None:
    """
    Проверяем, что predict_cumulative_from_probit воспроизводит кумулятиву,
    если fit соответствует истинным параметрам.
    """
    fit = ProbitLine(a=2.2, b=4.8, rmse=0.0)

    diam = np.array([0.5, 1.0, 2.0, 4.0], dtype=float)

    expected = _make_synthetic_probit_data(a=fit.a, b=fit.b, diam_um=diam)
    predicted = predict_cumulative_from_probit(fit, diam)

    assert np.allclose(predicted, expected, atol=1e-10, rtol=0.0)


def test_predict_cumulative_returns_nan_for_nonpositive_or_nonfinite_diameters() -> (
    None
):
    """
    Для d<=0 и нечисловых значений предсказание должно быть NaN,
    а для валидных — конечным числом в [0,100].
    """
    fit = ProbitLine(a=2.0, b=5.0, rmse=0.0)

    diam = np.array([np.nan, -1.0, 0.0, 1.0, 10.0, np.inf], dtype=float)
    out = predict_cumulative_from_probit(fit, diam)

    assert math.isnan(float(out[0]))
    assert math.isnan(float(out[1]))
    assert math.isnan(float(out[2]))
    assert math.isfinite(float(out[3]))
    assert math.isfinite(float(out[4]))
    assert math.isnan(float(out[5]))

    assert 0.0 <= float(out[3]) <= 100.0
    assert 0.0 <= float(out[4]) <= 100.0


def test_fit_and_predict_end_to_end_roundtrip() -> None:
    """
    End-to-end тест:
    1) генерируем cum_pct по истинной модели
    2) фиттим a,b
    3) предсказываем cum_pct обратно
    """
    a_true = 1.3
    b_true = 5.4

    diam = np.array([0.43, 0.65, 1.1, 2.1, 3.3, 4.7, 5.8, 9.0], dtype=float)
    cum = _make_synthetic_probit_data(a=a_true, b=b_true, diam_um=diam)

    # Добавим несколько "плохих" точек, которые должны быть отфильтрованы
    diam_bad = np.array([0.0, -1.0, np.nan], dtype=float)
    cum_bad = np.array([50.0, 50.0, 50.0], dtype=float)

    diam_all = np.concatenate([diam, diam_bad])
    cum_all = np.concatenate([cum, cum_bad])

    # И гарантируем отсутствие ровно 0/100 в "хороших" точках
    cum_all = np.clip(cum_all, 1e-6, 100.0 - 1e-6)

    fit = fit_probit(diam_all, cum_all)

    assert fit.a == pytest.approx(a_true, abs=1e-6)
    assert fit.b == pytest.approx(b_true, abs=1e-6)
    assert fit.rmse == pytest.approx(0.0, abs=1e-8)

    predicted = predict_cumulative_from_probit(fit, diam)
    assert np.allclose(predicted, cum, atol=1e-6, rtol=0.0)
