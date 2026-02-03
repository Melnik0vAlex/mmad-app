# -*- coding: utf-8 -*-
"""
Тесты для normal.py (normal_cdf и normal_ppf) без SciPy.

Проверяем:
1) Базовые значения:
   - Φ(0) = 0.5
   - Φ(-z) = 1 - Φ(z)
   - Φ^{-1}(0.5) = 0
2) Композиции:
   - Φ(Φ^{-1}(p)) ≈ p
   - Φ^{-1}(Φ(z)) ≈ z (в разумном диапазоне)
3) Монотонность:
   - Φ(z) возрастает
   - Φ^{-1}(p) возрастает
4) Обработка краёв p=0, p=1 через eps-клиппинг.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmad_app.core.normal import normal_cdf, normal_ppf


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (2.0, 0.9772498680518208),
        (-2.0, 0.02275013194817921),
    ],
)
def test_normal_cdf_known_values(z: float, expected: float) -> None:
    """
    Проверяем значения Φ(z) в стандартных точках.

    Здесь используются общеизвестные табличные значения стандартного нормального
    распределения. Точность в пределах ~1e-7 достаточно для аппроксимаций.
    """
    out = float(normal_cdf([z])[0])
    assert out == pytest.approx(expected, rel=0.0, abs=1e-7)


def test_normal_cdf_symmetry() -> None:
    """
    Проверяем симметрию:
        Φ(-z) = 1 - Φ(z)
    """
    z = np.array([0.2, 0.7, 1.3, 2.5], dtype=float)
    phi_z = normal_cdf(z)
    phi_mz = normal_cdf(-z)
    assert np.allclose(phi_mz, 1.0 - phi_z, atol=1e-12, rtol=0.0)


def test_normal_ppf_basic() -> None:
    """
    Проверяем базовое:
        Φ^{-1}(0.5) = 0
    """
    z = float(normal_ppf([0.5])[0])
    assert z == pytest.approx(0.0, abs=1e-12)


def test_normal_ppf_known_values() -> None:
    """
    Проверяем несколько известных квантилей:
        Φ^{-1}(0.158655...) ≈ -1
        Φ^{-1}(0.841344...) ≈  1
    """
    p = np.array([0.15865525393145707, 0.8413447460685429], dtype=float)
    z = normal_ppf(p)
    assert float(z[0]) == pytest.approx(-1.0, abs=1e-6)
    assert float(z[1]) == pytest.approx(1.0, abs=1e-6)


def test_composition_cdf_ppf() -> None:
    """
    Проверяем композицию:
        Φ(Φ^{-1}(p)) ≈ p
    (для p не слишком близких к 0 и 1)
    """
    p = np.array([0.01, 0.1, 0.25, 0.5, 0.9, 0.99], dtype=float)
    z = normal_ppf(p)
    p_back = normal_cdf(z)
    assert np.allclose(p_back, p, atol=1e-6, rtol=0.0)


def test_composition_ppf_cdf() -> None:
    """
    Проверяем композицию:
        Φ^{-1}(Φ(z)) ≈ z
    (для z в разумном диапазоне, где ошибки аппроксимации малы)
    """
    z = np.array([-3.0, -1.5, -0.2, 0.0, 0.4, 1.2, 2.5], dtype=float)
    p = normal_cdf(z)
    z_back = normal_ppf(p)
    assert np.allclose(z_back, z, atol=1e-6, rtol=0.0)


def test_monotonicity_cdf() -> None:
    """Φ(z) должна быть монотонно возрастающей."""
    z = np.linspace(-4.0, 4.0, 101)
    p = normal_cdf(z)
    assert np.all(np.diff(p) >= 0.0)


def test_monotonicity_ppf() -> None:
    """Φ^{-1}(p) должна быть монотонно возрастающей."""
    p = np.linspace(0.001, 0.999, 101)
    z = normal_ppf(p)
    assert np.all(np.diff(z) >= 0.0)


def test_ppf_clip_eps_at_edges() -> None:
    """
    Проверяем, что p=0 и p=1 не дают inf, а клипуются к eps и 1-eps.

    Ожидаем:
        Φ^{-1}(0) -> конечное большое отрицательное число
        Φ^{-1}(1) -> конечное большое положительное число
    """
    z = normal_ppf(np.array([0.0, 1.0], dtype=float), eps=1e-12)

    assert math.isfinite(float(z[0]))
    assert math.isfinite(float(z[1]))
    assert float(z[0]) < 0.0
    assert float(z[1]) > 0.0
    assert float(z[1]) > abs(float(z[0])) * 0.9  # примерно симметрия
