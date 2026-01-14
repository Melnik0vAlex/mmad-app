# -*- coding: utf-8 -*-
"""
Модели данных для каскадного импактора.

"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StageRecord:
    """
    Запись по одной ступени импактора.

    Parameters
    ----------
    stage_name:
        Название ступени (например, Stage 0..7, Filter).
    d_high:
        Верхний размер частиц (мкм).
    d_low:
        Нижний размер частиц - Cut-point (мкм) для данной ступени при заданном расходе.
    mass_ug:
        Масса (мкг), осаждённая на данной ступени.
    """
    name: str
    d_high: float
    d_low: float
    mass: float


@dataclass(frozen=True)
class MmadResult:
    """Результат расчёта MMAD и дополнительные метрики APSD."""

    mmad_um: float
    gsd: float
    d15_87_um: float
    d84_13_um: float

    d10_um: float
    d90_um: float
    span: float

    fpf_cutoff_um: float
    fpf_pct: float

    total_mass_ug: float

    diam_um: np.ndarray
    cum_undersize_pct: np.ndarray

@dataclass(frozen=True)
class ProbitFitResult:
    """Параметры пробит-регрессии."""

    a: float
    b: float
    r2: float
