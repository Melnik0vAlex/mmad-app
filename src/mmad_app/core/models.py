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
    d50_um:
        Cut-point (D50) в мкм для данной ступени при заданном расходе.
    mass_ug:
        Масса (мкг), осаждённая на данной ступени.
    """
    stage_name: str
    d50_um: float
    mass_ug: float


@dataclass(frozen=True)
class MmadResult:
    """Результат расчёта MMAD/GSD + дополнительные метрики APSD."""

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
