# -*- coding: utf-8 -*-
# src/mmad_app/core/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class StageRecord:
    """
    Запись по одной ступени импактора.

    Атрибуты
    --------
    name:
        Название ступени (например, Stage 0..7, Filter).
    d_high:
        Верхний размер частиц (мкм).
    d_low:
        Нижний размер частиц - Cut-point для данной ступени при заданном расходе (мкм).
    mass_ug:
        Масса, осаждённая на данной ступени (мкг).
    """

    name: str
    d_low: Optional[float]
    d_high: Optional[float]
    mass: float


@dataclass(frozen=True)
class MmadResult:
    """
    Результат расчёта MMAD и дополнительные метрики APSD.

    Все диаметры выражены в микрометрах (мкм), массы — в микрограммах (мкг),
    если не указано иначе.

    Атрибуты
    --------
    mmad:
        Mass Median Aerodynamic Diameter (MMAD), т.е. d50 по массе.
    gsd:
        Геометрическое стандартное отклонение (GSD), обычно оценивается как:
            GSD = sqrt(d84.13 / d15.87).
    d10:
        Диаметр d10 (10-й процентиль по массе) из накопительной кривой.
    d16:
        Диаметр d16 (16-й процентиль по массе) из накопительной кривой.
    d84:
        Диаметр d84 (84-й процентиль по массе) из накопительной кривой.
    d90:
        Диаметр d90 (90-й процентиль по массе) из накопительной кривой.
    d15_87:
        Диаметр d15.87 (квантиль -1σ для нормального распределения в пробит-подходе).
        Используется для расчёта GSD.
    d84_13:
        Диаметр d84.13 (квантиль +1σ для нормального распределения в пробит-подходе).
        Используется для расчёта GSD.
    span:
        Показатель ширины распределения:
            Span = (d90 - d10) / d50.
    fpf_cutoff_um:
        Пороговый диаметр для Fine Particle Fraction (FPF), например 5 мкм.
    fpf_pct:
        Fine Particle Fraction (FPF), доля массы частиц с диаметром < fpf_cutoff_um, %.
    total_mass:
        Суммарная масса аэрозоля, учтённая в расчёте (сумма масс по ступеням), мкг.
    log_mean:
        Логарифмический средний диаметр (геометрическое среднее по массе):
            d_g = exp(sum(w_i * ln(d_i))),
        где w_i — массовые доли, d_i — репрезентативные диаметры интервалов.
    mass_mean:
        Среднемассовый диаметр (массо-взвешенное арифметическое среднее):
            d_m = sum(w_i * d_i).
    modal:
        Модальный аэродинамический диаметр (оценка моды по интервалам):
        репрезентативный диаметр интервала с максимальной долей массы.
    diam:
        Массив диаметров (мкм) для построения накопительной кривой.
    cum_undersize_pct:
        Массив cumulative undersize (%): доля массы частиц с диаметром < d.
    """

    mmad: float
    gsd: float
    d10: float
    d16: float
    d84: float
    d90: float
    d15_87: float
    d84_13: float
    span: float
    fpf_cutoff_um: float
    fpf_pct: float
    total_mass: float
    log_mean: float
    mass_mean: float
    modal: float
    diam_um: np.ndarray
    cum_undersize_pct: np.ndarray


@dataclass(frozen=True)
class ProbitLine:
    """
    Параметры пробит-линейной регрессии для log–probit представления.

    Модель:
        probit = a * log10(d) + b,

    где:
        d — аэродинамический диаметр (мкм),
        probit = Φ⁻¹(p) + 5,
        p = F(d) / 100,
        F(d) — cumulative undersize (в %).

    Атрибуты
    --------
    a:
        Наклон линии в координатах probit vs log10(d).
    b:
        Свободный член (смещение) в координатах probit vs log10(d).
    r2:
        Коэффициент детерминации R^2.
    """

    a: float
    b: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class LinestStats:
    """
    Статистика линейной регрессии (аналог Excel ЛИНЕЙН с stats=TRUE).

    Атрибуты
    --------
    slope:
        Наклон m в модели y = a*x + b.
    intercept:
        Свободный член c в модели y = a*x + b.
    se_slope:
        Стандартная ошибка наклона.
    se_intercept:
        Стандартная ошибка свободного члена.
    r2:
        Коэффициент детерминации R^2.
    syx:
        Стандартная ошибка регрессии (SE_yx) = sqrt(SS_res / df).
    f_stat:
        F-статистика.
    df:
        Число степеней свободы (n - 2).
    ss_reg:
        Сумма квадратов регрессии.
    ss_res:
        Сумма квадратов остатков.
    """

    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    r2: float
    syx: float
    f_stat: float
    df: int
    ss_reg: float
    ss_res: float


@dataclass(frozen=True)
class MmadResultLS:
    """
    Результат расчёта MMAD и дополнительные метрики.

    Все диаметры выражены в микрометрах (мкм), массы — в микрограммах (мкг),
    если не указано иначе.

    Атрибуты
    --------
    mmad:
        Медианный аэродинамический диаметр распределения:
            mmad = D50 = exp(intercept).

    kor_k:
        «Корень K» — параметр ширины распределения, вычисляемый из sigma(ln):
            K = exp(sigma); конкретная формула зависит

    sigma:
        sigma(ln) — стандартное отклонение распределения в координате ln(d).
        В линейной модели соответствует наклону slope (m).

    r:
        Коэффициент корреляции Пирсона между x и y в линейной регрессии
        (например, между arcerf(2p-1) и ln(d)), безразмерный.

    slope:
        m — наклон линейной регрессии в модели ln(d) = m*x + c,
        где x = arcerf(2p-1). Обычно slope ≈ sigma(ln).

    intercept:
        c — свободный член регрессии. Обычно intercept = mu = ln(D50).

    se_slope:
        Стандартная ошибка оценки наклона slope (SE(m)).

    se_intercept:
        Стандартная ошибка оценки свободного члена intercept (SE(c)).

    r2:
        Коэффициент детерминации R² для линейной регрессии.

    syx:
        Стандартная ошибка регрессии (SE_yx, standard error of estimate),
        т.е. оценка разброса остатков по оси y (ln(d)).

    f_stat:
        F-статистика значимости регрессии (обычно для проверки гипотезы
        slope = 0) при заданных степенях свободы.

    df:
        Число степеней свободы для оценки дисперсии остатков.
        Для простой линейной регрессии обычно df = n - 2.

    ss_reg:
        SS_reg — сумма квадратов, объяснённая регрессией (регрессионная).

    ss_res:
        SS_res — сумма квадратов остатков (необъяснённая регрессией).

    y_ln_d:
        логарифмы диаметров, необходимы для построения графика линеаризации ФЗП.

    x_erfinv:
        значения arcerf(2p-1), необходимы для построения графика линеаризации ФЗП.
    """

    mmad: float
    kor_k: float
    sigma: float
    r: float
    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    r2: float
    syx: float
    f_stat: float
    df: int
    ss_reg: float
    ss_res: float
    y_ln_d: np.ndarray
    x_erfinv: np.ndarray
