# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from email import message
from statistics import NormalDist
from typing import Iterable, Optional
import math
import numpy as np


@dataclass(frozen=True)
class StageRecord:
    """Запись по элементу тракта/ступени импактора."""

    name: str
    d_low: Optional[float]
    d_high: Optional[float]
    mass: float


@dataclass(frozen=True)
class LinestStats:
    """
    Статистика линейной регрессии (аналог Excel LINEST с stats=TRUE).

    Attributes
    ----------
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


def _as_float_array_optional(values: Iterable[Optional[float]]) -> np.ndarray:
    """
    Преобразует последовательность Optional[float] в ndarray float,
    подставляя np.nan вместо None.
    """
    return np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=float,
    )


def _normal_ppf(p: np.ndarray) -> np.ndarray:
    """
    Φ^{-1}(p) без SciPy через стандартную библиотеку Python.
    """
    nd = NormalDist()
    inv = np.vectorize(nd.inv_cdf, otypes=[float])
    return inv(p.astype(float))


def _regression_significance_message(
    f_stat: float,
    df: int,
    alpha: float = 0.05,
) -> str:
    """
    Интерпретация статистической значимости линейной регрессии.

    Используется связь F и t для модели с одним предиктором:

        t = sqrt(F)

    Далее проверяется двухсторонний t-test.

    Параметры
    ----------
    f_stat:
        F-статистика регрессии.
    df:
        Степени свободы (n - 2).
    alpha:
        Уровень значимости.

    Возвращает
    ---------
    str
        Текстовое заключение.
    """

    if not math.isfinite(f_stat) or f_stat <= 0:
        return "Регрессия статистически незначима."

    # связь F и t
    t_value = math.sqrt(f_stat)

    # приближение через нормальное распределение
    nd = NormalDist()

    # двухсторонний p-value (аппроксимация)
    p_value = 2 * (1 - nd.cdf(abs(t_value)))

    if p_value < alpha:
        return (
            f"Регрессия статистически значима "
            f"(p ≈ {p_value:.3g} < {alpha}). "
            "Логнормальная модель согласуется с данными."
        )

    return (
        f"Регрессия НЕ статистически значима "
        f"(p ≈ {p_value:.3g}). "
        "Логнормальная модель может быть неприменима."
    )

def linest_with_stats(y: np.ndarray, x: np.ndarray) -> LinestStats:
    """
    LINEST(y, x, TRUE, TRUE): линейная регрессия y = a*x + b + статистика.

    Возвращаем статистику в формате, близком к Excel:
    - a, b
    - SE(a), SE(b)
    - R^2
    - SE_yx
    - F, df
    - SS_reg, SS_res
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = int(x.size)
    if n < 3:
        raise ValueError("Для статистики LINEST нужно минимум 3 точки (n >= 3).")

    # Оценки a и b методом МНК
    a, b = np.polyfit(x, y, deg=1)

    y_hat = a * x + b
    resid = y - y_hat

    ss_res = float(np.sum(resid**2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    ss_reg = float(np.sum((y_hat - y_mean) ** 2))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    df = n - 2
    syx = float(np.sqrt(ss_res / df))

    # Оценка стандартных ошибок коэффициентов
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0.0:
        raise ValueError("Невозможно оценить SE: все x одинаковые.")

    se_slope = float(syx / np.sqrt(sxx))
    se_intercept = float(syx * np.sqrt(1.0 / n + (x_mean**2) / sxx))

    # F-статистика (1 предиктор)
    if ss_res <= 0.0:
        f_stat = float("inf")
    else:
        f_stat = float((ss_reg / 1.0) / (ss_res / df))

    return LinestStats(
        slope=float(a),
        intercept=float(b),
        se_slope=float(se_slope),
        se_intercept=float(se_intercept),
        r2=float(r2),
        syx=float(syx),
        f_stat=float(f_stat),
        df=int(df),
        ss_reg=float(ss_reg),
        ss_res=float(ss_res),
    )


def main() -> None:

    # Тестовые исходные данные
    records = [
        StageRecord("Фильтр", None, 0.43, 0.0052),
        StageRecord("Ступень 7", 0.43, 0.65, 0.0009),
        StageRecord("Ступень 6", 0.65, 1.1, 0.003),
        StageRecord("Ступень 5", 1.1, 2.1, 0.0319),
        StageRecord("Ступень 4", 2.1, 3.3, 0.0714),
        StageRecord("Ступень 3", 3.3, 4.7, 0.0714),
        StageRecord("Ступень 2", 4.7, 5.8, 0.0427),
        StageRecord("Ступень 1", 5.8, 9.0, 0.0149),
        StageRecord("Ступень 0", 9.0, 10.0, 0.0028),
        StageRecord("Пресепаратор", 10.0, None, 0.0036),
        StageRecord("Входной канал", None, None, 0.0),
    ]

    d_low = _as_float_array_optional([r.d_low for r in records])
    d_high = _as_float_array_optional([r.d_high for r in records])
    mass = _as_float_array_optional([r.mass for r in records])

    # Репрезентативный диаметр ступени
    # d_i = sqrt(d_low * d_high)
    mask_intervals = (
        np.isfinite(d_low)
        & np.isfinite(d_high)
        & (d_low > 0.0)
        & (d_high > d_low)
        & np.isfinite(mass)
        & (mass >= 0.0)
    )

    if np.sum(mask_intervals) < 3:
        raise ValueError("Недостаточно корректных интервалов (нужно >= 3).")

    d_low = d_low[mask_intervals]
    d_high = d_high[mask_intervals]
    mass = np.asarray(
        [records[0].mass, *mass[mask_intervals], records[9].mass, records[10].mass],
        dtype=float,
    )

    d_g = np.sqrt(d_low * d_high)
    y_ln_d = np.log(d_g)

    # Считаем p как долю накопленной массы
    total_mass = np.sum(mass)
    if total_mass <= 0.0:
        raise ValueError("Суммарная масса (фильтр + ступени 7..0) должна быть > 0.")

    cum_mass = np.cumsum(mass)
    p = cum_mass / total_mass
    # Исключаем из расчета фильтр и пресепаратор
    p = p[1:9]

    # arcerf(2p-1) = NORMSINV(p)/sqrt(2)
    # исключаем p=0 и p=1
    mask_p = (p > 0.0) & (p < 1.0)
    if np.sum(mask_p) < 3:
        raise ValueError("После исключения p=0 и p=1 осталось < 3 точек.")

    y_ln_d = y_ln_d[mask_p]
    p = p[mask_p]

    x_erfinv = _normal_ppf(p) / np.sqrt(2.0)

    # LINEST
    stats = linest_with_stats(y=y_ln_d, x=x_erfinv)

    d50_um = float(np.exp(stats.intercept))
    sigma = float(stats.slope / np.sqrt(2.0))  # σ логнормального распределения
    kor_k = float(1 / stats.slope)
    r = float(stats.r2**0.5)

    significance_message = _regression_significance_message(stats.f_stat, stats.df)

    # Результата + статистика
    print("Модель: ln(d) = m * arcerf(2p-1) + c")
    print()
    print(f"m (slope)      = {stats.slope:.2f}")
    print(f"c (intercept)  = {stats.intercept:.2f}")
    print(f"SE(m)          = {stats.se_slope:.2f}")
    print(f"SE(c)          = {stats.se_intercept:.2f}")
    print(f"R^2            = {stats.r2:.2f}")
    print(f"SE_yx (syx)    = {stats.syx:.2f}")
    print(f"F              = {stats.f_stat:.2f}")
    print(f"df             = {stats.df:d}")
    print(f"SS_reg         = {stats.ss_reg:.2f}")
    print(f"SS_res         = {stats.ss_res:.2f}")
    print(f"{significance_message}")
    print()
    print("=== Параметры закона ===")
    print(f"D50 = exp(mu), мкм      = {d50_um:.2f}")
    print(f"Корень К                = {kor_k:.2f}")
    print(f"sigma(ln)               = {sigma:.2f}")
    print(f"Коэфф. корреляции       = {r:.2f}")


if __name__ == "__main__":
    main()
