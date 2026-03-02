# -*- coding: utf-8 -*-
# src/mmad_app/core/mmad.py
from __future__ import annotations

from typing import Iterable, List, Tuple, Optional
import numpy as np

from mmad_app.core.models import StageRecord, MmadResult, LinestStats, MmadResultLS
from mmad_app.core.normal import normal_ppf


def _validate_records(records: List[StageRecord]) -> None:
    """
    Проверяет корректность входных данных.

    Важно:
    - d_low/d_high могут быть None (например, фильтр, пресепаратор, входной канал)
    - масса должна быть конечной и неотрицательной
    - для расчётов нужна как минимум некоторая "размерная" часть данных
      (хотя бы 2 границы диаметра среди d_low/d_high)
    """
    if len(records) < 3:
        raise ValueError("Необходимо минимум 3 записи для расчёта.")

    masses = np.array([float(r.mass) for r in records], dtype=float)
    if np.any(~np.isfinite(masses)):
        raise ValueError("Обнаружены нечисловые (NaN/inf) значения массы.")
    if np.any(masses < 0.0):
        raise ValueError("Масса не может быть отрицательной.")

    total = float(np.sum(masses))
    if total <= 0.0:
        raise ValueError("Суммарная масса должна быть > 0.")

    # Собираем все заданные границы диаметра (и d_low, и d_high)
    bounds: List[float] = []
    for r in records:
        if r.d_low is not None:
            bounds.append(float(r.d_low))
        if r.d_high is not None:
            bounds.append(float(r.d_high))

    bounds_arr = np.array(bounds, dtype=float)
    bounds_arr = bounds_arr[np.isfinite(bounds_arr) & (bounds_arr > 0.0)]

    if bounds_arr.size < 2:
        raise ValueError("Недостаточно заданных границ диаметра (нужно >= 2).")


def _as_float_array_optional(values: Iterable[Optional[float]]) -> np.ndarray:
    """
    Преобразует последовательность Optional[float] в ndarray float,
    подставляя np.nan вместо None.
    """
    return np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=float,
    )


def compute_cumulative_undersize(
    records: Iterable[StageRecord],
    *,
    include_unclassified_in_total: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует накопительную кривую F(d) = cumulative undersize (%),
    включая фракции < 0.43 мкм (фильтр) и > 10 мкм (пресепаратор).

    Математически:
        F(d) = 100 * (Σ m_i, где D_i <= d) / (Σ m_total)

    Здесь мы трактуем элементы так:
    - если задан d_high (и d_low=None) -> "нижняя" фракция (фильтр): (0, d_high]
    - если заданы d_low и d_high -> "интервал" ступени: (d_low, d_high]
    - если задан d_low (и d_high=None) -> "верхняя" фракция (пресепаратор): (d_low, +∞)
    - если обе границы None -> неклассифицированная масса (входной канал/потери)

    Параметры
    ---------
    records:
        Итератор записей.
    include_unclassified_in_total:
        Если True, масса записей без границ (None/None) включается в знаменатель.
        По умолчанию False (иначе кумулятива “сожмётся” без привязки к диаметрам).

    Возвращает
    ----------
    (diam_um, cum_undersize_pct)
        diam_um: диаметры (мкм) строго > 0
        cum_undersize_pct: F(d), % (монотонно неубывающая)
    """
    res_list = list(records)
    _validate_records(res_list)

    # 1) Разделяем записи
    bins_upper: List[float] = []  # верхняя граница бина (для суммирования “<= d”)
    bins_mass: List[float] = []
    unclassified_mass = 0.0

    for r in res_list:
        m = float(r.mass)
        if m <= 0.0:
            continue

        if r.d_low is None and r.d_high is None:
            # Входной канал / потери: нет связи с диаметром
            unclassified_mass += m
            continue

        if r.d_low is None and r.d_high is not None:
            # Фильтр: (0, d_high]
            upper = float(r.d_high)
            if upper > 0.0 and np.isfinite(upper):
                bins_upper.append(upper)
                bins_mass.append(m)
            continue

        if r.d_low is not None and r.d_high is not None:
            # Ступень: (d_low, d_high]
            low = float(r.d_low)
            high = float(r.d_high)
            if (low > 0.0) and np.isfinite(low) and np.isfinite(high) and (high > low):
                bins_upper.append(high)
                bins_mass.append(m)
            continue

        if r.d_low is not None and r.d_high is None:
            # Пресепаратор: (d_low, +inf)
            # Для cumulative undersize по определению эта масса НЕ входит в F(d)
            # при d <= d_low. Чтобы довести кривую до 100%, добавим точку d_max.
            # Сохраним upper = +inf как маркер.
            bins_upper.append(float("inf"))
            bins_mass.append(m)
            continue

    if len(bins_mass) < 2:
        raise ValueError("Недостаточно интервалов/фракций для построения кумулятивы.")

    bins_upper_arr = np.asarray(bins_upper, dtype=float)
    bins_mass_arr = np.asarray(bins_mass, dtype=float)

    # 2) Знаменатель (total mass)
    total_mass = float(np.sum(bins_mass_arr))
    if include_unclassified_in_total:
        total_mass += float(unclassified_mass)

    if total_mass <= 0.0:
        raise ValueError("Суммарная масса должна быть > 0.")

    # 3) Формируем узлы кривой по всем конечным верхним границам
    finite_uppers = bins_upper_arr[np.isfinite(bins_upper_arr)]
    finite_uppers = finite_uppers[finite_uppers > 0.0]

    if finite_uppers.size == 0:
        raise ValueError(
            "Нет ни одной конечной границы диаметра для построения кривой."
        )

    unique_d = np.unique(np.sort(finite_uppers))

    # Добавим “нижнюю” стартовую точку (чтобы были < min_d)
    min_d = float(unique_d[0])
    d_min = max(min_d / 1_000.0, 1e-6)  # строго > 0 для лог-интерполяций

    # Добавим “верхнюю” точку d_max, чтобы F(d_max)=100%
    max_d = float(unique_d[-1])
    d_max = max(max_d * 1.5, max_d + 1.0)

    diam_points = np.concatenate(([d_min], unique_d, [d_max])).astype(float)

    # 4) Кумулятива: F(d) = sum(mass where upper <= d)/total*100,
    # а пресепаратор (upper=inf) добавит скачок только на d_max (через 100%).
    cum_points: List[float] = [0.0]

    for d in unique_d:
        frac = float(np.sum(bins_mass_arr[bins_upper_arr <= d])) / total_mass
        cum_points.append(100.0 * frac)

    cum_points.append(100.0)
    cum_arr = np.asarray(cum_points, dtype=float)

    # 5) Гарантируем монотонность
    cum_arr = np.maximum.accumulate(cum_arr)

    return diam_points, cum_arr


def interpolate_d_at_cum_pct_logx(
    diam_um: np.ndarray,
    cum_pct: np.ndarray,
    target_pct: float,
) -> float:
    """
    Находит диаметр, соответствующий заданной накопленной доле массы (< d).

    Интерполяция выполняется:
    - линейно по оси Y (проценты),
    - по оси X в логарифмической координате log10(d).

    Параметры
    ---------
    diam_um:
        Диаметры (мкм) для накопительной кривой. Используются только значения > 0.
    cum_pct:
        Накопленная доля массы частиц с диаметром < d, %.
    target_pct:
        Целевой уровень кумулятивы в процентах (0 < target_pct < 100).

    Возвращает
    ----------
    float
        Диаметр d (мкм), для которого F(d) = target_pct (с log-интерполяцией).

    Исключения
    ----------
    ValueError
        Если target_pct вне (0, 100), недостаточно точек для интерполяции,
        или target_pct вне диапазона заданной кумулятивной кривой.
    """
    # Проверка корректности целевого процента
    if not 0.0 < target_pct < 100.0:
        raise ValueError("target_pct должен быть в (0, 100).")

    mask = diam_um > 0.0
    x = diam_um[mask].astype(float)
    y = cum_pct[mask].astype(float)

    if x.size < 2:
        raise ValueError("Недостаточно точек с положительным диаметром.")

    # Сортировка пары (x, y) по возрастанию диаметра x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Накопительная кривая должна возрастать.
    y = np.maximum.accumulate(y)

    # Проверка что target_pct лежит в диапазоне доступных значений
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if not y_min <= target_pct <= y_max:
        raise ValueError(
            f"target_pct={target_pct} вне диапазона кривой [{y_min:.2f}, {y_max:.2f}]."
        )

    # Нахождение индекса k первого значения y[k], которое >= target_pct
    k = int(np.searchsorted(y, target_pct, side="left"))
    if k == 0:
        # target_pct совпал/ниже первой точки
        return float(x[0])
    if k >= len(y):
        # target_pct выше последней точки
        return float(x[-1])

    # Опорные точки для интерполяции
    x0, x1 = float(x[k - 1]), float(x[k])
    y0, y1 = float(y[k - 1]), float(y[k])

    # Если y не меняется, возвращается левая граница
    if y1 == y0:
        return float(x0)

    # Доля прохождения по Y от y0 к y1
    t = (target_pct - y0) / (y1 - y0)

    # Интерполяция по логарифму диаметра:
    # log10(d) меняется линейно, затем возвращение к d через 10**(...)
    lx0 = float(np.log10(x0))
    lx1 = float(np.log10(x1))
    lxt = lx0 + t * (lx1 - lx0)

    return float(10.0**lxt)


def interpolate_cum_pct_at_d_logx(
    diam_um: np.ndarray,
    cum_pct: np.ndarray,
    d_um: float,
) -> float:
    """
    Определяет накопленную долю массы частиц с диаметром меньше заданного.

    Функция вычисляет значение cumulative undersize F(d) (в процентах)
    при заданном аэродинамическом диаметре d, используя интерполяцию
    в логарифмической координате диаметра.

    Интерполяция выполняется:
    - по оси X в координате log10(d),
    - по оси Y линейно (по процентам).

    Параметры
    ---------
    diam_um:
        Диаметры (мкм), соответствующие точкам накопительной кривой.
        Используются только значения d > 0.
    cum_pct:
        Накопленная доля массы частиц с диаметром < d, выраженная в процентах.
    d_um:
        Аэродинамический диаметр (мкм), для которого требуется определить
        накопленную долю массы.

    Возвращает
    ----------
    float
        Накопленная доля массы частиц с диаметром < d_um, выраженная в процентах.

    """
    if d_um <= 0.0:
        return 0.0

    mask = (diam_um > 0.0) & np.isfinite(diam_um) & np.isfinite(cum_pct)
    x = diam_um[mask].astype(float)
    y = cum_pct[mask].astype(float)

    if x.size < 2:
        raise ValueError(
            "Недостаточно точек с положительным диаметром для интерполяции."
        )

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Накопителньая кривая должна возрастать
    y = np.maximum.accumulate(y)

    # Clamp по X
    if d_um <= float(x[0]):
        return float(y[0])
    if d_um >= float(x[-1]):
        return float(y[-1])

    # Интервал по X
    k = int(np.searchsorted(x, d_um, side="left"))
    if k == 0:
        return float(y[0])
    if k >= len(x):
        return float(y[-1])

    x0, x1 = float(x[k - 1]), float(x[k])
    y0, y1 = float(y[k - 1]), float(y[k])

    if x1 == x0:
        return float(y0)

    # Интерполяция по log10(x)
    lx0 = float(np.log10(x0))
    lx1 = float(np.log10(x1))
    lxd = float(np.log10(d_um))

    t = (lxd - lx0) / (lx1 - lx0)
    return float(y0 + t * (y1 - y0))


def computate_linest_with_stats(y: np.ndarray, x: np.ndarray) -> LinestStats:
    """
    Выполняет линейную регрессию y = a*x + b и возвращает статистику,
    аналогичную Excel LINEST(y_range, x_range, TRUE, TRUE).

    Используемые обозначения
    ------------------------
    Пусть есть n наблюдений (x_i, y_i), i = 1..n. Модель:

        y_i = a * x_i + b + ε_i

    где a — наклон (slope), b — свободный член (intercept), ε_i — остатки.

    Возвращаемые показатели (как в Excel)
    -------------------------------------
    - slope (a), intercept (b)
    - se_slope, se_intercept:
        стандартные ошибки коэффициентов a и b
    - r2:
        коэффициент детерминации R²
    - syx:
        стандартная ошибка регрессии (SE_yx):
            syx = sqrt(SS_res / df)
    - f_stat:
        F-статистика общей значимости регрессии (для 1 предиктора)
    - df:
        степени свободы df = n - 2
    - ss_reg:
        сумма квадратов регрессии SS_reg = Σ(ŷ_i - ȳ)²
    - ss_res:
        сумма квадратов остатков SS_res = Σ(y_i - ŷ_i)²

    Параметры
    ---------
    y:
        Наблюдаемые значения отклика (зависимая переменная).
    x:
        Значения предиктора (независимая переменная).

    Возвращает
    ----------
    LinestStats
        Набор оценок коэффициентов и статистик регрессии.
    """
    # Приведение входов к ndarray float (на случай списков/кортежей).
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # Оставляем только конечные значения (исключаем NaN и +/-inf).
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = int(x.size)
    if n < 3:
        # Для stats=TRUE в Excel нужна статистика, а она требует df = n - 2 >= 1.
        raise ValueError("Для статистики ЛИНЕЙН нужно минимум 3 точки (n >= 3).")

    # 1) Оценка коэффициентов a, b методом МНК
    # np.polyfit(x, y, 1) возвращает [a, b] для y ≈ a*x + b.
    a, b = np.polyfit(x, y, deg=1, full=False)

    # Предсказания модели и остатки.
    y_hat = a * x + b
    resid = y - y_hat

    # 2) Суммы квадратов и R²
    ss_res = float(np.sum(resid**2))  # SS_res = Σ (y - ŷ)^2
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))  # SS_tot = Σ (y - ȳ)^2
    ss_reg = float(np.sum((y_hat - y_mean) ** 2))  # SS_reg = Σ (ŷ - ȳ)^2

    # R² = 1 - SS_res / SS_tot. Если SS_tot = 0 (все y одинаковые),
    # считаем R² = 1.0 (модель "объясняет" нулевую дисперсию).
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    # 3) Стандартная ошибка регрессии (SE_yx) и стандартные ошибки коэффициентов
    df = n - 2  # степени свободы для модели с 2 параметрами (a и b)
    syx = float(np.sqrt(ss_res / df))  # SE_yx = sqrt(SS_res / df)

    # Sxx = Σ (x - x̄)^2
    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0.0:
        # Все x одинаковые => линия не идентифицируема (наклон оценить можно,
        # но ошибки/значимость некорректны).
        raise ValueError("Невозможно оценить SE: все x одинаковые.")

    # SE(a) = syx / sqrt(Sxx)
    se_slope = float(syx / np.sqrt(sxx))

    # SE(b) = syx * sqrt(1/n + x̄^2 / Sxx)
    se_intercept = float(syx * np.sqrt(1.0 / n + (x_mean**2) / sxx))

    # 4) F-статистика общей значимости регрессии
    # Для одной независимой переменной:
    #   F = (SS_reg / 1) / (SS_res / df)
    if ss_res <= 0.0:
        # Идеальная аппроксимация (остатки ~ 0) => F -> бесконечность.
        f_stat = float("inf")
    else:
        f_stat = float(ss_reg / (ss_res / df))

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


def compute_mmad(records: Iterable[StageRecord]) -> MmadResult:
    """
    Выполняет расчёт основных метрик APSD по данным каскадного импактора.

    Функция строит накопленную массовую долю частиц с диаметром меньше заданного
    (cumulative undersize) и извлекает из неё квантильные диаметры, а также
    вычисляет дополнительные характеристики распределения.

    Этапы расчёта
    -------------
    1. Построение накопительной кривой F(d), %:
       F(d) — накопленная доля массы частиц с аэродинамическим диаметром < d,
       выраженная в процентах.

    2. Расчёт квантильных диаметров по накопительной кривой:
       - d10, d16, d50 (MMAD), d84, d90.

    3. Расчёт GSD по «нормальным» квантилям (±1σ в пробит-подходе):
       - d15.87 и d84.13,
       - GSD = sqrt(d84.13 / d15.87).

    4. Расчёт ширины распределения (Span):
       - Span = (d90 - d10) / d50.

    5. Расчёт FPF (Fine Particle Fraction) для заданного порога d_cut
       (по умолчанию 5 мкм):
       - FPF(<d_cut) = F(d_cut), %.

    6. Дополнительные диаметры по интервалам ступеней (если доступны d_low/d_high):
       Для каждой ступени используется репрезентативный диаметр интервала:
           d_i = sqrt(d_low * d_high).

       На их основе вычисляются:
       - логарифмический средний диаметр (геометрическое среднее по массе):
             d_g = exp(Σ w_i ln(d_i))
       - среднемассовый диаметр (массо-взвешенное арифметическое среднее):
             d_m = Σ w_i d_i
       - модальный аэродинамический диаметр:
             d_mode = d_i, для которого w_i максимально

       где w_i = m_i / Σ m_i — массовые доли.

    Параметры
    ---------
    records:
        Итератор записей ступеней. Должны быть заданы массы, а для расчёта
        дополнительных средних/моды — границы интервалов d_low и d_high.

    Возвращает
    ----------
    MmadResult
        Структура с рассчитанными метриками и данными накопительной кривой
        для построения графиков.
    """
    rec_list = list(records)
    diam, cum_pct = compute_cumulative_undersize(rec_list)

    # Суммарная масса
    total_mass = float(sum(float(r.mass) for r in rec_list))
    if total_mass <= 0.0:
        raise ValueError("Суммарная масса должна быть > 0.")

    # Квантили по накопительной кривой
    mmad = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=50.0)

    # "Инженерные" квантили 16/84
    d16 = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=16.0)
    d84 = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=84.0)

    # "Нормальные" квантили для GSD
    d15_87 = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=15.87)
    d84_13 = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=84.13)

    if d15_87 <= 0.0:
        raise ValueError("d15.87 получился <= 0, невозможно вычислить GSD.")
    gsd = float(np.sqrt(d84_13 / d15_87))

    # D10/D90 и Span
    d10 = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=10.0)
    d90 = interpolate_d_at_cum_pct_logx(diam, cum_pct, target_pct=90.0)

    if mmad <= 0.0:
        raise ValueError("MMAD получился <= 0, невозможно вычислить Span.")
    span = float((d90 - d10) / mmad)

    # FPF для порога 5 мкм
    fpf_cutoff_um = 5.0
    fpf_pct = interpolate_cum_pct_at_d_logx(diam, cum_pct, d_um=fpf_cutoff_um)

    # Массо-взвешенные средние и мода по интервалам ступеней
    dg_bins: List[float] = []
    m_bins: List[float] = []

    for r in rec_list[1:9]:
        # Пропускаем записи без интервала
        if r.d_low is None or r.d_high is None:
            continue

        d_low = float(r.d_low)
        d_high = float(r.d_high)
        m = float(r.mass)

        # Пропуск некорректных интервалов и нулевой/отрицательной массы
        if not np.isfinite(d_low) or not np.isfinite(d_high) or not np.isfinite(m):
            continue
        if m <= 0.0:
            continue
        if d_low <= 0.0 or d_high <= d_low:
            continue

        # Репрезентативный диаметр интервала: геометрический центр
        d_bin = float(np.sqrt(d_low * d_high))
        dg_bins.append(d_bin)
        m_bins.append(m)

    # Значения по умолчанию, если интервальные метрики посчитать нельзя
    log_mean = float("nan")
    mass_mean = float("nan")
    modal = float("nan")

    if dg_bins:
        d_arr = np.asarray(dg_bins, dtype=float)
        m_arr = np.asarray(m_bins, dtype=float)

        m_sum = float(np.sum(m_arr))
        if m_sum > 0.0:
            # Массовые доли внутри выбранных бинов (ступени 7..0)
            w_arr = m_arr / m_sum

            # Логарифмический средний диаметр
            log_mean = float(np.exp(np.sum(w_arr * np.log(d_arr))))

            # Среднемассовый диаметр
            mass_mean = float(np.sum(w_arr * d_arr))

            # Модальный диаметр
            modal = float(d_arr[int(np.argmax(w_arr))])

    return MmadResult(
        mmad=float(mmad),
        gsd=float(gsd),
        d10=float(d10),
        d16=float(d16),
        d84=float(d84),
        d90=float(d90),
        d15_87=float(d15_87),
        d84_13=float(d84_13),
        span=float(span),
        fpf_cutoff_um=float(fpf_cutoff_um),
        fpf_pct=float(fpf_pct),
        total_mass=float(total_mass),
        log_mean=float(log_mean),
        mass_mean=float(mass_mean),
        modal=float(modal),
        diam_um=diam,
        cum_undersize_pct=cum_pct,
    )


def compute_mmad_least_squares(records: Iterable[StageRecord]) -> MmadResultLS:

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
    rec_list = list(records)

    mass = np.asarray(
        [rec_list[0].mass, *mass[mask_intervals], rec_list[9].mass, rec_list[10].mass],
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

    x_erfinv = normal_ppf(p) / np.sqrt(2.0)

    # LINEST
    stats = computate_linest_with_stats(y=y_ln_d, x=x_erfinv)

    d50 = float(np.exp(stats.intercept))
    sigma = float(stats.slope / np.sqrt(2.0))  # σ логнормального распределения
    kor_k = float(1 / stats.slope)
    r = float(stats.r2**0.5)

    return MmadResultLS(
        mmad=d50,
        kor_k=kor_k,
        sigma=sigma,
        r=r,
        slope=stats.slope,
        intercept=stats.intercept,
        se_slope=stats.se_slope,
        se_intercept=stats.se_intercept,
        r2=stats.r2,
        syx=stats.syx,
        f_stat=stats.f_stat,
        df=stats.df,
        ss_reg=stats.ss_reg,
        ss_res=stats.ss_res,
        # добавление точек для построения графика
        y_ln_d=y_ln_d,
        x_erfinv=x_erfinv,
    )
