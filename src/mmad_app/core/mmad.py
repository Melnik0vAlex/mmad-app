# -*- coding: utf-8 -*-
# src/mmad_app/core/mmad.py
from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

from mmad_app.core.models import StageRecord, MmadResult


def _validate_records(records: List[StageRecord]) -> None:
    """
    Валидация исходных данных
    """
    if len(records) < 3:
        raise ValueError("Необходимо минимум 3 заполненные ступени для расчета.")

    d_low = np.array([r.d_low for r in records], dtype=float)

    mass = np.array([r.mass for r in records], dtype=float)

    if np.any(~np.isfinite(d_low)) or np.any(~np.isfinite(mass)):
        raise ValueError("Обнаружены нечисловые (NaN/inf) значения.")

    if np.any(mass < 0):
        raise ValueError("Масса не может быть отрицательно.")

    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("Суммарная масса должна быть > 0.")

    if np.sum(d_low > 0.0) < 2:
        raise ValueError("Необходимо минимум 2 положительных D50 (мкм)")


def compute_cumulative_undersize(
    records: Iterable[StageRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Формирует накопительную кривую накопленной доли массы частиц (< d).

    Для каждой ступени каскадного импактора cumulative undersize
    определяется как доля массы частиц с аэродинамическим диаметром
    меньше отсечного диаметра данной ступени.

    Алгоритм:
    1. Ступени сортируются по отсечному диаметру (от крупных к мелким).
    2. Для каждого отсечного диаметра d_i вычисляется:
           F(d_i) = Σ_{j > i} m_j / Σ m * 100 %,
       где m_j — масса, осевшая на более мелких ступенях.
    3. Добавляется граничная точка для крупных частиц: F(d_top) = 100 %.
    4. Данные упорядочиваются по возрастанию диаметра и очищаются от дубликатов.

    Параметры
    ---------
    records:
        Итератор записей ступеней импактора. Для каждой записи
        должны быть заданы нижний отсечной диаметр и масса.

    Возвращает
    ----------
    Tuple[np.ndarray, np.ndarray]
        Два массива одинаковой длины:
        - diam_um : диаметры (мкм),
        - cum_undersize_pct : накопленная доля массы частиц с диаметром < d, %.
    """
    res_list = list(records)
    _validate_records(res_list)

    # Сортировка отсечных диаметров D50 (сверху крупные, снизу мелкие)
    res_sorted = sorted(res_list, key=lambda r: r.d_low, reverse=True)

    d_cut = np.array([r.d_low for r in res_sorted], dtype=float)
    mass = np.array([r.mass for r in res_sorted], dtype=float)

    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("Суммарная масса должна быть > 0.")

    cum_at_d50 = np.array(
        [100 * float(np.sum(mass[i + 1 :])) / total for i in range(len(mass))],
        dtype=float,
    )

    # Граничные условия для графика и интерполяции
    d_top = 10.0

    diam = np.concatenate((d_cut, [d_top])).astype(float)
    cum = np.concatenate((cum_at_d50, [100.0])).astype(float)

    # Упорядочивание по возрастанию диаметра
    order = np.argsort(diam)
    diam = diam[order]
    cum = cum[order]

    # Удаление дубликатов диаметров
    uniq_d = []
    uniq_c = []

    for d, c in zip(diam, cum):
        if not uniq_d or d != uniq_d[-1]:
            uniq_d.append(float(d))
            uniq_c.append(float(c))
        else:
            uniq_c[-1] = max(uniq_c[-1], float(c))

    diam_um = np.array(uniq_d, dtype=float)
    cut_pct = np.array(uniq_c, dtype=float)
    return diam_um, cut_pct


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
    d10 = interpolate_d_at_cum_pct_logx(diam[1:7], cum_pct[1:7], target_pct=10.0)
    d90 = interpolate_d_at_cum_pct_logx(diam[1:7], cum_pct[1:7], target_pct=90.0)

    if mmad <= 0.0:
        raise ValueError("MMAD получился <= 0, невозможно вычислить Span.")
    span = float((d90 - d10) / mmad)

    # FPF для порога 5 мкм
    fpf_cutoff_um = 5.0
    fpf_pct = interpolate_cum_pct_at_d_logx(diam, cum_pct, d_um=fpf_cutoff_um)

    # Массо-взвешенные средние и мода по интервалам ступеней
    dg_bins: List[float] = []
    w_bins: List[float] = []

    for r in rec_list:
        d_low = float(getattr(r, "d_low", 0.0))
        d_high = float(getattr(r, "d_high", 0.0))
        m = float(getattr(r, "mass", 0.0))

        # Пропуск некорректных интервалов и нулевой массы
        if m <= 0.0:
            continue
        if d_low <= 0.0 or d_high <= 0.0 or d_high <= d_low:
            continue

        # Репрезентативный диаметр интервала: геометрический центр
        d_bin = float(np.sqrt(d_low * d_high))
        dg_bins.append(d_bin)
        w_bins.append(m / total_mass)

    # Значения по умолчанию, если не удалось собрать интервалы
    log_mean = float("nan")
    mass_mean = float("nan")
    modal = float("nan")

    if dg_bins:
        d_arr = np.array(dg_bins, dtype=float)
        w_arr = np.array(w_bins, dtype=float)

        # Нормировка весов на случай численных отклонений
        w_sum = float(np.sum(w_arr))
        if w_sum > 0.0:
            w_arr = w_arr / w_sum

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
