from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np

from mmad_app.core.models import StageRecord, MmadResult


def _validate_records(records: List[StageRecord]) -> None:
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
    Строит кумулятивную зависимость cumulative undersize (%) от диаметра (µm).
    """
    res_list = list(records)
    _validate_records(res_list)

    # Сортировка отсечных диаметров D50 (сверху крупные, снизу мелкие)
    res_sorted = sorted(res_list, key=lambda r: r.d_low, reverse=True)

    d50 = np.array([r.d_low for r in res_sorted], dtype=float)
    mass = np.array([r.mass for r in res_sorted], dtype=float)

    total = float(np.sum(mass))

    cum_at_d50 = np.array(
        [100 * float(np.sum(mass[i + 1 :]))/ total for i in range(len(mass))],
        dtype=float,
    )

    # Граничные условия для графика и интерполяции
    d_top = 10.0

    diam = np.concatenate((d50, [d_top])).astype(float)
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
    Находит диаметр d, при котором cumulative undersize = target_pct.

    Интерполяция:
    - линейная по y (процент)
    - по x используем log10(d), чтобы корректнее работать с размерными шкалами
    """
    if not (0.0 < target_pct < 100.0):
        raise ValueError("target_pct должен быть в (0, 100).")

    mask = diam_um > 0.0
    x = diam_um[mask].astype(float)
    y = cum_pct[mask].astype(float)

    if x.size < 2:
        raise ValueError("Недостаточно точек с положительным диаметром.")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # обеспечим монотонность по y
    y = np.maximum.accumulate(y)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if not (y_min <= target_pct <= y_max):
        raise ValueError(
            f"target_pct={target_pct} вне диапазона кривой [{y_min:.2f}, {y_max:.2f}]."
        )

    k = int(np.searchsorted(y, target_pct, side="left"))
    if k == 0:
        return float(x[0])
    if k >= len(y):
        return float(x[-1])

    x0, x1 = float(x[k - 1]), float(x[k])
    y0, y1 = float(y[k - 1]), float(y[k])

    if y1 == y0:
        return float(x0)

    t = (target_pct - y0) / (y1 - y0)
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
    Находит cumulative undersize (%) при заданном диаметре d_um.

    Интерполяция:
    - по оси X используем log10(d)
    - по оси Y интерполируем линейно

    Важно:
    - d_um должен быть > 0 (для log10)
    - если d_um вне диапазона X, используем "clamp" к границам
    """
    if d_um <= 0.0:
        return 0.0

    mask = diam_um > 0.0
    x = diam_um[mask].astype(float)
    y = cum_pct[mask].astype(float)

    if x.size < 2:
        raise ValueError("Недостаточно точек с положительным диаметром для интерполяции.")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Кумулятива должна быть неубывающей
    y = np.maximum.accumulate(y)

    # Clamp по X
    if d_um <= float(x[0]):
        return float(y[0])
    if d_um >= float(x[-1]):
        return float(y[-1])

    # Находим интервал по X
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
    Полный расчёт метрик APSD (массовое распределение по аэродинамическому диаметру).

    Считаем:
    1) Кумулятивную кривую cumulative undersize (%): F(d)
    2) Квантили распределения: d10, d16, d50 (MMAD), d84, d90
       (d16 и d84 — это 16% и 84% кумулятивы; иногда вместо 15.87/84.13)
    3) GSD по "нормальным" квантилям 15.87/84.13:
          GSD = sqrt(d84.13 / d15.87)
    4) Span = (d90 - d10) / d50
    5) FPF(<5 µm) по умолчанию
    6) Доп. диаметры по интервалам (используем границы ступеней):
       - Логарифмический средний диаметр (геометрическое среднее):
             d_g = exp(Σ w_i ln(d_i))
       - Среднемассовый диаметр (массо-взвешенный арифметический):
             d_mass = Σ w_i d_i
       - Модальный аэродинамический диаметр (мода по интервалам):
             d_mode = d_i, где w_i максимально

    Где w_i = m_i / Σ m_i, а d_i — репрезентативный диаметр интервала ступени:
       d_i = sqrt(d_low * d_high) (средний геометрический диаметр интервала).
    """
    rec_list = list(records)
    diam_um, cum_pct = compute_cumulative_undersize(rec_list)

    # Суммарная масса
    total_mass_ug = float(sum(float(r.mass) for r in rec_list))
    if total_mass_ug <= 0.0:
        raise ValueError("Суммарная масса должна быть > 0.")

    # -----------------------------
    # 1) Квантили по кумулятиве
    # -----------------------------
    mmad_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=50.0)

    # "Инженерные" квантили 16/84
    d16_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=16.0)
    d84_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=84.0)

    # "Нормальные" квантили для GSD
    d15_87_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=15.87)
    d84_13_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=84.13)

    if d15_87_um <= 0.0:
        raise ValueError("d15.87 получился <= 0, невозможно вычислить GSD.")
    gsd = float(np.sqrt(d84_13_um / d15_87_um))

    # D10/D90 и Span
    d10_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=10.0)
    d90_um = interpolate_d_at_cum_pct_logx(diam_um, cum_pct, target_pct=90.0)

    if mmad_um <= 0.0:
        raise ValueError("MMAD получился <= 0, невозможно вычислить Span.")
    span = float((d90_um - d10_um) / mmad_um)

    # FPF для порога 5 µm
    fpf_cutoff_um = 5.0
    fpf_pct = interpolate_cum_pct_at_d_logx(diam_um, cum_pct, d_um=fpf_cutoff_um)

    # ---------------------------------------------------
    # 2) Массо-взвешенные средние и мода по интервалам
    # ---------------------------------------------------
    # Важно: эти величины считаем по интервалам ступеней, а не по кумулятиве.
    # Для каждой ступени нужен d_low, d_high и масса.
    dg_bins: List[float] = []
    w_bins: List[float] = []

    for r in rec_list:
        d_low = float(getattr(r, "d_low", 0.0))
        d_high = float(getattr(r, "d_high", 0.0))
        m = float(getattr(r, "mass", 0.0))

        # Пропускаем некорректные интервалы и нулевую массу
        if m <= 0.0:
            continue
        if d_low <= 0.0 or d_high <= 0.0 or d_high <= d_low:
            continue

        # Репрезентативный диаметр интервала: геометрический центр
        d_bin = float(np.sqrt(d_low * d_high))
        dg_bins.append(d_bin)
        w_bins.append(m / total_mass_ug)

    # Значения по умолчанию, если не удалось собрать интервалы
    log_mean_um = float("nan")
    mass_mean_um = float("nan")
    modal_um = float("nan")

    if dg_bins:
        d_arr = np.array(dg_bins, dtype=float)
        w_arr = np.array(w_bins, dtype=float)

        # Нормировка весов на случай численных отклонений
        w_sum = float(np.sum(w_arr))
        if w_sum > 0.0:
            w_arr = w_arr / w_sum

        # Логарифмический средний диаметр (геометрическое среднее)
        # d_g = exp(Σ w_i ln(d_i))
        log_mean_um = float(np.exp(np.sum(w_arr * np.log(d_arr))))

        # Среднемассовый (массо-взвешенный арифметический) диаметр
        # d_mass = Σ w_i d_i
        mass_mean_um = float(np.sum(w_arr * d_arr))

        # Модальный диаметр (диаметр интервала с максимальной долей массы)
        modal_um = float(d_arr[int(np.argmax(w_arr))])

    # ---------------------------------------------------
    # 3) Возвращаем результат (добавьте поля в MmadResult)
    # ---------------------------------------------------
    return MmadResult(
        mmad_um=float(mmad_um),
        gsd=float(gsd),
        d10_um=float(d10_um),
        d16_um=float(d16_um),
        d84_um=float(d84_um),
        d90_um=float(d90_um),
        d15_87_um=float(d15_87_um),
        d84_13_um=float(d84_13_um),
        span=float(span),
        fpf_cutoff_um=float(fpf_cutoff_um),
        fpf_pct=float(fpf_pct),
        total_mass_ug=float(total_mass_ug),
        log_mean_um=float(log_mean_um),
        mass_mean_um=float(mass_mean_um),
        modal_um=float(modal_um),
        diam_um=diam_um,
        cum_undersize_pct=cum_pct,
    )