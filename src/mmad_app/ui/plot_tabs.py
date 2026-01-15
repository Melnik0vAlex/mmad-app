# -*- coding: utf-8 -*-
# src/mmad_app/ui/plot_tabs.py
"""
Вкладки с графиками (QTabWidget):
1) Cumulative undersize (обычный график)
2) Log–probit (z+5 против log10(d))
3) Barplot (масса по ступеням)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from scipy.stats import norm

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import MmadResult
from mmad_app.ui.plot_widget import PlotWidget


@dataclass(frozen=True)
class ProbitLine:
    """Параметры линии пробит-регрессии: probit = a*log10(d) + b."""

    a: float
    b: float
    r2: float


def _safe_clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Ограничение вероятностей, чтобы norm.ppf не дал +/-inf."""
    return np.clip(p, eps, 1.0 - eps)


def fit_probit(d_um: np.ndarray, cum_pct: np.ndarray) -> ProbitLine:
    """
    Оцениваем линейную модель для пробит-графика:
        z = Phi^{-1}(p)
        probit = z + 5
        probit = a*log10(d) + b

    Возвращает a, b и R^2.
    """
    x = np.asarray(d_um, dtype=float)
    y = np.asarray(cum_pct, dtype=float)

    mask = (x > 0.0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    p = _safe_clip_prob(y / 100.0)
    z = norm.ppf(p)
    probit = z + 5.0

    lx = np.log10(x)

    a, b = np.polyfit(lx, probit, deg=1)

    y_hat = a * lx + b
    ss_res = float(np.sum((probit - y_hat) ** 2))
    ss_tot = float(np.sum((probit - float(np.mean(probit))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ProbitLine(a=float(a), b=float(b), r2=float(r2))


class PlotTabs(QWidget):
    """Контейнер вкладок с графиками."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.tabs = QTabWidget()
        self.tab_cum = PlotWidget()
        self.tab_probit = PlotWidget()
        self.tab_bar = PlotWidget()
        self.tab_mass_density_distribution = PlotWidget()

        self.tabs.addTab(self.tab_cum, "Накопительная кривая")
        self.tabs.addTab(self.tab_probit, "Лог-пробит")
        self.tabs.addTab(self.tab_bar, "Столбчатая диаграмма")
        self.tabs.addTab(self.tab_mass_density_distribution, "Плотность распределения")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Пустые шаблоны
        self.tab_cum.plot_empty()
        self.tab_probit.plot_empty()
        self.tab_bar.plot_empty()
        self.tab_mass_density_distribution.plot_empty()


    def plot_all(self, records: List[StageRecord], result: MmadResult) -> None:
        """
        Рисует все графики.

        records нужны для barplot (масса по ступеням),
        result содержит cumulative и MMAD/GSD/и т.д.
        """
        self._plot_cumulative(result)
        self._plot_log_probit(result)
        self._plot_bar(records)
        self._plot_mass_density_distribution(records)


    def _plot_cumulative(self, result: MmadResult) -> None:
        """Обычный cumulative-график"""
        self.tab_cum.plot_cumulative(
            diam_um=result.diam_um,
            cum_pct=result.cum_undersize_pct,
            mmad_um=result.mmad_um
        )

    def _plot_log_probit(self, result: MmadResult) -> None:
        """
        Пробит-график:
            X = log10(d)
            Y = probit = norm.ppf(p) + 5
        """
        d = np.asarray(result.diam_um, dtype=float)
        cum = np.asarray(result.cum_undersize_pct, dtype=float)

        mask = d > 0.0
        d = d[mask]
        cum = cum[mask]

        p = _safe_clip_prob(cum / 100.0)
        probit = norm.ppf(p) + 5.0
        lx = np.log10(d)

        fit = fit_probit(d, cum)

        lx_line = np.linspace(float(np.min(lx)), float(np.max(lx)), 200)
        probit_line = fit.a * lx_line + fit.b

        # Используем внутренний canvas PlotWidget: просто перерисуем "вручную"
        fig = self.tab_probit._figure  # noqa: SLF001 (для простоты обучения)
        canvas = self.tab_probit._canvas  # noqa: SLF001

        fig.clear()
        ax = fig.add_subplot(111)

        ax.plot(lx, probit, marker="o", linestyle="None", label="Data (probit)")
        ax.plot(lx_line, probit_line, linestyle="--", label=f"Fit (R²={fit.r2:.3f})")

        ax.set_xlabel("log10(d), d в µm")
        ax.set_ylabel("Probit = Φ⁻¹(p) + 5")
        ax.set_title("Log–probit представление (линейная аппроксимация)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(loc="best")

        fig.tight_layout()
        canvas.draw()

    def _plot_bar(self, records: List[StageRecord]) -> None:
        """
        Строит распределение массы по размерам через средние геометрические диаметры интервалов.
        """
        centers_um: List[float] = []
        masses_ug: List[float] = []
        labels: List[str] = []

        for stage in records:
            # Центр интервала по геометрическому среднему
            d_g = float(np.sqrt(float(stage.d_low) * float(stage.d_high)))

            # Масса по названию (если пользователь не ввёл — считаем 0)
            m = float(stage.mass)

            centers_um.append(d_g)
            masses_ug.append(m)
            labels.append(f"{d_g:g}")  # подпись тика = центр интервала

            # 3) Упорядочивание по размеру
            order = np.argsort(np.array(centers_um, dtype=float))
            centers_um = [centers_um[i] for i in order]
            masses_ug = [masses_ug[i] for i in order]
            labels = [labels[i] for i in order]

        # 4) Отрисовка
        fig = self.tab_bar._figure  # noqa: SLF001
        canvas = self.tab_bar._canvas  # noqa: SLF001

        fig.clear()
        ax = fig.add_subplot(111)
        x_pos = np.arange(len(centers_um), dtype=int)
        ax.bar(x_pos, np.array(masses_ug, dtype=float))
        ax.set_xticks(x_pos)
        formatted_labels = [f"{float(x):.2f}" for x in labels]
        ax.set_xticklabels(formatted_labels)
        ax.set_xlabel("dср, мкм")
        ax.set_ylabel("Масса, мкг")
        ax.set_title("Распределение размеров частиц аэрозоля по массе")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
        fig.tight_layout()
        canvas.draw()

"""
    def _plot_mass_density_distribution(self, records: List[StageRecord]) -> None:

        dg_list = []
        mass_list = []
        dln_list = []

        for stage in records:
            d_low = float(stage.d_low)
            d_high = float(stage.d_high)
            m = float(stage.mass)

            if d_low <= 0.0 or d_high <= 0.0 or d_high <= d_low:
                # Некорректные интервалы лучше явно отбрасывать
                continue

            d_g = float(np.sqrt(d_low * d_high))
            d_ln = float(np.log(d_high / d_low))

            if d_ln <= 0.0:
                continue

            dg_list.append(d_g)
            dln_list.append(d_ln)
            mass_list.append(m)

        fig = self.tab_mass_density_distribution._figure  # noqa: SLF001
        canvas = self.tab_mass_density_distribution._canvas  # noqa: SLF001

        fig.clear()
        ax = fig.add_subplot(111)

        if len(mass_list) < 2:
            ax.set_title("Недостаточно корректных интервалов для построения графика")
            ax.set_xlabel("ln(dg / µm)")
            ax.set_ylabel("(m/M) / Δln(d)")
            ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
            fig.tight_layout()
            canvas.draw()
            return

        mass_total = float(np.sum(mass_list))
        if mass_total <= 0.0:
            ax.set_title("Суммарная масса <= 0: невозможно нормировать")
            ax.set_xlabel("ln(dg / µm)")
            ax.set_ylabel("(m/M) / Δln(d)")
            ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
            fig.tight_layout()
            canvas.draw()
            return

        dg = np.array(dg_list, dtype=float)
        dln = np.array(dln_list, dtype=float)
        mass = np.array(mass_list, dtype=float)

        # Нормированная плотность по ln(d): интеграл по ln(d) равен 1
        dens_norm = (mass / mass_total) / dln

        # Ось X: ln(dg / µm)
        x = np.log(dg)

        # Сортировка по X для красивой линии
        order = np.argsort(x)
        x = x[order]
        dens_norm = dens_norm[order]
        dln = dln[order]
        area = float(np.sum(dens_norm * dln))
        print("Площадь по ln(d):", area)

        ax.bar(
            x,
            dens_norm,
            width=dln,
            align="center",
            alpha=0.7,
            edgecolor="black",
        )

        ax.set_xlabel("ln(dg / µm)")
        ax.set_ylabel("(m/M) / Δln(d)")
        ax.set_title("Нормированная массовая плотность по ln(d) (по интервалам ступеней)")
        ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)

        fig.tight_layout()
        canvas.draw()
"""

