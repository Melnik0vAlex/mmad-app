# -*- coding: utf-8 -*-
# src/mmad_app/ui/plot_tabs.py
"""
Вкладки с графиками (QTabWidget)
"""

from __future__ import annotations

from typing import List
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from mmad_app.core.models import MmadResultLS, StageRecord, MmadResult
from mmad_app.ui.plot_widget import PlotWidget


class PlotTabs(QWidget):
    """Контейнер вкладок с графиками."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.tabs = QTabWidget()
        self.tab_cum = PlotWidget()
        self.tab_probit = PlotWidget()
        self.tab_least_squares = PlotWidget()
        self.tab_bar = PlotWidget()
        self.tab_mass_density_distribution = PlotWidget()

        self.tabs.addTab(self.tab_cum, "Интегральная кривая")
        self.tabs.addTab(self.tab_probit, "Лог-пробит")
        self.tabs.addTab(self.tab_least_squares, "МНК")
        self.tabs.addTab(self.tab_bar, "Столбчатая диаграмма")
        self.tabs.addTab(self.tab_mass_density_distribution, "Плотность распределения")

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)

        # Пустые шаблоны
        self.tab_cum.plot_empty()
        self.tab_probit.plot_empty()
        self.tab_bar.plot_empty()
        self.tab_least_squares.plot_empty()
        self.tab_mass_density_distribution.plot_empty()

    def plot_all(
        self, records: List[StageRecord], result_lp: MmadResult, result_ls: MmadResultLS
    ) -> None:
        self._plot_cumulative(result_lp)
        self._plot_log_probit(result_lp)
        self._plot_least_squares(result_ls)
        self._plot_bar(records)
        self._plot_mass_density_distribution(records, result_lp)

    def _plot_cumulative(self, result: MmadResult) -> None:
        self.tab_cum.plot_cumulative(
            result.diam_um, result.cum_undersize_pct, result.mmad
        )

    def _plot_log_probit(self, result: MmadResult) -> None:
        self.tab_probit.plot_probit(result.diam_um, result.cum_undersize_pct)

    def _plot_least_squares(self, result: MmadResultLS) -> None:
        self.tab_least_squares.plot_least_squares(result.x_erfinv, result.y_ln_d)

    def _plot_bar(self, records: List[StageRecord]) -> None:
        """Строит bar-график масс по ступеням импактора."""

        # Сбор масс краевых фракций
        mass_filter = 0.0
        mass_gt_10 = 0.0

        # Сбор интервалов (ступени 0..7)
        centers: list[float] = []
        masses: list[float] = []
        labels: list[str] = []

        for r in records:
            name = (r.name or "").strip().lower()

            if "фильтр" in name:
                if np.isfinite(r.mass):
                    mass_filter += float(r.mass)
                continue

            # Пресепаратор и входной канал: суммируем в ">10"
            if "пресепаратор" in name or "входной" in name:
                if np.isfinite(r.mass):
                    mass_gt_10 += float(r.mass)
                continue

            # Ступени 0..7
            if r.d_low is None or r.d_high is None:
                continue

            d_low = float(r.d_low)
            d_high = float(r.d_high)
            m = float(r.mass)

            if not (np.isfinite(d_low) and np.isfinite(d_high) and np.isfinite(m)):
                continue
            if m < 0.0:
                continue
            if d_low <= 0.0 or d_high <= d_low:
                continue

            d_g = float(np.sqrt(d_low * d_high))
            centers.append(d_g)
            masses.append(m)
            # Для ступеней 0..7 в подписи передаем число
            labels.append(f"{d_g:.2f}")

        # Сортировка ступеней по центру (по возрастанию диаметра)
        if centers:
            order = np.argsort(np.asarray(centers, dtype=float))
            centers = [centers[i] for i in order]
            masses = [masses[i] for i in order]
            labels = [labels[i] for i in order]

        # Добавляем крайние фракции как отдельные бары
        plot_centers: list[float] = []
        plot_masses: list[float] = []
        plot_labels: list[str] = []

        if mass_filter > 0.0:
            plot_centers.append(0.43)
            plot_masses.append(mass_filter)
            plot_labels.append("<0.43")

        plot_centers.extend(centers)
        plot_masses.extend(masses)
        plot_labels.extend(labels)

        if mass_gt_10 > 0.0:
            plot_centers.append(10.0)
            plot_masses.append(mass_gt_10)
            plot_labels.append(">10")

        self.tab_bar.plot_bar(
            plot_centers,
            plot_masses,
            labels=plot_labels,
        )

    def _plot_mass_density_distribution(
        self, records: List[StageRecord], result: MmadResult
    ) -> None:
        d_low_list: list[float] = []
        d_high_list: list[float] = []
        mass_list: list[float] = []

        for r in records:
            if r.d_low is None or r.d_high is None:
                continue

            d_low = float(r.d_low)
            d_high = float(r.d_high)

            if d_low <= 0.0 or d_high <= d_low:
                continue

            d_low_list.append(d_low)
            d_high_list.append(d_high)
            mass_list.append(float(r.mass))

        self.tab_mass_density_distribution.plot_density(
            d_low_list,
            d_high_list,
            mass_list,
            show_model=True,
            model_source=(result.diam_um, result.cum_undersize_pct),
        )

    def save_current_plot(self, filepath: str | Path, *, dpi: int = 300) -> None:
        current = self.tabs.currentWidget()

        # current должен быть PlotWidget
        if hasattr(current, "save_figure"):
            current.save_figure(filepath, dpi=dpi)  # type: ignore
        else:
            raise TypeError(
                "Текущая вкладка не является PlotWidget и не поддерживает сохранение."
            )
