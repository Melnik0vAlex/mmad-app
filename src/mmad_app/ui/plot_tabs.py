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

from mmad_app.core.models import StageRecord, MmadResult
from mmad_app.ui.plot_widget import PlotWidget


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
        self._plot_cumulative(result)
        self._plot_log_probit(result)
        self._plot_bar(records)
        self._plot_mass_density_distribution(records, result)

    def _plot_cumulative(self, result: MmadResult) -> None:
        self.tab_cum.plot_cumulative(
            result.diam_um,
            result.cum_undersize_pct,
            result.mmad_um
        )

    def _plot_log_probit(self, result: MmadResult) -> None:
        self.tab_probit.plot_probit(
            result.diam_um,
            result.cum_undersize_pct
        )

    def _plot_bar(self, records: List[StageRecord]) -> None:
        centers = [np.sqrt(r.d_low * r.d_high) for r in records]
        masses = [r.mass for r in records]
        self.tab_bar.plot_bar(
            centers,
            masses
        )

    def _plot_mass_density_distribution(
        self, records: List[StageRecord], result: MmadResult
    ) -> None:
        d_low = [r.d_low for r in records]
        d_high = [r.d_high for r in records]
        mass = [r.mass for r in records]
        self.tab_mass_density_distribution.plot_density(
            d_low,
            d_high,
            mass,
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
