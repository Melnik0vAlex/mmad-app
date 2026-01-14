# -*- coding: utf-8 -*-
# src/mmad_app/ui/plot_widget.py
"""
Виджет графика (matplotlib) для Qt (PySide6).

Задача:
- встроить matplotlib Figure в Qt-интерфейс
- дать методы для отрисовки результатов
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator


class PlotWidget(QWidget):
    """
    Qt-виджет, содержащий matplotlib canvas.

    Это обёртка, чтобы в main_window.py было просто вызывать:
        plot_widget.plot_cumulative(...)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = Figure()
        self._canvas = FigureCanvas(self._figure)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas)

        self.plot_empty()

    def plot_empty(self) -> None:
        """Рисует пустой шаблон графика."""
        self._figure.clear()
        # add_subplot(111) - создание оси на фигуре
        # 111 читается как: 1 строка, 1 колонка, 1-й график.
        ax = self._figure.add_subplot(111)
        ax.set_xlabel("dв, мкм")
        ax.set_ylabel("Накопительная масса < dв, %")

        ax.set_ylim(0, 100)
        ax.set_xlim(0, 10)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.5)
        # Важно: matplotlib не рисует автоматически.
        # draw() говорит: "перерисуй canvas".
        self._canvas.draw()

    def plot_cumulative(
        self,
        diam_um: np.ndarray,
        cum_pct: np.ndarray,
        mmad_um: float
    ) -> None:

        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # ---------------------------
        # 1) Основная кривая
        # ---------------------------
        ax.plot(
            diam_um,
            cum_pct,
            marker="s",
            markersize=6,               # размер маркера
            markerfacecolor="orange",   # заливка маркера
            markeredgecolor="black",    # обводка маркера
            markeredgewidth=0.8,
            linewidth=1.6,
            label="Cumulative undersize"
        )

        # ---------------------------
        # 2) Оси и пределы
        # ---------------------------
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 10)

        # X-лимиты: берём только положительные диаметры
        x_pos = diam_um[diam_um > 0.0]
        if x_pos.size > 0:
            d_top = float(np.max(x_pos))
            stage_ticks = np.sort(np.unique(x_pos[x_pos < d_top]))
            if stage_ticks.size > 0:
                ax.set_xlabel("Аэродинамический диаметр, мкм")
                ax.set_ylabel("Накопительная масса < dв, %")
                ax.set_title("Распределение накопленной массы аэрозольных частиц по размерам")

        # ---------------------------
        # 3) Красивые деления (ticks)
        # ---------------------------
        # Y: основные деления каждые 10%, мелкие — более частые
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # ---------------------------
        # 4) Сетка: major + minor
        # ---------------------------
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.5)

        # ---------------------------
        # 5) Процентильные уровни (горизонтальные линии)
        # ---------------------------
        perc_levels = [16, 50.0, 84]
        for p in perc_levels:
            ax.axhline(p, linestyle="--", linewidth=1.2, color="brown")

        # ---------------------------
        # 6) Вертикальные линии: d16/MMAD/d84 + подписи
        # ---------------------------
        ax.axvline(mmad_um, linestyle="--", linewidth=1.4, color="red", label=f"MMAD = {mmad_um:.2f} мкм")

        # ---------------------------
        # 7) Легенда и финальная укладка
        # ---------------------------
        ax.legend(loc="lower right", frameon=True)
        self._figure.tight_layout()
        self._canvas.draw()
