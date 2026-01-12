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
from matplotlib.ticker import AutoMinorLocator, LogLocator, MultipleLocator


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
        ax.set_xlabel("Аэродинамический диаметр, µm (лог шкала)")
        ax.set_ylabel("Накопительная фракция < размера, %")
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.grid(True, which="both")
        # Важно: matplotlib не рисует автоматически.
        # draw() говорит: "перерисуй canvas".
        self._canvas.draw()

    def plot_cumulative(
        self,
        diam_um: np.ndarray,
        cum_pct: np.ndarray,
        mmad_um: float,
        d15_87_um: float | None = None,
        d84_13_um: float | None = None,
        gsd: float | None = None,
    ) -> None:
        """
        Научная визуализация кумулятивной кривой undersize.

        Рисуем:
        - кумулятивную кривую
        - горизонтальные линии уровней 15.87%, 50%, 84.13%
        - вертикальные линии d15.87, MMAD, d84.13
        - аккуратные major/minor ticks и сетку
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        # ---------------------------
        # 1) Основная кривая
        # ---------------------------
        ax.plot(
            diam_um,
            cum_pct,
            marker="o",
            linewidth=1.6,
            label="Cumulative undersize"
        )

        # ---------------------------
        # 2) Оси и пределы
        # ---------------------------
        ax.set_xscale("log")
        ax.set_ylim(0, 100)

        # X-лимиты: берём только положительные диаметры
        x_pos = diam_um[diam_um > 0.0]
        if x_pos.size >= 2:
            x_min = float(np.min(x_pos))
            x_max = float(np.max(x_pos))
            # небольшие поля по краям, чтобы подписи не упирались
            ax.set_xlim(x_min * 0.8, x_max * 1.2)

        ax.set_xlabel("Аэродинамический диаметр, µm (лог шкала)")
        ax.set_ylabel("Накопительная фракция < размера, %")
        ax.set_title("Аэродинамическое распределение по массе (APSD): cumulative undersize")

        # ---------------------------
        # 3) Красивые деления (ticks)
        # ---------------------------
        # Y: основные деления каждые 10%, мелкие — более частые
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        # X: лог-деления
        ax.xaxis.set_major_locator(LogLocator(base=10.0))
        minor_subs = (np.arange(2, 10) * 0.1).tolist()
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=minor_subs))

        # ---------------------------
        # 4) Сетка: major + minor
        # ---------------------------
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.5)

        # ---------------------------
        # 5) Процентильные уровни (горизонтальные линии)
        # ---------------------------
        perc_levels = [15.87, 50.0, 84.13]
        for p in perc_levels:
            ax.axhline(p, linestyle=":", linewidth=1.0)

        # ---------------------------
        # 6) Вертикальные линии: d15/MMAD/d84 + подписи
        # ---------------------------
        ax.axvline(mmad_um, linestyle="--", linewidth=1.4, label=f"MMAD = {mmad_um:.3f} µm")

        if d15_87_um is not None:
            ax.axvline(d15_87_um, linestyle=":", linewidth=1.2, label=f"d15.87 = {d15_87_um:.3f} µm")

        if d84_13_um is not None:
            ax.axvline(d84_13_um, linestyle=":", linewidth=1.2, label=f"d84.13 = {d84_13_um:.3f} µm")

        # Текстовая аннотация GSD (если есть)
        if gsd is not None:
            ax.text(
                0.02,
                0.02,
                f"GSD = {gsd:.3f}",
                transform=ax.transAxes,  # координаты внутри области графика (0..1)
                ha="left",
                va="bottom",
            )

        # ---------------------------
        # 7) Легенда и финальная укладка
        # ---------------------------
        ax.legend(loc="lower right", frameon=True)
        self._figure.tight_layout()
        self._canvas.draw()
