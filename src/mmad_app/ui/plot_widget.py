# -*- coding: utf-8 -*-
# src/mmad_app/ui/plot_widget.py
"""
Виджет графика (matplotlib) для Qt (PySide6).
"""

from __future__ import annotations

from pathlib import Path

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
        layout.addWidget(self._canvas, stretch=1)

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

        # 1) Основная кривая
        ax.plot(
            diam_um,
            cum_pct,
            marker="s",
            markersize=6,               # размер маркера
            markerfacecolor="orange",   # заливка маркера
            markeredgecolor="black",    # обводка маркера
            markeredgewidth=0.8,
            linewidth=1.6,
            label="Накопительная кривая"
        )

        ax.set_title("Накопительная кривая массового распределения по аэродинамическому диаметру", wrap=True)

        # 2) Оси и пределы
        ax.set_ylim(0, 102)

        # X-лимиты: берём только положительные диаметры
        x_pos = diam_um[diam_um > 0.0]
        if x_pos.size > 0:
            x_min = float(np.min(x_pos))
            x_max = float(np.max(x_pos))

            stage_ticks = np.sort(np.unique(x_pos[x_pos < x_max]))
            if stage_ticks.size > 0:
                ax.set_xlabel("Аэродинамический диаметр, мкм")
                ax.set_ylabel("Накопленная масса < dв, %")

            # Отступ 5% диапазона, но не меньше небольшой константы
            pad = max(0.05 * (x_max - x_min), 0.2)
            ax.set_xlim(max(0.0, x_min - pad), x_max + pad)

        # 3) Красивые деления (ticks)
        # Y: основные деления каждые 10%, мелкие — более частые
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        # 4) Сетка: major + minor
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.5)

        # 5) Процентильные уровни (горизонтальные линии)
        def _annotate_hline(y: float, text: str) -> None:
            """Подпись горизонтальной линии справа."""
            x_right = ax.get_xlim()[1]
            ax.text(
                x_right,
                y,
                f"  {text}",
                va="center",
                ha="left",
                fontsize=9,
                color="brown",
                clip_on=False,  # чтобы подпись могла выходить за пределы осей
            )

        perc_levels = [16, 50.0, 84]
        for p in perc_levels:
            ax.axhline(p, linestyle="--", linewidth=1.2, color="brown")

        _annotate_hline(16.0, "16 %")
        _annotate_hline(84.0, "84 %")
        _annotate_hline(50, "50 %")

        # 6) Вертикальные линии: d16/MMAD/d84 + подписи
        ax.axvline(mmad_um, linestyle="--", linewidth=1.4, color="red", label=f"MMAD = {mmad_um:.2f} мкм")

        # 7) Легенда и финальная укладка
        ax.legend(loc="lower right", frameon=True)
        self._figure.tight_layout()
        self._canvas.draw()


    def save_figure(self, filepath: str | Path, *, dpi: int = 300) -> None:
        """
        Сохраняет текущую фигуру matplotlib в файл.

        Parameters
        ----------
        filepath:
            Путь к файлу (например: "plot.png", "plot.pdf").
        dpi:
            Разрешение для растровых форматов (png, jpg).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # bbox_inches="tight" — чтобы не обрезались подписи/легенда
        self._figure.savefig(path, dpi=dpi, bbox_inches="tight")