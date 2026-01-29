# -*- coding: utf-8 -*-
# src/mmad_app/ui/plot_widget.py
"""
Виджет графика (matplotlib).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm
from PySide6.QtWidgets import QVBoxLayout, QWidget

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

from mmad_app.core.models import ProbitLine
from mmad_app.core.probit import clip_prob, fit_probit


class PlotWidget(QWidget):
    """Qt-виджет, содержащий matplotlib canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._figure = Figure()
        self._canvas = FigureCanvas(self._figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas, stretch=1)

        self.plot_empty()

    def new_axes(self) -> Axes:
        """Очищает фигуру и возвращает новую ось (subplot 1x1)."""
        self._figure.clear()
        return self._figure.add_subplot(111)

    def redraw(self, *, tight: bool = True) -> None:
        """Перерисовывает canvas."""
        if tight:
            self._figure.tight_layout()
        self._canvas.draw()

    def apply_default_style(self, ax: Axes) -> None:
        """Единый стиль делений и сетки."""
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.5)

    def plot_empty(
        self,
        *,
        title: str = "Нет данных",
        xlabel: str = "",
        ylabel: str = "",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        grid: bool = True,
        use_default_style: bool = False,
    ) -> None:
        """
        Универсальная "заглушка" для всех графиков.

        Параметры
        ---------
        title:
            Заголовок графика.
        xlabel, ylabel:
            Подписи осей.
        xlim, ylim:
            Пределы осей (если None — matplotlib выберет автоматически).
        grid:
            Отрисовка сетки.
        use_default_style:
            Стиль по умолчанию (тики/сетка).
        """
        ax = self.new_axes()

        if title:
            ax.set_title(title, wrap=True)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if use_default_style:
            self.apply_default_style(ax)

        if grid:
            ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)

        self.redraw()

    def plot_cumulative(
        self,
        diam_um: np.ndarray,
        cum_pct: np.ndarray,
        mmad_um: float,
        *,
        d16_um: float | None = None,
        d84_um: float | None = None,
    ) -> None:
        """
        Отрисовывает накопленную массовую долю частиц с диаметром меньше d.

        Параметры
        ---------
        diam_um:
            Массив диаметров (мкм), соответствующих точкам кумулятивной кривой.
            Для корректного отображения ожидаются значения > 0 и без NaN/inf.
        cum_pct:
            Массив значений F(d) в процентах (0..100), той же длины что diam_um.
        mmad_um:
            Диаметр MMAD (d50) в мкм, используется для вертикальной линии.
        d16_um:
            Диаметр d16 (16-й процентиль по массе), мкм. Если None, линия не рисуется.
        d84_um:
            Диаметр d84 (84-й процентиль по массе), мкм. Если None, линия не рисуется.

        Возвращает
        ----------
        None
            Функция обновляет содержимое matplotlib-холста внутри виджета.

        """
        ax = self.new_axes()

        # 1) Основная кривая
        ax.plot(
            diam_um,
            cum_pct,
            marker="s",
            markersize=6,
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.8,
            linewidth=1.6,
            label="Накопительная кривая",
        )

        ax.set_title(
            "Накопительная кривая массового распределения "
            "по аэродинамическому диаметру",
            wrap=True,
        )
        ax.set_xlabel("Аэродинамический диаметр, мкм")
        ax.set_ylabel("Накопленная масса < dв, %")

        # 2) Пределы по Y
        ax.set_ylim(-2.0, 102.0)

        # 3) Пределы по X: только положительные диаметры + небольшой padding
        x_pos = np.asarray(diam_um, dtype=float)
        x_pos = x_pos[np.isfinite(x_pos) & (x_pos > 0.0)]

        if x_pos.size > 0:
            x_min = float(np.min(x_pos))
            x_max = float(np.max(x_pos))
            pad = max(0.05 * (x_max - x_min), 0.2)
            ax.set_xlim(max(0.0, x_min - pad), x_max + pad)
        else:
            ax.set_xlim(0.0, 10.0)

        # 4) Тики: только "основные" (без minor)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MultipleLocator(10))

        # 5) Сетка
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)

        # 6) Горизонтальные уровни 16/50/84% + подписи справа
        def _annotate_hline(y: float, text: str) -> None:
            """Подпись горизонтальной линии справа (слегка за правой границей)."""
            x_right = ax.get_xlim()[1]
            ax.text(
                x_right,
                y,
                f"  {text}",
                va="center",
                ha="left",
                fontsize=9,
                color="brown",
                clip_on=False,
            )

        for p in (16.0, 50.0, 84.0):
            ax.axhline(p, linestyle="--", linewidth=1.2, color="brown", alpha=0.9)

        # Подписи уровней
        if d16_um is not None and np.isfinite(float(d16_um)) and float(d16_um) > 0.0:
            _annotate_hline(16.0, f"16% (d16={float(d16_um):.2f} µm)")
        else:
            _annotate_hline(16.0, "16%")

        _annotate_hline(50.0, "50%")

        if d84_um is not None and np.isfinite(float(d84_um)) and float(d84_um) > 0.0:
            _annotate_hline(84.0, f"84% (d84={float(d84_um):.2f} µm)")
        else:
            _annotate_hline(84.0, "84%")

        # 7) Вертикали: MMAD
        ax.axvline(
            mmad_um,
            linestyle="--",
            linewidth=1.4,
            color="red",
            label=f"MMAD = {mmad_um:.2f} мкм",
        )

        if d16_um is not None and np.isfinite(float(d16_um)) and float(d16_um) > 0.0:
            ax.axvline(
                float(d16_um), linestyle=":", linewidth=1.2, color="brown", alpha=0.9
            )

        if d84_um is not None and np.isfinite(float(d84_um)) and float(d84_um) > 0.0:
            ax.axvline(
                float(d84_um), linestyle=":", linewidth=1.2, color="brown", alpha=0.9
            )

        # 8) Легенда и отрисовка
        ax.legend(loc="lower right", frameon=True)

        # Если подписи выходят за пределы, tight_layout можно чуть ужать справа
        self.redraw(tight=True)

    def plot_probit(
        self,
        diam_um: np.ndarray,
        cum_pct: np.ndarray,
        *,
        title: str = "Лог-пробит распределение накопленной массы",
        use_log10_x: bool = True,
        clip_eps: float = 1e-6,
    ) -> ProbitLine:
        """
        Отрисовывает log–probit график накопленного массового распределения.

        Параметры
        ---------
        diam_um:
            Аэродинамические диаметры частиц, мкм. Используются только значения d > 0.
        cum_pct:
            Накопленная доля массы частиц с диаметром < d, выраженная в процентах.
        use_log10_x:
            Если True, ось X задаётся как log10(d).
        clip_eps:
            Малое положительное число для ограничения вероятностей p в диапазоне
            (clip_eps, 1 − clip_eps) с целью предотвращения ±∞ при вычислении Φ⁻¹(p).

        Возвращает
        ----------
        ProbitLine
            Параметры пробит-линейной аппроксимации.
        """
        ax = self.new_axes()

        d = np.asarray(diam_um, dtype=float)
        y = np.asarray(cum_pct, dtype=float)

        mask = (
            np.isfinite(d)
            & np.isfinite(y)
            & (d > 0.0)
            & (y > 0.0)
            & (y < 100.0)
        )

        d = d[mask]
        y = y[mask]

        if d.size < 2:
            ax.set_title("Недостаточно точек для probit-графика")
            ax.set_xlabel("log10(d), мкм" if use_log10_x else "d, мкм")
            ax.set_ylabel("Probit = Φ⁻¹(p) + 5")
            self.apply_default_style(ax)
            self.redraw()
            return ProbitLine(a=float("nan"), b=float("nan"), rmse=float("nan"))

        p = clip_prob(y / 100.0, eps=clip_eps)
        probit = norm.ppf(p) + 5.0

        if use_log10_x:
            x = np.log10(d)
            x_label = "log10(d), мкм"
        else:
            x = d
            x_label = "d, мкм"

        # Фит выполняем по исходным d и cum (%)
        fit = fit_probit(d, y)

        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        # Линия фит-предсказания в координате log10(d)
        if use_log10_x:
            y_line = fit.a * x_line + fit.b
        else:
            # Если рисуем по d, всё равно фит построен по log10(d),
            y_line = fit.a * np.log10(np.maximum(x_line, 1e-12)) + fit.b

        ax.plot(
            x,
            probit,
            linestyle="None",
            marker="s",
            markersize=6,
            markerfacecolor="orange",
            markeredgecolor="black",
            markeredgewidth=0.8,
            linewidth=1.6,
            label="Экспериментальные значения",
        )
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            color="tab:blue",
            label=f"Линейная аппроксимация (RMSE = {fit.rmse:.2f})",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("Probit = Φ⁻¹(p) + 5")
        ax.set_title(title, wrap=True)
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(loc="lower right", frameon=True)

        self.redraw()
        return fit

    def plot_bar(
        self,
        centers_um: Sequence[float],
        masses_ug: Sequence[float],
        *,
        title: str = "Распределение размеров частиц по массе",
        xlabel: str = "Средний геометрический диаметр ступени, мкм",
        ylabel: str = "Масса, мкг",
        fmt_ticks: str = "{:.2f}",
    ) -> None:
        """Рисует barplot: масса по бинам (центрам интервалов)."""
        ax = self.new_axes()

        x = np.asarray(centers_um, dtype=float)
        y = np.asarray(masses_ug, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x.size == 0:
            ax.set_title("Нет данных для построения")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
            self.redraw()
            return

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_pos = np.arange(x.size, dtype=int)
        ax.bar(
            x=x_pos,
            height=y,
            width=0.8,
            align="center",
            alpha=0.6,
            edgecolor="black",
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([fmt_ticks.format(v) for v in x])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, wrap=True)
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)

        self.redraw()

    def plot_density(
        self,
        d_low_um: Sequence[float],
        d_high_um: Sequence[float],
        mass_ug: Sequence[float],
        *,
        title: str = "Дифференциальное массовое распределение по логарифму диаметра",
        show_model: bool = True,
        model_source: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """
        Отрисовывает дифференциальное массовое распределение по ln(d).

        Поверх экспериментальных столбцов наносится логнормальная
        аппроксимация, восстановленная из лог-пробит подгонки накопленного
        распределения F(d) (cumulative undersize).

        Параметры
        ---------
        d_low_um, d_high_um:
            Нижняя и верхняя границы интервалов ступеней (мкм).
        mass_ug:
            Массы, соответствующие интервалам (мкг).
        title:
            Заголовок графика.
        show_model:
            Если True и задан model_source, отображает логнормальную модель.
        model_source:
            Пара (diam_um, cum_pct) для подгонки лог-пробит и восстановления
            параметров логнормальной модели. diam_um в мкм, cum_pct в %.

        Возвращает
        ----------
        None
            Функция обновляет содержимое matplotlib-холста внутри виджета.

        """
        ax = self.new_axes()

        d_low = np.asarray(d_low_um, dtype=float)
        d_high = np.asarray(d_high_um, dtype=float)
        m = np.asarray(mass_ug, dtype=float)

        # Фильтрация входных данных
        mask = (
            np.isfinite(d_low)
            & np.isfinite(d_high)
            & np.isfinite(m)
            & (d_low > 0.0)
            & (d_high > 0.0)
            & (d_high > d_low)
        )
        d_low = d_low[mask]
        d_high = d_high[mask]
        m = m[mask]

        # Проверка интервалов и суммарной массы
        if m.size < 2:
            ax.set_title("Недостаточно корректных интервалов для построения")
            ax.set_xlabel("ln(d / µm)")
            ax.set_ylabel("(m/M) / Δln(d)")
            ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
            self.redraw()
            return

        mass_total = float(np.sum(m))
        if mass_total <= 0.0:
            ax.set_title("Суммарная масса <= 0: невозможно нормировать")
            ax.set_xlabel("ln(d / µm)")
            ax.set_ylabel("(m/M) / Δln(d)")
            ax.grid(True, which="both", linestyle=":", linewidth=0.8, alpha=0.6)
            self.redraw()
            return

        # Построение экспериментальной плотности по ln(d)
        dln = np.log(d_high / d_low)
        d_g = np.sqrt(d_low * d_high)
        x = np.log(d_g)

        dens_norm = (m / mass_total) / dln

        order = np.argsort(x)
        x = x[order]
        dens_norm = dens_norm[order]
        dln = dln[order]

        ax.bar(
            x,
            dens_norm,
            width=dln,
            align="center",
            alpha=0.6,
            edgecolor="black",
            label="Экспериментальные данные",
        )

        # Логнормальной модель, восстановленная из log–probit аппроксимации
        # накопленного распределения
        if show_model and model_source is not None:
            diam_um, cum_pct = model_source
            fit = fit_probit(
                np.asarray(diam_um, dtype=float),
                np.asarray(cum_pct, dtype=float),
            )

            a = float(fit.a)
            b = float(fit.b)

            if a > 0.0 and np.isfinite(a) and np.isfinite(b):
                sigma = float(np.log(10.0) / a)
                mu = float(-(b - 5.0) * sigma)

                x_grid = np.linspace(
                    float(np.min(x)) - 0.25, float(np.max(x)) + 0.25, 400
                )
                model = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
                    -0.5 * ((x_grid - mu) / sigma) ** 2
                )
                ax.plot(
                    x_grid,
                    model,
                    linewidth=2.0,
                    label=f"Логнормальная аппроксимация"
                        rf" ($\mu={mu:.2f}$, $\sigma={sigma:.2f}$, RMSE={fit.rmse:.2f})"
                )

        # Ось X: натуральный логарифм аэродинамического диаметра частиц
        ax.set_xlabel("ln(d)")
        # Ось Y: нормированная массовая плотность распределения по логарифму диаметра
        ax.set_ylabel("(Δm/M) / Δln(d)")
        ax.set_title(title, wrap=True)
        ax.grid(True, which="major", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(loc="best")

        self.redraw()

    def save_figure(self, filepath: str | Path, *, dpi: int = 300) -> None:
        """
        Сохраняет текущую фигуру matplotlib в файл.

        Параметры
        ---------
        filepath:
            Путь к файлу (например: "plot.png", "plot.pdf").
        dpi:
            Разрешение для растровых форматов (png, jpg).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # bbox_inches="tight" — чтобы не обрезались подписи/легенда
        self._figure.savefig(path, dpi=dpi, bbox_inches="tight")
