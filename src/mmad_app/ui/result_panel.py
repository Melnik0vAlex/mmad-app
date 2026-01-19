# -*- coding: utf-8 -*-
# ui/results_panel.py

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel
from PySide6.QtGui import QFont


class ResultsPanel(QGroupBox):
    """
    Карточка "Результаты" для отображения рассчитанных метрик APSD.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._layout = QFormLayout(self)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setHorizontalSpacing(16)
        self._layout.setVerticalSpacing(6)

        # Создаем поля (Label справа)
        self.lbl_mmad = self._add_row("MMAD (d50), мкм")
        self.lbl_gsd = self._add_row("GSD")
        self.lbl_d10 = self._add_row("d10, мкм")
        self.lbl_d16 = self._add_row("d16, мкм")
        self.lbl_d84 = self._add_row("d84, мкм")
        self.lbl_d90 = self._add_row("d90, мкм")
        self.lbl_span = self._add_row("Span")
        self.lbl_fpf = self._add_row("FPF (< 5 мкм), %")

        # Дополнительные метрики (если вы добавляли в compute_mmad)
        self.lbl_log_mean = self._add_row("Лог. средний диаметр, мкм")
        self.lbl_mass_mean = self._add_row("Среднемассовый диаметр, мкм")
        self.lbl_modal = self._add_row("Модальный диаметр, мкм")

        self.clear()

    def _add_row(self, name: str) -> QLabel:
        """Добавляет строку в форму и возвращает QLabel значения."""
        value = QLabel("—")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # value.setTextInteractionFlags(Qt.TextFla)
        self._layout.addRow(name, value)
        return value

    @staticmethod
    def _fmt(x: Optional[float], ndigits: int = 2) -> str:
        """Формат числа с защитой от None/NaN."""
        if x is None:
            return "—"
        try:
            if x != x:  # NaN
                return "—"
        except Exception:
            return "—"
        return f"{float(x):.{ndigits}f}"

    def clear(self) -> None:
        """Сбрасывает отображение результатов."""
        for w in (
            self.lbl_mmad,
            self.lbl_gsd,
            self.lbl_d10,
            self.lbl_d16,
            self.lbl_d84,
            self.lbl_d90,
            self.lbl_span,
            self.lbl_fpf,
            self.lbl_log_mean,
            self.lbl_mass_mean,
            self.lbl_modal,
        ):
            w.setText("—")

    def set_result(self, result) -> None:
        """
        Заполняет карточку данными результата.

        Ожидается объект MmadResult с полями:
        mmad_um, gsd, d10_um, d16_um, d84_um, d90_um, span, fpf_pct
        + опционально log_mean_um, mass_mean_um, modal_um
        """
        # Базовые
        self.lbl_mmad.setText(self._fmt(getattr(result, "mmad_um", None), 2))
        self.lbl_gsd.setText(self._fmt(getattr(result, "gsd", None), 2))

        self.lbl_d10.setText(self._fmt(getattr(result, "d10_um", None), 2))
        self.lbl_d16.setText(self._fmt(getattr(result, "d16_um", None), 2))
        self.lbl_d84.setText(self._fmt(getattr(result, "d84_um", None), 2))
        self.lbl_d90.setText(self._fmt(getattr(result, "d90_um", None), 2))

        self.lbl_span.setText(self._fmt(getattr(result, "span", None), 2))
        self.lbl_fpf.setText(self._fmt(getattr(result, "fpf_pct", None), 2))

        # Дополнительные (могут быть NaN, если интервалы не заданы)
        self.lbl_log_mean.setText(self._fmt(getattr(result, "log_mean_um", None), 2))
        self.lbl_mass_mean.setText(self._fmt(getattr(result, "mass_mean_um", None), 2))
        self.lbl_modal.setText(self._fmt(getattr(result, "modal_um", None), 2))