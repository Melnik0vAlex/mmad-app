# -*- coding: utf-8 -*-
# ui/results_panel.py

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Union

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel

Number = Union[int, float]
Value = Union[None, Number, str]


class ResultsPanel(QGroupBox):
    """Универсальная карточка отображения результатов расчета."""

    def __init__(
        self,
        title: str = "Результаты",
        rows: Optional[Iterable[Tuple[str, str]]] = None,
        parent=None,
    ) -> None:
        """
        Параметры
        ---------
        title:
            Заголовок метода по которому были получены результаты.
        rows:
            Итерируемое пар (key, caption), где:
            key     — внутренний ключ поля (используется в set_values)
            caption — подпись слева в форме
        """
        super().__init__(parent)
        self.setTitle(title)
        self.setStyleSheet(
            """
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 6px 6px;
                font-size: 16px;
                font-weight: 600;
            }
            """)
        self._layout = QFormLayout(self)
        self._layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self._layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setHorizontalSpacing(16)
        self._layout.setVerticalSpacing(6)

        self._fields: Dict[str, QLabel] = {}

        if rows is not None:
            self.set_rows(rows)

        self.clear()

    def set_rows(self, rows: Iterable[Tuple[str, str]]) -> None:
        """Полностью пересоздаёт строки формы."""
        # Удаляем старые виджеты из layout
        while self._layout.rowCount() > 0:
            self._layout.removeRow(0)

        self._fields.clear()

        for key, caption in rows:
            self._fields[key] = self._add_row(caption)

    def _add_row(self, caption: str) -> QLabel:
        """Добавляет строку и возвращает QLabel значения (правая колонка)."""
        value = QLabel("—")
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._layout.addRow(caption, value)
        return value

    @staticmethod
    def _fmt(value: Value, ndigits: int = 2) -> str:
        """Форматирует число/строку с защитой от None/NaN."""
        if value is None:
            return "—"

        if isinstance(value, str):
            s = value.strip()
            return s if s != "" else "—"

        # Число
        try:
            x = float(value)
            if x != x:  # NaN
                return "—"
        except Exception:
            return "—"

        return f"{x:.{ndigits}f}"

    def clear(self) -> None:
        """Сбрасывает отображение всех полей в '—'."""
        for lbl in self._fields.values():
            lbl.setText("—")

    def set_values(self, values: Dict[str, Value], ndigits: int = 2) -> None:
        """Обновляет значения по ключам."""
        for key, lbl in self._fields.items():
            lbl.setText(self._fmt(values.get(key), ndigits))

    def set_result(self, result: Any, ndigits: int = 2) -> None:
        """Читает значения через getattr(result, key)."""
        values = {key: getattr(result, key, None) for key in self._fields}
        self.set_values(values, ndigits=ndigits)
