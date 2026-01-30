# -*- coding: utf-8 -*-
# src/mmad_app/ui/db_panel.py
"""Виджет вкладки "База данных": список сохранённых расчётов (SQLite)."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mmad_app.db.repo import delete_run, list_runs


@dataclass(frozen=True)
class RunRow:
    """Упрощённое представление записи из таблицы runs для отображения в UI."""

    run_id: int
    created_at: str
    sample_code: str
    mmad: float
    gsd: float
    fpf_pct: float


class DbHistoryPanel(QWidget):
    """Панель истории расчётов."""

    load_requested = Signal(int)

    def __init__(self, conn: sqlite3.Connection, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._conn = conn

        self._rows: List[RunRow] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ---- Верхняя панель: поиск + кнопки ----
        top = QHBoxLayout()
        top.setSpacing(8)
        root.addLayout(top)

        top.addWidget(QLabel("Поиск (шифр пробы):"))

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Например: A-001")
        self.search_edit.textChanged.connect(self._apply_filter)
        top.addWidget(self.search_edit, stretch=1)

        self.btn_reload = QPushButton("Обновить")
        self.btn_reload.clicked.connect(self.reload)
        top.addWidget(self.btn_reload)

        self.btn_load = QPushButton("Загрузить")
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_load.setEnabled(False)
        top.addWidget(self.btn_load)

        self.btn_delete = QPushButton("Удалить")
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_delete.setEnabled(False)
        top.addWidget(self.btn_delete)

        # ---- Таблица ----
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Дата", "Шифр пробы", "MMAD, µm", "GSD", "FPF, %"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        # Растягиваем некоторые колонки
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellDoubleClicked.connect(self._on_double_click_row)

        root.addWidget(self.table, stretch=1)

        # ---- Подсказка снизу ----
        self.hint = QLabel("Двойной клик по строке — загрузить запись.")
        self.hint.setStyleSheet("color: #777;")
        root.addWidget(self.hint)

    # Публичный метод: перезагрузка данных
    def reload(self) -> None:
        """Загружает последние записи из БД и обновляет таблицу."""
        rows = list_runs(self._conn, limit=200)

        self._rows = [
            RunRow(
                run_id=int(r["id"]),
                created_at=str(r["created_at"]),
                sample_code=str(r["sample_code"]),
                mmad=float(r["mmad"]),
                gsd=float(r["gsd"]),
                fpf_pct=float(r["fpf_pct"]),
            )
            for r in rows
        ]

        self._render_table(self._rows)


    # Внутреннее: отрисовка таблицы
    def _render_table(self, rows: List[RunRow]) -> None:
        """Заполняет QTableWidget данными."""
        self.table.setRowCount(0)

        for row_idx, r in enumerate(rows):
            self.table.insertRow(row_idx)

            # ID
            self._set_item(row_idx, 0, str(r.run_id), align_right=True)

            # Дата
            self._set_item(row_idx, 1, r.created_at)

            # Шифр пробы
            self._set_item(row_idx, 2, r.sample_code)

            # MMAD
            self._set_item(row_idx, 3, f"{r.mmad:.3f}", align_right=True)

            # GSD
            self._set_item(row_idx, 4, f"{r.gsd:.3f}", align_right=True)

            # FPF
            self._set_item(row_idx, 5, f"{r.fpf_pct:.2f}", align_right=True)

        self.table.resizeColumnsToContents()
        self._on_selection_changed()
        self._apply_filter(self.search_edit.text())

    def _set_item(
        self, row: int, col: int, text: str, *, align_right: bool = False
    ) -> None:
        """Утилита для добавления текста в ячейку с выравниванием."""
        item = QTableWidgetItem(text)
        if align_right:
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
        else:
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
        self.table.setItem(row, col, item)

    # Выбор/фильтр/действия пользователя
    def _on_selection_changed(self) -> None:
        """Включает/выключает кнопки действий в зависимости от выбора строки."""
        has_selection = bool(self.table.selectionModel().selectedRows())
        self.btn_load.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)

    def _get_selected_run_id(self) -> Optional[int]:
        """Возвращает run_id выбранной строки или None."""
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None

        row_idx = selected[0].row()
        item = self.table.item(row_idx, 0)
        if item is None:
            return None

        try:
            return int(item.text())
        except ValueError:
            return None

    def _on_double_click_row(self, row: int, col: int) -> None:
        """Двойной клик по строке — загрузить."""
        run_id = self._get_selected_run_id()
        if run_id is not None:
            self.load_requested.emit(run_id)

    def _on_load_clicked(self) -> None:
        """Кнопка 'Загрузить'."""
        run_id = self._get_selected_run_id()
        if run_id is not None:
            self.load_requested.emit(run_id)

    def _on_delete_clicked(self) -> None:
        """Кнопка 'Удалить' (с подтверждением)."""
        run_id = self._get_selected_run_id()
        if run_id is None:
            return

        answer = QMessageBox.question(
            self,
            "Удаление записи",
            f"Удалить запись ID={run_id}?\n"
            "Будут удалены и связанные данные ступеней.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            delete_run(self._conn, run_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        # Обновим таблицу и очистим выбор
        self.reload()

    def _apply_filter(self, text: str) -> None:
        """
        Фильтрует таблицу по sample_code (простая подстрока, без SQL).

        Реализовано на уровне UI: скрываем строки, которые не подходят.
        """
        query = (text or "").strip().lower()

        for i in range(self.table.rowCount()):
            item = self.table.item(i, 2)  # column "Шифр пробы"
            code = item.text().lower() if item else ""
            visible = True if not query else (query in code)
            self.table.setRowHidden(i, not visible)
