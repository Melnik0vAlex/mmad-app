# -*- coding: utf-8 -*-
# src/mmad_app/ui/db_panel.py
"""
Панель с результатами расчетов
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
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
    """Представление записи из таблицы runs для отображения в UI."""

    run_id: int
    created_at: str
    sample_code: str

    mmad_lp: float
    lmd: float
    mmd: float
    mod: float
    gsd: float
    d10: float
    d16: float
    d84: float
    d90: float
    span: float
    fpf_pct: float

    mmad_ls: float
    kor_k: float
    sigma: float
    r: float
    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    r2: float
    syx: float
    f_stat: float
    df: int
    ss_reg: float
    ss_res: float


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

        # Верхняя панель: поиск + кнопки
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

        # Таблицичное отображение результатов расчета
        self.table = QTableWidget(0, 18)
        self.table.setHorizontalHeaderLabels(
            [
                "ID",
                "Дата",
                "Шифр пробы",
                "MMAD (log-probit), мкм",
                "LMD, мкм",
                "MMD, мкм",
                "mod, мкм",
                "GSD",
                "d10, мкм",
                "d16, мкм",
                "d84, мкм",
                "d90, мкм",
                "Span",
                "FPF, %",
                "d50 (мнк), мкм",
                "√k",
                "σ(lnD)",
                "r"
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        # Растягивание некоторх колонок
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )

        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.cellDoubleClicked.connect(self._on_double_click_row)

        root.addWidget(self.table, stretch=1)

        # Отображение подсказки снизу
        self.hint = QLabel("Двойной клик по строке — загрузить запись.")
        self.hint.setStyleSheet("color: #777;")
        root.addWidget(self.hint)

    def reload(self) -> None:
        """Загружает последние записи из БД и обновляет таблицу."""
        rows = list_runs(self._conn, limit=200)

        self._rows = [
            RunRow(
                run_id=int(r["id"]),
                created_at=str(r["created_at"]),
                sample_code=str(r["sample_code"]),

                mmad_lp=float(r["mmad"]),
                lmd=float(r["log_mean"]),
                mmd=float(r["mass_mean"]),
                mod=float(r["modal"]),
                gsd=float(r["gsd"]),
                d10=float(r["d10"]),
                d16=float(r["d16"]),
                d84=float(r["d84"]),
                d90=float(r["d90"]),
                span=float(r["span"]),
                fpf_pct=float(r["fpf_pct"]),

                mmad_ls=float(r["mmad_ls"]),
                kor_k=float(r["kor_k"]),
                sigma=float(r["sigma"]),
                r=float(r["r"]),
                slope=float(r["slope"]),
                intercept=float(r["intercept"]),
                se_slope=float(r["se_slope"]),
                se_intercept=float(r["se_intercept"]),
                r2=float(r["r2"]),
                syx=float(r["syx"]),
                f_stat=float(r["f_stat"]),
                df=int(r["df"]),
                ss_reg=float(r["ss_reg"]),
                ss_res=float(r["ss_res"]),
            )
            for r in rows
        ]

        self._render_table(self._rows)

    def _render_table(self, rows: List[RunRow]) -> None:
        """Заполняет QTableWidget данными."""
        self.table.setRowCount(0)

        for row_idx, r in enumerate(rows):
            self.table.insertRow(row_idx)
            self._set_item(row_idx, 0, str(r.run_id), align_right=True)
            self._set_item(row_idx, 1, self._format_created_at(r.created_at))
            self._set_item(row_idx, 2, r.sample_code)
            self._set_item(row_idx, 3, f"{r.mmad_lp:.2f}", align_right=True)
            self._set_item(row_idx, 4, f"{r.lmd:.2f}", align_right=True)
            self._set_item(row_idx, 5, f"{r.mmd:.2f}", align_right=True)
            self._set_item(row_idx, 6, f"{r.mod:.2f}", align_right=True)
            self._set_item(row_idx, 7, f"{r.gsd:.2f}", align_right=True)
            self._set_item(row_idx, 8, f"{r.d10:.2f}", align_right=True)
            self._set_item(row_idx, 9, f"{r.d16:.2f}", align_right=True)
            self._set_item(row_idx, 10, f"{r.d84:.2f}", align_right=True)
            self._set_item(row_idx, 11, f"{r.d90:.2f}", align_right=True)
            self._set_item(row_idx, 12, f"{r.span:.2f}", align_right=True)
            self._set_item(row_idx, 13, f"{r.fpf_pct:.2f}", align_right=True)
            self._set_item(row_idx, 14, f"{r.mmad_ls:.2f}", align_right=True)
            self._set_item(row_idx, 15, f"{r.kor_k:.2f}", align_right=True)
            self._set_item(row_idx, 16, f"{r.sigma:.2f}", align_right=True)
            self._set_item(row_idx, 17, f"{r.r:.2f}", align_right=True)

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

    def _on_double_click_row(
        self, row: int, col: int  # pylint: disable=unused-argument
    ) -> None:
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

        # Обновление таблицы и очистка выбора
        self.reload()

    def _apply_filter(self, text: str) -> None:
        """Фильтрует таблицу по sample_code (простая подстрока, без SQL)."""
        query = (text or "").strip().lower()

        for i in range(self.table.rowCount()):
            item = self.table.item(i, 2)  # column "Шифр пробы"
            code = item.text().lower() if item else ""
            visible = True if not query else (query in code)
            self.table.setRowHidden(i, not visible)

    def _format_created_at(self, iso_str: str) -> str:
        """Преобразует дату/время из ISO 8601."""
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is not None:
                dt = dt.astimezone()
            return dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            return iso_str
