# -*- coding: utf-8 -*-
"""
Главное окно приложения (QMainWindow).

Содержит:
- таблицу ввода ступеней
- кнопки (Рассчитать, Очистить)
- вывод MMAD
- график (PlotWidget)
"""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import compute_mmad
from mmad_app.ui.plot_widget import PlotWidget


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("MMAD калькулятор (Andersen Cascade Impactor)")
        self.resize(1100, 700)

        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # Левая панель: ввод
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=1)

        title = QLabel("Ввод данных по ступеням импактора")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        left_layout.addWidget(title)

        hint = QLabel(
            "Введите D50 (µm) и массу (µg) для каждой ступени.\n"
            "Важно: D50 зависит от расхода (flow rate) через импактор."
        )
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        self.table = QTableWidget(9, 3)
        self.table.setHorizontalHeaderLabels(["Ступень", "D50, µm", "Масса, µg"])
        self.table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.table)

        self._fill_default_rows()

        btn_row = QHBoxLayout()
        left_layout.addLayout(btn_row)

        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self.on_calculate)
        btn_row.addWidget(self.btn_calc)

        self.btn_demo = QPushButton("Тестовые данные")
        self.btn_demo.clicked.connect(self.on_fill_demo)
        btn_row.addWidget(self.btn_demo)

        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.on_clear)
        btn_row.addWidget(self.btn_clear)

        self.result_label = QLabel("MMAD: —")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        left_layout.addWidget(self.result_label)

        # left_layout.addStretch(1)

        # Правая панель: график
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)

        self.plot_widget = PlotWidget()
        right_layout.addWidget(self.plot_widget)

    def _fill_default_rows(self) -> None:
        """
        Заполняет таблицу базовыми названиями ступеней.

        Значения D50 и массы оставляем пустыми — пользователь вводит сам.
        """
        default_names = [
            "Ступень 0",
            "Ступень 1",
            "Ступень 2",
            "Ступень 3",
            "Ступень 4",
            "Ступень 5",
            "Ступень 6",
            "Ступень 7",
            "Фильтр",
        ]

        for row, name in enumerate(default_names):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            d50_item = QTableWidgetItem("")
            d50_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, d50_item)

            mass_item = QTableWidgetItem("")
            mass_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                       Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, mass_item)

    def on_clear(self) -> None:
        """Очищает числовые поля, сбрасывает результат и график."""
        for row in range(self.table.rowCount()):
            d50_item = self.table.item(row, 1)
            mass_item = self.table.item(row, 2)
            if d50_item is not None:
                d50_item.setText("")
            if mass_item is not None:
                mass_item.setText("")

        self.result_label.setText("MMAD: —")
        self.plot_widget.plot_empty()

    def _read_records_from_table(self) -> List[StageRecord]:
        """
        Читает заполненные строки таблицы и возвращает список StageRecord.

        Правило:
        - если строка полностью пустая (и D50, и масса), то игнорируем её
        - иначе требуем, чтобы оба поля были числами
        """
        records: List[StageRecord] = []

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            d50_item = self.table.item(row, 1)
            mass_item = self.table.item(row, 2)

            stage_name = name_item.text().strip() if name_item else f"Row {row + 1}"
            d50_text = d50_item.text().strip() if d50_item else ""
            mass_text = mass_item.text().strip() if mass_item else ""

            if d50_text == "" and mass_text == "":
                continue

            try:
                d50_um = float(d50_text)
                mass_ug = float(mass_text)
            except ValueError as exc:
                raise ValueError(
                    f"Строка {row + 1}: D50 и Масса должны быть числами."
                ) from exc

            records.append(
                StageRecord(stage_name=stage_name, d50_um=d50_um, mass_ug=mass_ug)
            )

        if not records:
            raise ValueError(
                "Таблица пуста. Введите данные хотя бы для нескольких ступеней.")

        return records

    def on_calculate(self) -> None:
        """Считывает таблицу, считает MMAD, обновляет результат и график."""
        try:
            records = self._read_records_from_table()
            result = compute_mmad(records)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

        self.result_label.setText(
            "MMAD (D50): {mmad:.3f} µm    GSD: {gsd:.3f}\n"
            "D10: {d10:.3f} µm    D90: {d90:.3f} µm    Span: {span:.3f}\n"
            "FPF(<{cut:.1f} µm): {fpf:.2f}%    Total: {tot:.2f} µg".format(
                mmad=result.mmad_um,
                gsd=result.gsd,
                d10=result.d10_um,
                d90=result.d90_um,
                span=result.span,
                cut=result.fpf_cutoff_um,
                fpf=result.fpf_pct,
                tot=result.total_mass_ug,
            )
        )
        self.plot_widget.plot_cumulative(
            diam_um=result.diam_um,
            cum_pct=result.cum_undersize_pct,
            mmad_um=result.mmad_um,
            d15_87_um=result.d15_87_um,
            d84_13_um=result.d84_13_um,
            gsd=result.gsd,
        )

    def on_fill_demo(self) -> None:
        """
        Заполняет таблицу тестовыми значениями.
        """
        demo_rows = [
        # (name, d50_um, mass_ug)
            ("Ступень 0", 9.0, 0.0034),
            ("Ступень 1", 5.8, 0.0115),
            ("Ступень 2", 4.7, 0.0233),
            ("Ступень 3", 3.3, 0.0582),
            ("Ступень 4", 2.1, 0.0408),
            ("Ступень 5", 1.1, 0.0140),
            ("Ступень 6", 0.65, 0.0015),
            ("Ступень 7", 0.43, 0.0000),
            ("Фильтр", 0.0, 0.0000),
        ]

        # Запишем в таблицу построчно
        for row, (name, d50_um, mass_ug) in enumerate(demo_rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            d50_item = QTableWidgetItem(f"{d50_um}")
            d50_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, d50_item)

            mass_item = QTableWidgetItem(f"{mass_ug}")
            mass_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                       Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, mass_item)

        # После заполнения сразу считаем и рисуем
        self.on_calculate()
