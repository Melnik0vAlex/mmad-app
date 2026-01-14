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
from mmad_app.ui.plot_tabs import PlotTabs


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
            "Введите D50 (мкм) и массу (мкг) для каждой ступени импактора.\n"
        )
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        self.table = QTableWidget(8, 4)
        self.table.setHorizontalHeaderLabels(
            ["Ступень", "dн, мкм", "dв, мкм", "Масса, мкг"])
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

        self.result_label = QLabel("Результаты расчета:")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        left_layout.addWidget(self.result_label)

        # left_layout.addStretch(1)

        # Правая панель: график
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=1)

        self.plot_tabs = PlotTabs()
        right_layout.addWidget(self.plot_tabs)

    def _fill_default_rows(self) -> None:
        """
        Заполняет таблицу базовыми названиями ступеней.
        """
        rows = [
            ("Ступень 0", 9.0, 10.0),
            ("Ступень 1", 5.8, 9.0),
            ("Ступень 2", 4.7, 5.8),
            ("Ступень 3", 3.3, 4.7),
            ("Ступень 4", 2.1, 3.3),
            ("Ступень 5", 1.1, 2.1),
            ("Ступень 6", 0.65, 1.1),
            ("Ступень 7", 0.43, 0.65)
        ]

        for row, (name, d50_bottom, d50_top) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            d50_bottom_item = QTableWidgetItem(f"{d50_bottom}")
            d50_bottom_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 1, d50_bottom_item)
            d50_top_item = QTableWidgetItem(f"{d50_top}")
            d50_top_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                      Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 2, d50_top_item)

            mass_item = QTableWidgetItem("")
            mass_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                       Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 3, mass_item)

    def on_clear(self) -> None:
        """Очищает числовые поля, сбрасывает результат и график."""
        for row in range(self.table.rowCount()):
            mass_item = self.table.item(row, 3)
            if mass_item is not None:
                mass_item.setText("")

        self.result_label.setText("Результаты расчета:")
        self.plot_tabs.tab_cum.plot_empty()
        self.plot_tabs.tab_probit.plot_empty()
        self.plot_tabs.tab_bar.plot_empty()

    def _read_records_from_table(self) -> List[StageRecord]:
        """
        Читает заполненные строки таблицы и возвращает список StageRecord.
        """
        records: List[StageRecord] = []

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            d_low_item = self.table.item(row, 1)
            d_high_item = self.table.item(row, 2)
            mass_item = self.table.item(row, 3)

            stage_name = name_item.text().strip() if name_item else f"Row {row + 1}"
            d_low_text = d_low_item.text().strip() if d_low_item else ""
            d_high_text = d_high_item.text().strip() if d_high_item else ""
            mass_text = mass_item.text().strip() if mass_item else ""

            if d_low_text == "" and d_high_text == "" and mass_text == "":
                continue

            try:
                d_low = float(d_low_text)
                d_high = float(d_high_text)
                mass = float(mass_text)
            except ValueError as exc:
                raise ValueError(
                    f"Строка {row + 1}: Диаметры ступеней и Масса должны быть числами."
                ) from exc

            records.append(
                StageRecord(name=stage_name, d_low=d_low, d_high=d_high, mass=mass)
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
            f"MMAD: {result.mmad_um:.2f} мкм\n"
            f"GSD: {result.gsd:.2f}\n"
            f"d10: {result.d10_um:.2f} мкм\n"
            f"d90: {result.d90_um:.2f} мкм\n"
            f"Span: {result.span:.2f}\n"
            f"FPF(<{result.fpf_cutoff_um:.1f} мкм): {result.fpf_pct:.2f} %"
        )

        self.plot_tabs.plot_all(records=records, result=result)

    def on_fill_demo(self) -> None:
        """
        Заполняет таблицу тестовыми значениями.
        """
        demo_data = (0.0034, 0.0115, 0.0233, 0.0582, 0.0408, 0.0140, 0.0015, 0.0000)

        # Запишем в таблицу построчно
        for row, mass_ug in enumerate(demo_data):
            mass_item = QTableWidgetItem(f"{mass_ug}")
            mass_item.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                       Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 3, mass_item)

        # После заполнения сразу считаем и рисуем
        self.on_calculate()
