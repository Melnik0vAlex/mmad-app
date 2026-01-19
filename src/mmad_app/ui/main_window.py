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


import csv
from pathlib import Path

from typing import List, Optional
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
    QSplitter,
    QFileDialog
)

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import compute_mmad
from mmad_app.ui.plot_tabs import PlotTabs
from mmad_app.ui.result_panel import ResultsPanel


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("MMAD калькулятор")
        self.resize(1200, 800)

        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # QSplitter по горизонтали
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)  # панели не схлопываются в "ноль"
        splitter.setHandleWidth(8)

        main_layout.addWidget(splitter)

        # QSplitter принимает ТОЛЬКО QWidget
        # Левая панель: исходные данные
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Исходные данные")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        left_layout.addWidget(title)

        hint = QLabel(
            "Распределение массы частиц аэрозоля по ступеням импактора\n"
        )
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        self.table = QTableWidget(8, 4)
        self.table.setHorizontalHeaderLabels(
            ["Ступень", "dн, мкм", "dв, мкм", "Масса, мкг"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(self.table.sizePolicy().horizontalPolicy(),
                                 self.table.sizePolicy().verticalPolicy())
        left_layout.addWidget(self.table, stretch=1)

        self._fill_default_rows()

        btn_row = QHBoxLayout()
        left_layout.addLayout(btn_row)

        self.btn_calc = QPushButton("Рассчитать")
        self.btn_calc.clicked.connect(self.on_calculate)
        btn_row.addWidget(self.btn_calc, stretch=1)

        self.btn_demo = QPushButton("Тестовые данные")
        self.btn_demo.clicked.connect(self.on_fill_demo)
        btn_row.addWidget(self.btn_demo, stretch=1)

        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.on_clear)
        btn_row.addWidget(self.btn_clear, stretch=1)

        left_layout.addStretch(1)

        # Правая панель
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        result_title = QLabel("Результаты расчетов")
        result_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        right_layout.addWidget(result_title)

        self.results_panel = ResultsPanel()
        right_layout.addWidget(self.results_panel)

        self.plot_tabs = PlotTabs()
        right_layout.addWidget(self.plot_tabs)

        # Кнопки под графиком
        self.btn_plot_row = QHBoxLayout()

        self.btn_save_current = QPushButton("Сохранить")
        self.btn_save_all = QPushButton("Сохранить все графики…")
        self.btn_export_csv = QPushButton("Экспорт CSV…")

        self.btn_export_csv.setEnabled(False)
        self.btn_save_current.setEnabled(False)
        self.btn_save_all.setEnabled(False)

        self.btn_save_current.clicked.connect(self.on_export_save_current_plot)
        self.btn_save_all.clicked.connect(self.on_export_save_all_plots)
        self.btn_export_csv.clicked.connect(self.on_export_csv)

        self.btn_plot_row.addWidget(self.btn_save_current)
        self.btn_plot_row.addWidget(self.btn_save_all)
        self.btn_plot_row.addWidget(self.btn_export_csv)
        self.btn_plot_row.addStretch(1)

        right_layout.addLayout(self.btn_plot_row)

        # Добавлие панелей в splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # Настройка стартовых пропорции splitter (в пикселях)
        splitter.setSizes([480, 620])

        # Настройка минимальной ширины
        left_panel.setMinimumWidth(400)
        right_panel.setMinimumWidth(400)


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

        self.btn_export_csv.setEnabled(False)
        self.btn_save_current.setEnabled(False)
        self.btn_save_all.setEnabled(False)

        self.results_panel.clear()

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

        self.results_panel.set_result(result)
        self.plot_tabs.plot_all(records=records, result=result)

        self.btn_export_csv.setEnabled(True)
        self.btn_save_current.setEnabled(True)
        self.btn_save_all.setEnabled(True)


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


    def on_save_plot_clicked(self) -> None:
        """
        Сохраняет график текущей вкладки.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график",
            "plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not filename:
            return

        try:
            self.plot_tabs.save_current_plot(filename, dpi=300)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))


    def save_all_plots(self, directory: str | Path, *, dpi: int = 300) -> None:
        """
        Сохраняет все графики в указанную папку.

        Файлы сохраняются в PNG. При желании можно сменить расширение на PDF/SVG.
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        count_saved = 0

        for i in range(self.plot_tabs.tabs.count()):
            widget = self.plot_tabs.tabs.widget(i)
            title = self.plot_tabs.tabs.tabText(i).strip()

            # Безопасное имя файла
            safe = (
                title.replace(" ", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace("–", "-")
            )

            filename = f"{i + 1:02d}_{safe}.png"
            path = out_dir / filename

            if hasattr(widget, "save_figure"):
                widget.save_figure(path, dpi=dpi)
                count_saved += 1

        if count_saved == 0:
            raise RuntimeError("Нет вкладок с графиками, поддерживающих сохранение.")


    def on_export_save_current_plot(self) -> None:
        """
        Сохраняет график текущей вкладки PlotTabs.
        """
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график (текущая вкладка)",
            "plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not filename:
            return

        try:
            # Вы добавляли ранее метод save_current_plot(...)
            self.plot_tabs.save_current_plot(filename, dpi=300)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить график:\n{exc}")


    def on_export_save_all_plots(self) -> None:
        """
        Сохраняет все вкладки с графиками в выбранную папку.
        """
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения графиков",
        )
        if not directory:
            return
        try:
            self.save_all_plots(directory, dpi=300)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить графики:\n{exc}")
            return

        QMessageBox.information(self, "Готово", "Графики сохранены.")


    def on_export_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт CSV",
            "mmad_export.csv",
            "CSV (*.csv)",
        )
        if not filename:
            return

        try:
            records = self._read_records_from_table()

            # Важно: у вас должен быть сохранён последний результат.
            # Например, после расчёта:
            # self._last_result = result
            result = getattr(self, "_last_result", None)

            self._write_export_csv(Path(filename), records, result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать CSV:\n{exc}")
            return

        QMessageBox.information(self, "Готово", "CSV успешно сохранён.")


    def _write_export_csv(
        self,
        path: Path,
        records: List["StageRecord"],
        result
    ) -> None:
        """
        Записывает CSV:
        1) Секция метрик (если result не None)
        2) Секция таблицы ступеней
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")

            # 1) Метрики
            w.writerow(["# META"])
            if result is None:
                w.writerow(["message", "Результаты не рассчитаны"])
            else:
                # Пишем только ключевые поля. Добавьте/уберите по вкусу.
                w.writerow(["mmad_um", f"{result.mmad_um:.6g}"])
                w.writerow(["gsd", f"{result.gsd:.6g}"])
                w.writerow(["d10_um", f"{result.d10_um:.6g}"])
                if hasattr(result, "d16_um"):
                    w.writerow(["d16_um", f"{result.d16_um:.6g}"])
                if hasattr(result, "d84_um"):
                    w.writerow(["d84_um", f"{result.d84_um:.6g}"])
                w.writerow(["d90_um", f"{result.d90_um:.6g}"])
                w.writerow(["span", f"{result.span:.6g}"])
                w.writerow(["fpf_cutoff_um", f"{result.fpf_cutoff_um:.6g}"])
                w.writerow(["fpf_pct", f"{result.fpf_pct:.6g}"])
                w.writerow(["total_mass_ug", f"{result.total_mass_ug:.6g}"])

                # Доп. поля (если вы добавили)
                for name in ("log_mean_um", "mass_mean_um", "modal_um"):
                    if hasattr(result, name):
                        val = getattr(result, name)
                        try:
                            w.writerow([name, f"{float(val):.6g}"])
                        except Exception:
                            w.writerow([name, ""])

            w.writerow([])

            # ----------------
            # 2) Таблица ступеней
            # ----------------
            w.writerow(["# TABLE"])
            w.writerow(["stage_name", "d_low_um", "d_high_um", "mass_ug"])

            for r in records:
                # Подстройте под ваши реальные поля StageRecord
                stage_name = getattr(r, "stage_name", "")
                d_low = getattr(r, "d_low", "")
                d_high = getattr(r, "d_high", "")
                mass = getattr(r, "mass", "")

                w.writerow([stage_name, d_low, d_high, mass])