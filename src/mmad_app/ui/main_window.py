# -*- coding: utf-8 -*-
# src/mmad_app/ui/main_window.py
"""
Главное окно приложения
"""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from typing import List
from typing import Optional
import csv
import sqlite3
import numpy as np
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
    QFileDialog,
    QLineEdit,
    QTabWidget,
)

from mmad_app.core.models import StageRecord, MmadResult
from mmad_app.core.mmad import compute_mmad, compute_cumulative_undersize
from mmad_app.db.repo import save_run, load_run, connect
from mmad_app.db.path import get_db_path
from mmad_app.ui.plot_tabs import PlotTabs
from mmad_app.ui.result_panel import ResultsPanel
from mmad_app.ui.db_panel import DbHistoryPanel


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        self._db_conn: sqlite3.Connection = connect(str(get_db_path()))

        # cохранение результатов после расчета
        self._last_records: Optional[list[StageRecord]] = None
        self._last_result: Optional[MmadResult] = None

        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("MMAD калькулятор")
        self.resize(1200, 800)

        # Центральный виджет
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Разделение интерфейса
        self.tabs = QTabWidget()
        # Растягивания названия вкладок на всю ширину
        self.tabs.tabBar().setExpanding(True)
        self.tabs.setUsesScrollButtons(False)
        main_layout.addWidget(self.tabs)

        page_calculate = QWidget()
        calc_layout = QHBoxLayout()

        # QSplitter по горизонтали
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)  # панели не схлопываются в "ноль"
        splitter.setHandleWidth(8)

        calc_layout.addWidget(splitter)
        # QSplitter принимает ТОЛЬКО QWidget
        # Левая панель: исходные данные
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Исходные данные")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        left_layout.addWidget(title)

        row_search = QHBoxLayout()
        row_search.setSpacing(8)
        row_search.addWidget(QLabel("Шифр пробы:"))

        self.sample_code_edit = QLineEdit()
        self.sample_code_edit.setPlaceholderText("Например: A-001")
        row_search.addWidget(self.sample_code_edit)

        left_layout.addLayout(row_search)

        hint = QLabel("Распределение массы частиц аэрозоля по ступеням импактора\n")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)

        self.table = QTableWidget(8, 4)
        self.table.setHorizontalHeaderLabels(
            ["Ступень", "dн, мкм", "dв, мкм", "Масса, мкг"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(
            self.table.sizePolicy().horizontalPolicy(),
            self.table.sizePolicy().verticalPolicy(),
        )
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

        # Вкладки графиков
        self.plot_tabs = PlotTabs()
        right_layout.addWidget(self.plot_tabs)

        # Кнопки под графиком
        self.btn_plot_row = QHBoxLayout()

        self.btn_save_current = QPushButton("Сохранить график")
        self.btn_save_all = QPushButton("Сохранить все графики…")
        self.btn_save_db = QPushButton("Сохранить в БД…")
        self.btn_export_csv = QPushButton("Экспорт CSV…")

        self.btn_export_csv.setEnabled(False)
        self.btn_save_db.setEnabled(False)
        self.btn_save_current.setEnabled(False)
        self.btn_save_all.setEnabled(False)

        self.btn_save_current.clicked.connect(self.on_export_save_current_plot)
        self.btn_save_db.clicked.connect(self.on_save_db)
        self.btn_save_all.clicked.connect(self.on_export_save_all_plots)
        self.btn_export_csv.clicked.connect(self.on_export_csv)

        self.btn_plot_row.addWidget(self.btn_save_current)
        self.btn_plot_row.addWidget(self.btn_save_db)
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
        left_panel.setMinimumWidth(600)
        right_panel.setMinimumWidth(600)

        page_calculate.setLayout(calc_layout)
        self.tabs.addTab(page_calculate, "Новый расчет")

        page_db = QWidget()
        self.db_panel = DbHistoryPanel(self._db_conn, parent=page_db)
        layout = QVBoxLayout(page_db)
        layout.addWidget(self.db_panel)
        self.db_panel.load_requested.connect(self.on_load_run_from_db)

        # Первичная загрузка
        self.db_panel.reload()
        self.tabs.addTab(page_db, "База данных")

    def _fill_default_rows(self) -> None:
        """Заполняет таблицу базовыми названиями ступеней."""
        rows = [
            ("Ступень 0", 9.0, 10.0),
            ("Ступень 1", 5.8, 9.0),
            ("Ступень 2", 4.7, 5.8),
            ("Ступень 3", 3.3, 4.7),
            ("Ступень 4", 2.1, 3.3),
            ("Ступень 5", 1.1, 2.1),
            ("Ступень 6", 0.65, 1.1),
            ("Ступень 7", 0.43, 0.65),
        ]

        for row, (name, d50_bottom, d50_top) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            d50_bottom_item = QTableWidgetItem(f"{d50_bottom}")
            d50_bottom_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 1, d50_bottom_item)
            d50_top_item = QTableWidgetItem(f"{d50_top}")
            d50_top_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 2, d50_top_item)

            mass_item = QTableWidgetItem("")
            mass_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
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
        self.btn_save_db.setEnabled(False)

        self.results_panel.clear()

        self.plot_tabs.tab_cum.plot_empty(
            title="Накопительная кривая массового распределения "
            "по аэродинамическому диаметру",
            xlabel="Аэродинамический диаметр, мкм",
            ylabel="Накопленная масса < dв, %",
            xlim=(0.0, 10.0),
            ylim=(0, 100),
        )
        self.plot_tabs.tab_probit.plot_empty(
            title="Лог-пробит распределение накопленной массы",
            xlabel="log10(d), мкм",
            ylabel="Probit = Φ⁻¹(p) + 5",
            xlim=(0.0, 10.0),
            ylim=(0, 8),
        )
        self.plot_tabs.tab_bar.plot_empty(
            title="Распределение размеров частиц по массе",
            xlabel="Средний геометрический диаметр ступени, мкм",
            ylabel="Масса, мкг",
            xlim=(0.0, 10.0),
            ylim=(0.0, 1.0),
        )
        self.plot_tabs.tab_mass_density_distribution.plot_empty(
            title="Дифференциальное массовое распределение по логарифму диаметра",
            xlabel="ln(d)",
            ylabel="(Δm/M) / Δln(d)",
            xlim=(0.0, 2.5),
            ylim=(0.0, 1.0),
        )

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
                "Таблица пуста. Введите данные хотя бы для нескольких ступеней."
            )

        return records

    def on_calculate(self) -> None:
        """Считывает таблицу, считает MMAD, обновляет результат и график."""
        try:
            records = self._read_records_from_table()
            result = compute_mmad(records)

            # Для последующего сохранения в БД/экспорта
            self._last_records = records
            self._last_result = result

            # Обновляем интерфейс
            self.results_panel.set_result(result)
            self.plot_tabs.plot_all(records=records, result=result)

            self.btn_export_csv.setEnabled(True)
            self.btn_save_current.setEnabled(True)
            self.btn_save_all.setEnabled(True)
            self.btn_save_db.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

    def on_fill_demo(self) -> None:
        """Заполняет таблицу тестовыми значениями."""
        demo_data = (0.0034, 0.0115, 0.0233, 0.0582, 0.0408, 0.0140, 0.0015, 0.0000)

        # Запись в таблицу построчно
        for row, mass_ug in enumerate(demo_data):
            mass_item = QTableWidgetItem(f"{mass_ug}")
            mass_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, 3, mass_item)

        # После заполнения сразу считаем и рисуем
        self.on_calculate()

    def on_save_plot_clicked(self) -> None:
        """Сохраняет график текущей вкладки."""
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
        """Сохраняет все графики в указанную папку."""
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
                widget.save_figure(path, dpi=dpi)  # type: ignore
                count_saved += 1

        if count_saved == 0:
            raise RuntimeError("Нет вкладок с графиками, поддерживающих сохранение.")

    def on_export_save_current_plot(self) -> None:
        """Сохраняет график текущей вкладки PlotTabs."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить график (текущая вкладка)",
            "plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not filename:
            return

        try:
            self.plot_tabs.save_current_plot(filename, dpi=300)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить график:\n{exc}")

    def on_export_save_all_plots(self) -> None:
        """Сохраняет все вкладки с графиками в выбранную папку."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для сохранения графиков",
        )
        if not directory:
            return
        try:
            self.save_all_plots(directory, dpi=300)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось сохранить графики:\n{exc}"
            )
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
            result = getattr(self, "_last_result", None)

            self._write_export_csv(Path(filename), records, result)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось экспортировать CSV:\n{exc}"
            )
            return

        QMessageBox.information(self, "Готово", "CSV успешно сохранён.")

    def _write_export_csv(
        self, path: Path, records: List["StageRecord"], result
    ) -> None:
        """
        Сохранение исходных данных и результатов расчета в CSV:
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")

            # 1) Исходные данные
            w.writerow(["# Исходные данные"])
            w.writerow(["Ступень", "dн, мкм", "dв, мкм", "Масса, мкг"])

            for r in records:
                stage_name = getattr(r, "name", "")
                d_low = getattr(r, "d_low", "")
                d_high = getattr(r, "d_high", "")
                mass = getattr(r, "mass", "")

                w.writerow([stage_name, d_low, d_high, mass])

            # 2) Результаты расчета
            w.writerow(["# Результаты расчета"])
            if result is None:
                w.writerow(["message", "Результаты не рассчитаны"])
            else:
                w.writerow(["MMAD (d50), мкм", f"{result.mmad:.2g}"])
                w.writerow(["GSD", f"{result.gsd:.2g}"])
                w.writerow(["d10, мкм", f"{result.d10:.2g}"])
                w.writerow(["d16, мкм", f"{result.d16:.2g}"])
                w.writerow(["d84, мкм", f"{result.d84:.2g}"])
                w.writerow(["d90, мкм", f"{result.d90:.2g}"])
                w.writerow(["Span", f"{result.span:.2g}"])
                w.writerow(["FPF (< 5 мкм), %", f"{result.fpf_pct:.2g}"])
                w.writerow(["Лог. средний диаметр, мкм, %", f"{result.log_mean:.2g}"])
                w.writerow(["Среднемассовый диаметр, мкм", f"{result.mass_mean:.2g}"])
                w.writerow(["Модальный диаметр, мкм", f"{result.modal:.2g}"])

            w.writerow([])

    def on_save_db(self) -> None:
        """
        Сохраняет последний рассчитанный результат и исходные данные в SQLite.

        """
        try:
            sample_code = self.sample_code_edit.text().strip()
            if not sample_code:
                raise ValueError("Введите шифр пробы перед сохранением в БД.")

            records = getattr(self, "_last_records", None)
            result = getattr(self, "_last_result", None)

            if records is None or result is None:
                raise ValueError(
                    "Сначала выполните расчёт, затем сохраните результат в БД."
                )

            run_id = save_run(
                self._db_conn,
                sample_code=sample_code,
                records=records,
                result=result,
                notes=None,
            )

            QMessageBox.information(
                self, "Готово", f"Результат сохранён в БД (ID={run_id})."
            )

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))

    def on_load_run_from_db(self, run_id: int) -> None:
        """Загружает сохранённый расчёт из SQLite и отображает его в интерфейсе."""
        try:
            run_row, stage_rows = load_run(self._db_conn, run_id)

            # Шифр пробы
            self.sample_code_edit.setText(str(run_row["sample_code"]))

            # Заполннение таблицы исходных данных
            self.table.setRowCount(len(stage_rows))
            records: list[StageRecord] = []

            for row_idx, s in enumerate(stage_rows):
                stage_name = str(s["stage_name"])
                d_low = float(s["d_low"])
                d_high = float(s["d_high"])
                mass = float(s["mass"])

                # Таблица: названия/значения
                self.table.setItem(row_idx, 0, QTableWidgetItem(stage_name))

                low_item = QTableWidgetItem(f"{d_low:g}")
                low_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, 1, low_item)

                high_item = QTableWidgetItem(f"{d_high:g}")
                high_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, 2, high_item)

                mass_item = QTableWidgetItem(f"{mass:g}")
                mass_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, 3, mass_item)

                records.append(
                    StageRecord(
                        name=stage_name,
                        d_low=d_low,
                        d_high=d_high,
                        mass=mass,
                    )
                )

            # 3) Формирование MmadResult из runs (без пересчёта)
            result = MmadResult(
                mmad=float(run_row["mmad"]),
                gsd=float(run_row["gsd"]),
                d10=float(run_row["d10"]),
                d16=float(run_row["d16"]),
                d84=float(run_row["d84"]),
                d90=float(run_row["d90"]),
                d15_87=float(run_row["d15_87"]),
                d84_13=float(run_row["d84_13"]),
                span=float(run_row["span"]),
                fpf_cutoff_um=float(run_row["fpf_cutoff_um"]),
                fpf_pct=float(run_row["fpf_pct"]),
                total_mass=float(run_row["total_mass_ug"]),
                log_mean=float(run_row["log_mean"]),
                mass_mean=float(run_row["mass_mean"]),
                modal=float(run_row["modal"]),
                diam_um=np.array([], dtype=float),
                cum_undersize_pct=np.array([], dtype=float),
            )

            # Пересчет кумулятивной кривой для графиков
            diam_um, cum_pct = compute_cumulative_undersize(records)

            # "дозаполняем" result для графиков
            result = replace(result, diam_um=diam_um, cum_undersize_pct=cum_pct)

            # Сохраняем "последние" значения (для экспорта/сохранения)
            self._last_records = records
            self._last_result = result

            # Обновление UI: карточка + графики
            self.results_panel.set_result(result)
            self.plot_tabs.plot_all(records=records, result=result)

            # Переключимся на вкладку "Новый расчёт"
            if hasattr(self, "tabs"):
                self.tabs.setCurrentIndex(0)

            self.btn_export_csv.setEnabled(True)
            self.btn_save_current.setEnabled(True)
            self.btn_save_all.setEnabled(True)
            self.btn_save_db.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
