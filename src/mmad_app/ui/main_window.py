# -*- coding: utf-8 -*-
# src/mmad_app/ui/main_window.py
"""
Главное окно приложения
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Union, Any
import csv
import math
import sqlite3
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

from mmad_app.core.models import StageRecord, MmadResult, MmadResultLS
from mmad_app.core.mmad import (
    compute_mmad,
    compute_mmad_least_squares,
)
from mmad_app.db.repo import save_run, load_run, connect
from mmad_app.db.path import get_db_path
from mmad_app.ui.plot_tabs import PlotTabs
from mmad_app.ui.result_panel import ResultsPanel
from mmad_app.ui.db_panel import DbHistoryPanel


def _fmt_optional_num(value: object, fmt: str = "{:.4g}") -> str:
    """
    Форматирование чисел для CSV.

    - None -> пустая строка
    - NaN/inf -> пустая строка
    - прочее -> fmt
    """
    if value is None:
        return ""
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(x):
        return ""
    return fmt.format(x)


class MainWindow(QMainWindow):
    """Главное окно приложения."""

    def __init__(self) -> None:
        super().__init__()
        self._db_conn: sqlite3.Connection = connect(str(get_db_path()))

        # cохранение результатов после расчета
        self._last_records: Optional[list[StageRecord]] = None
        self._last_result_lp: Optional[MmadResult] = None
        self._last_result_ls: Optional[MmadResultLS] = None

        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("MMAD калькулятор")
        self.resize(1300, 1080)

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
        calc_layout.setContentsMargins(8, 8, 8, 8)

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

        self.table = QTableWidget(11, 4)
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
        right_layout.setContentsMargins(8, 8, 8, 8)

        result_title = QLabel("Результаты расчетов")
        result_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        right_layout.addWidget(result_title)

        layout_result = QHBoxLayout()
        right_layout.addLayout(layout_result)

        self.panel_method_lp = ResultsPanel(
            title="Интерполяция интегральной функции распределения",
            rows=(
                ("mmad", "MMAD, мкм"),
                ("log_mean", "LMD, мкм"),
                ("mass_mean", "MMD, мкм"),
                ("modal", "mod, мкм"),
                ("gsd", "GSD"),
                ("d10", "d10, мкм"),
                ("d16", "d16, мкм"),
                ("d84", "d84, мкм"),
                ("d90", "d90, мкм"),
                ("span", "Span"),
                ("fpf_pct", "FPF (< 5 мкм), %"),
            ),
        )

        self.panel_method_ls = ResultsPanel(
            title="Линеаризация лог-нормального распределения",
            rows=(
                ("mmad", "d50, мкм"),
                ("kor_k", "Корень K"),
                ("sigma", "sigma(ln)"),
                ("r", "Коэфф. корреляции"),
                ("slope", "a"),
                ("intercept", "b"),
                ("se_slope", "SE(a)"),
                ("se_intercept", "SE(b)"),
                ("r2", "R²"),
                ("syx", "SE_yx"),
                ("f_stat", "F"),
                ("df", "df"),
                ("ss_reg", "SS_reg"),
                ("ss_res", "SS_res"),
            ),
        )

        layout_result.addWidget(self.panel_method_lp)
        layout_result.addWidget(self.panel_method_ls)

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
            ("Фильтр", "-", 0.43),
            ("Ступень 7", 0.43, 0.65),
            ("Ступень 6", 0.65, 1.1),
            ("Ступень 5", 1.1, 2.1),
            ("Ступень 4", 2.1, 3.3),
            ("Ступень 3", 3.3, 4.7),
            ("Ступень 2", 4.7, 5.8),
            ("Ступень 1", 5.8, 9.0),
            ("Ступень 0", 9.0, 10.0),
            ("Пресепаратор", 10.0, "-"),
            ("Входной канал", "-", "-"),
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

        # Запрет редактирования выделенных колонок
        readonly_cols = (0, 1, 2)
        for row in range(self.table.rowCount()):
            for col in readonly_cols:
                item = self.table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("")
                    self.table.setItem(row, col, item)

                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

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

        self.panel_method_lp.clear()
        self.panel_method_ls.clear()

        self.plot_tabs.tab_cum.plot_empty(
            title="Интегральная кривая массового распределения "
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

    def _parse_bound(self, value: Union[float, str]) -> Optional[float]:
        """
        Преобразует границу диаметра из таблицы: "-" -> None; float -> float
        """
        if isinstance(value, str):
            if value.strip() == "-":
                return None
            return float(value)

    def _parse_mass_text(self, text: str, row: int) -> float:
        """
        Парсинг массы на ступени импактора из текста таблицы.
        """
        s = text.strip()
        if s == "":
            return 0.0

        try:
            mass = float(s)
        except ValueError as exc:
            raise ValueError(f"Строка {row + 1}: Масса должна быть числом.") from exc

        if mass < 0:
            raise ValueError(f"Строка {row + 1}: Масса не может быть отрицательной.")

        return mass

    def _read_records_from_table(self) -> List[StageRecord]:
        """
        Читает заполненные строки таблицы и возвращает список StageRecord.
        """
        records: List[StageRecord] = []

        for row in range(self.table.rowCount()):
            name_item: QTableWidgetItem | None = self.table.item(row, 0)
            name = name_item.text() if name_item else ""

            d_low_item: QTableWidgetItem | None = self.table.item(row, 1)
            d_low_text = d_low_item.text() if d_low_item else ""
            d_low = self._parse_bound(d_low_text)

            d_high_item: QTableWidgetItem | None = self.table.item(row, 2)
            d_high_text = d_high_item.text() if d_high_item else ""
            d_high = self._parse_bound(d_high_text)

            mass_item: QTableWidgetItem | None = self.table.item(row, 3)
            mass_text = mass_item.text() if mass_item else ""
            mass = self._parse_mass_text(mass_text, row)

            records.append(
                StageRecord(name=name, d_low=d_low, d_high=d_high, mass=mass)
            )

        return records

    def on_calculate(self) -> None:
        """Считывает таблицу, считает MMAD, обновляет результат и график."""
        try:
            records = self._read_records_from_table()

            # результаты по модели лог-пробит
            result_lp = compute_mmad(records)
            # результаты по модели МНК
            result_ls = compute_mmad_least_squares(records)

            # Для последующего сохранения в БД/экспорта
            self._last_records = records
            self._last_result_lp = result_lp
            self._last_result_ls = result_ls

            # Обновляем интерфейс
            self.panel_method_lp.set_result(result_lp)
            self.panel_method_ls.set_result(result_ls)

            self.plot_tabs.plot_all(
                records=records, result_lp=result_lp, result_ls=result_ls
            )

            self.btn_export_csv.setEnabled(True)
            self.btn_save_current.setEnabled(True)
            self.btn_save_all.setEnabled(True)
            self.btn_save_db.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))
            return

    def on_fill_demo(self) -> None:
        """Заполняет таблицу тестовыми значениями."""
        demo_data = (
            0.0052,
            0.0009,
            0.003,
            0.0319,
            0.0714,
            0.0714,
            0.0427,
            0.0149,
            0.0028,
            0.0036,
            0.0,
        )

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
            result_lp = getattr(self, "_last_result_lp", None)
            result_ls = getattr(self, "_last_result_ls", None)

            self._write_export_csv(Path(filename), records, result_lp, result_ls)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось экспортировать CSV:\n{exc}"
            )
            return

        QMessageBox.information(self, "Готово", "CSV успешно сохранён.")

    def _write_export_csv(
        self, path: Path, records: List["StageRecord"], result_lp, result_ls
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
            if result_lp is None:
                w.writerow(["message", "Результаты (LP) не рассчитаны"])
            else:
                w.writerow(["MMAD (d50), мкм", _fmt_optional_num(result_lp.mmad)])
                w.writerow(["GSD", _fmt_optional_num(result_lp.gsd)])
                w.writerow(["d10, мкм", _fmt_optional_num(result_lp.d10)])
                w.writerow(["d16, мкм", _fmt_optional_num(result_lp.d16)])
                w.writerow(["d84, мкм", _fmt_optional_num(result_lp.d84)])
                w.writerow(["d90, мкм", _fmt_optional_num(result_lp.d90)])
                w.writerow(["d15.87, мкм", _fmt_optional_num(result_lp.d15_87)])
                w.writerow(["d84.13, мкм", _fmt_optional_num(result_lp.d84_13)])
                w.writerow(["Span", _fmt_optional_num(result_lp.span)])
                w.writerow(
                    [
                        f"FPF (< {result_lp.fpf_cutoff_um:g} мкм), %",
                        _fmt_optional_num(result_lp.fpf_pct),
                    ]
                )
                w.writerow(["Σ масса, мкг", _fmt_optional_num(result_lp.total_mass)])
                w.writerow(
                    ["LMD (log-mean), мкм", _fmt_optional_num(result_lp.log_mean)]
                )
                w.writerow(
                    ["MMD (mass-mean), мкм", _fmt_optional_num(result_lp.mass_mean)]
                )
                w.writerow(["Mod, мкм", _fmt_optional_num(result_lp.modal)])

            w.writerow([])

            # Результаты метода наименьших квадратов (LS)
            w.writerow(["# Результаты (метод наименьших квадратов"])
            if result_ls is None:
                w.writerow(["message", "Результаты (LS) не рассчитаны"])
            else:
                # Параметры логнормального распределения (по линейной модели)
                w.writerow(
                    ["D50 = exp(intercept), мкм", _fmt_optional_num(result_ls.mmad)]
                )
                w.writerow(["σ(lnD)", _fmt_optional_num(result_ls.sigma)])
                w.writerow(["√K (kor_k)", _fmt_optional_num(result_ls.kor_k)])
                w.writerow(["r (corr)", _fmt_optional_num(result_ls.r)])

                w.writerow([])

                # Статистика регрессии
                w.writerow(["# Статистика регрессии (LINEST)"])
                w.writerow(["slope (m)", _fmt_optional_num(result_ls.slope)])
                w.writerow(["intercept (c)", _fmt_optional_num(result_ls.intercept)])
                w.writerow(["SE(slope)", _fmt_optional_num(result_ls.se_slope)])
                w.writerow(["SE(intercept)", _fmt_optional_num(result_ls.se_intercept)])
                w.writerow(["R^2", _fmt_optional_num(result_ls.r2)])
                w.writerow(["SE_yx (syx)", _fmt_optional_num(result_ls.syx)])
                w.writerow(["F", _fmt_optional_num(result_ls.f_stat)])
                w.writerow(["df", str(int(result_ls.df))])
                w.writerow(["SS_reg", _fmt_optional_num(result_ls.ss_reg)])
                w.writerow(["SS_res", _fmt_optional_num(result_ls.ss_res)])

            w.writerow([])

    def on_save_db(self) -> None:
        """Сохраняет последний рассчитанный результат и исходные данные в SQLite."""
        try:
            sample_code = self.sample_code_edit.text().strip()
            if not sample_code:
                raise ValueError("Введите шифр пробы перед сохранением в БД.")

            records = getattr(self, "_last_records", None)
            result_lp = getattr(self, "_last_result_lp", None)
            result_ls = getattr(self, "_last_result_ls", None)

            if records is None or result_lp is None or result_ls is None:
                raise ValueError(
                    "Сначала выполните расчёт, затем сохраните результат в БД."
                )

            run_id = save_run(
                self._db_conn,
                sample_code=sample_code,
                records=records,
                result_lp=result_lp,
                result_ls=result_ls,
                notes=None,
            )

            # обновляем вкладку БД
            if hasattr(self, "db_panel") and self.db_panel is not None:
                self.db_panel.reload()

            QMessageBox.information(
                self, "Готово", f"Результат сохранён в БД (ID={run_id})."
            )

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка", str(exc))

    def _opt_float(self, value: Any) -> Optional[float]:
        """Преобразует значение в float, но сохраняет None как None."""
        if value is None:
            return None
        return float(value)

    def _req_float(self, value: Any, *, field: str) -> float:
        """Преобразует значение в float."""
        if value is None:
            raise ValueError(
                f"Поле '{field}' = NULL в БД, невозможно загрузить расчёт."
            )
        return float(value)

    def on_load_run_from_db(self, run_id: int) -> None:
        """Загружает сохранённый расчёт из SQLite и отображает его в интерфейсе."""
        try:
            run_row, stage_rows = load_run(self._db_conn, run_id)

            # Шифр пробы
            self.sample_code_edit.setText(str(run_row["sample_code"]))

            # Заполннение таблицы исходных данных
            self.table.setRowCount(len(stage_rows))

            for row_idx, s in enumerate(stage_rows):
                stage_name = str(s["stage_name"])
                d_low = "-" if s["d_low"] is None else float(s["d_low"])
                d_high = "-" if s["d_high"] is None else float(s["d_high"])
                mass = float(s["mass"])

                # Таблица: названия/значения
                self.table.setItem(row_idx, 0, QTableWidgetItem(stage_name))

                # d_low
                low_text = "" if d_low is None else f"{d_low}"
                low_item = QTableWidgetItem(low_text)
                low_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, 1, low_item)

                # d_high
                high_text = "" if d_high is None else f"{d_high}"
                high_item = QTableWidgetItem(high_text)
                high_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, 2, high_item)

                mass_item = QTableWidgetItem(f"{mass:g}")
                mass_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self.table.setItem(row_idx, 3, mass_item)

            self.on_calculate()

            # Переключимся на вкладку "Новый расчёт"
            if hasattr(self, "tabs"):
                self.tabs.setCurrentIndex(0)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка загрузки", str(exc))
