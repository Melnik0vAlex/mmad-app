# -*- coding: utf-8 -*-
"""
Точка входа GUI-приложения.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from mmad_app.ui.main_window import MainWindow


def main() -> None:
    """Запуск Qt-приложения."""
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
