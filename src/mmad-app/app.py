# -*- coding: utf-8 -*-
"""
Точка входа GUI-приложения.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QMainWindow


def main() -> None:
    """Запуск Qt-приложения"""
    app = QApplication([])
    window = QMainWindow()
    window.setWindowTitle("MMAD калькулятор")
    window.resize(1000, 650)
    window.show()
    app.exec()

