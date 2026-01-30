# -*- coding: utf-8 -*-
# src/mmad_app/db/path.py

from __future__ import annotations

import sys
from pathlib import Path


def get_app_base_dir() -> Path:
    """
    Возвращает директорию, где расположен исполняемый файл приложения.

    - При запуске из PyInstaller (.exe): это папка с .exe
    - При запуске из исходников: это папка с текущим файлом/проектом
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "executable"):
        # PyInstaller: sys.executable указывает на путь к .exe
        return Path(sys.executable).resolve().parent

    # Dev-режим: рядом с проектом
    return Path.cwd().resolve()


def get_db_path() -> Path:
    """Путь к SQLite базе данных рядом с приложением."""
    return get_app_base_dir() / "mmad.sqlite3"
