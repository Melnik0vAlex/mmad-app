# Структура приложения

```txt
mmad-app/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml
├─ requirements-dev.txt
├─ src/
│  └─ mmad_app/
│     ├─ __init__.py
│     ├─ __main__.py          # python -m mmad_app
│     ├─ app.py               # точка входа GUI (Qt)
│     ├─ core/
│     │  ├─ __init__.py
│     │  ├─ models.py         # dataclass: StageRecord
│     │  └─ mmad.py           # расчёт MMAD + интерполяция
│     ├─ ui/
│     │  ├─ __init__.py
│     │  ├─ main_window.py    # QMainWindow + таблица
│     │  └─ plot_widget.py    # matplotlib canvas для Qt
│     └─ resources/
│        └─ icons/            # иконки приложения
├─ tests/
│  ├─ __init__.py
│  └─ test_mmad.py            # тесты ядра (pytest)
└─ data/
   └─ example.csv             # пример входных данных
```
