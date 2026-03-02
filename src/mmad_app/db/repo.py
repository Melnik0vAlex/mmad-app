# -*- coding: utf-8 -*-
# src/mmad_app/db/repo.py

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import MmadResult, MmadResultLS
from mmad_app.db.schema import init_db


def _to_float_or_none(x: Optional[float]) -> Optional[float]:
    """Приводит Optional[float] к Optional[float] для записи в БД."""
    return None if x is None else float(x)


def connect(db_path: str) -> sqlite3.Connection:
    """Открывает соединение с SQLite и инициализирует схему."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def save_run(
    conn: sqlite3.Connection,
    *,
    sample_code: str,
    records: List[StageRecord],
    result_lp: MmadResult,
    result_ls: MmadResultLS,
    notes: Optional[str] = None,
) -> int:
    """
    Сохраняет расчёт и связанные ступени в БД.

    Параметры
    ---------
    sample_code:
        Шифр пробы (вводится пользователем). Рекомендуется не пустая строка.
    records:
        Исходные данные по ступеням (интервалы и массы).
    result:
        Результаты расчёта (MMAD, GSD, FPF и т.д.).
    notes:
        Необязательное поле комментария.

    Возвращает
    ----------
    int
        ID сохранённой записи runs.id.
    """
    code = sample_code.strip()
    if not code:
        raise ValueError("Шифр пробы (sample_code) не должен быть пустым.")

    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO runs (
            created_at, sample_code, fpf_cutoff_um, total_mass_ug,
            mmad, gsd, d10, d16, d84, d90, d15_87, d84_13,
            span, fpf_pct, log_mean, mass_mean, modal,
            mmad_ls, kor_k, sigma, r, slope, intercept, se_slope,
            se_intercept, r2, syx, f_stat, df, ss_reg, ss_res,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?)
        """,
        (
            created_at,
            code,
            float(result_lp.fpf_cutoff_um),
            float(result_lp.total_mass),
            float(result_lp.mmad),
            float(result_lp.gsd),
            float(result_lp.d10),
            float(result_lp.d16),
            float(result_lp.d84),
            float(result_lp.d90),
            float(result_lp.d15_87),
            float(result_lp.d84_13),
            float(result_lp.span),
            float(result_lp.fpf_pct),
            float(result_lp.log_mean),
            float(result_lp.mass_mean),
            float(result_lp.modal),
            float(result_ls.mmad),
            float(result_ls.kor_k),
            float(result_ls.sigma),
            float(result_ls.r),
            float(result_ls.slope),
            float(result_ls.intercept),
            float(result_ls.se_slope),
            float(result_ls.se_intercept),
            float(result_ls.r2),
            float(result_ls.syx),
            float(result_ls.f_stat),
            float(result_ls.df),
            float(result_ls.ss_reg),
            float(result_ls.ss_res),
            notes,
        ),
    )
    row_id = cur.lastrowid
    if row_id is None:
        raise RuntimeError("SQLite: не удалось получить lastrowid после INSERT.")
    run_id = int(row_id)

    stage_rows = [
        (
            int(run_id),
            str(r.name),
            _to_float_or_none(r.d_low),  # NULL, если None
            _to_float_or_none(r.d_high),  # NULL, если None
            float(r.mass),
        )
        for r in records
    ]

    cur.executemany(
        """
        INSERT INTO run_stages (run_id, stage_name, d_low, d_high, mass)
        VALUES (?, ?, ?, ?, ?)
        """,
        stage_rows,
    )

    conn.commit()
    return run_id


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> List[sqlite3.Row]:
    """Список последних расчётов (для истории)."""
    cur = conn.execute(
        """
        SELECT id, created_at, sample_code, mmad, gsd, d10, d16, d84, d90,
        span, fpf_pct, log_mean, mass_mean, modal,
        mmad_ls, kor_k, sigma, r, slope, intercept, se_slope,
        se_intercept, r2, syx, f_stat, df, ss_reg, ss_res
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    return list(cur.fetchall())


def load_run(
    conn: sqlite3.Connection, run_id: int
) -> Tuple[sqlite3.Row, List[sqlite3.Row]]:
    """Загружает расчёт и его ступени по ID."""
    run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),)).fetchone()
    if run_row is None:
        raise ValueError(f"Расчёт run_id={run_id} не найден.")

    stage_rows = conn.execute(
        """
        SELECT stage_name, d_low, d_high, mass
        FROM run_stages
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (int(run_id),),
    ).fetchall()

    return run_row, list(stage_rows)


def delete_run(conn: sqlite3.Connection, run_id: int) -> None:
    """Удаляет расчёт и связанные ступени (через ON DELETE CASCADE)."""
    conn.execute("DELETE FROM runs WHERE id = ?", (int(run_id),))
    conn.commit()
