# -*- coding: utf-8 -*-
# src/mmad_app/db/repo.py

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from mmad_app.core.models import StageRecord
from mmad_app.core.mmad import MmadResult
from mmad_app.db.schema import init_db


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
    result: MmadResult,
    notes: Optional[str] = None,
) -> int:
    """
    Сохраняет расчёт и связанные ступени в БД.

    Parameters
    ----------
    sample_code:
        Шифр пробы (вводится пользователем). Рекомендуется не пустая строка.
    records:
        Исходные данные по ступеням (интервалы и массы).
    result:
        Результаты расчёта (MMAD, GSD, FPF и т.д.).
    notes:
        Необязательное поле комментария.

    Returns
    -------
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
            span, fpf_pct, log_mean, mass_mean, modal, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            code,
            float(result.fpf_cutoff_um),
            float(result.total_mass),
            float(result.mmad),
            float(result.gsd),
            float(result.d10),
            float(result.d16),
            float(result.d84),
            float(result.d90),
            float(result.d15_87),
            float(result.d84_13),
            float(result.span),
            float(result.fpf_pct),
            float(result.log_mean),
            float(result.mass_mean),
            float(result.modal),
            notes,
        ),
    )
    row_id = cur.lastrowid
    if row_id is None:
        raise RuntimeError("SQLite: не удалось получить lastrowid после INSERT.")
    run_id = int(row_id)

    stage_rows = [
        (
            run_id,
            str(r.name),
            float(r.d_low),
            float(r.d_high),
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
        SELECT id, created_at, sample_code, mmad, gsd, fpf_pct, total_mass_ug
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