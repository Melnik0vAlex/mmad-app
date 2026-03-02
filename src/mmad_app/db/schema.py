# -*- coding: utf-8 -*-
# src/mmad_app/db/schema.py

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    sample_code     TEXT NOT NULL,
    fpf_cutoff_um   REAL NOT NULL,
    total_mass_ug   REAL NOT NULL,

    mmad            REAL NOT NULL,
    gsd             REAL NOT NULL,
    d10             REAL NOT NULL,
    d16             REAL NOT NULL,
    d84             REAL NOT NULL,
    d90             REAL NOT NULL,
    d15_87          REAL NOT NULL,
    d84_13          REAL NOT NULL,
    span            REAL NOT NULL,
    fpf_pct         REAL NOT NULL,

    log_mean        REAL NOT NULL,
    mass_mean       REAL NOT NULL,
    modal           REAL NOT NULL,

    mmad_ls         REAL NOT NULL,
    kor_k           REAL NOT NULL,
    sigma           REAL NOT NULL,
    r               REAL NOT NULL,
    slope           REAL NOT NULL,
    intercept       REAL NOT NULL,
    se_slope        REAL NOT NULL,
    se_intercept    REAL NOT NULL,
    r2              REAL NOT NULL,
    syx             REAL NOT NULL,
    f_stat          REAL NOT NULL,
    df              INT NOT NULL,
    ss_reg          REAL NOT NULL,
    ss_res          REAL NOT NULL,

    notes           TEXT
);

CREATE TABLE IF NOT EXISTS run_stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL,
    stage_name  TEXT NOT NULL,
    d_low       REAL NULL,
    d_high      REAL NULL,
    mass        REAL NOT NULL,

    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_stages_run_id ON run_stages(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);

-- Опционально: быстрый поиск по шифру пробы
CREATE INDEX IF NOT EXISTS idx_runs_sample_code ON runs(sample_code);
"""


def init_db(conn: sqlite3.Connection) -> None:
    """Создаёт таблицы, если их ещё нет."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
