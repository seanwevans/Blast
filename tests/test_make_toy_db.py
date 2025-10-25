import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import make_toy_db


def test_make_toy_db_creates_expected_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "toy.db"
    monkeypatch.setattr(make_toy_db, "DB_FILE", str(db_path))

    make_toy_db.make_toy_db()

    assert db_path.exists(), "Database file should be created"

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name, age, comment FROM test_data ORDER BY id")
        rows = cur.fetchall()
    finally:
        conn.close()

    expected = [
        ("Alice", 30, "Loves databases."),
        ("Bob", 25, "Enjoys GPU acceleration."),
        ("Charlie", 28, "Prefers OpenMP."),
        ("Diana", 35, "Thinks varints are cool."),
        ("Eve", 22, "Just here for the bytes."),
    ]

    assert rows == expected


def test_make_toy_db_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "toy.db"
    monkeypatch.setattr(make_toy_db, "DB_FILE", str(db_path))

    make_toy_db.make_toy_db()
    first_mtime = Path(db_path).stat().st_mtime

    make_toy_db.make_toy_db()
    second_mtime = Path(db_path).stat().st_mtime

    assert second_mtime >= first_mtime
