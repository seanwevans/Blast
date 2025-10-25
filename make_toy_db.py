#!/usr/bin/env python3
"""
make_toy_db.py
Generates a simple SQLite database for testing the BLAST dumper.
"""

import sqlite3
import os

DB_FILE = "toy.db"


def make_toy_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Create a table with multiple data types
    cur.execute(
        """
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            comment TEXT
        );
    """
    )

    # Insert a few rows
    rows = [
        ("Alice", 30, "Loves databases."),
        ("Bob", 25, "Enjoys GPU acceleration."),
        ("Charlie", 28, "Prefers OpenMP."),
        ("Diana", 35, "Thinks varints are cool."),
        ("Eve", 22, "Just here for the bytes."),
    ]
    cur.executemany("INSERT INTO test_data (name, age, comment) VALUES (?, ?, ?)", rows)

    conn.commit()
    conn.close()
    print(f"[+] Created test database: {DB_FILE}")


if __name__ == "__main__":
    make_toy_db()
