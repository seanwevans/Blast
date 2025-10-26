#!/usr/bin/env python3
"""
Benchmark BLAST against the reference sqlite3 CLI dumper.

This script measures how long it takes for BLAST to dump a SQLite table to CSV
compared to the stock `sqlite3` command line tool. By default it exports the
first user table that appears in the database; a different table can be
selected via ``--table``.

Usage examples
--------------

Benchmark a user-provided database:

    python benchmark_sqlite3.py path/to/database.db

Run the benchmark against a generated toy database:

    python benchmark_sqlite3.py

To keep the generated CSV artifacts for later inspection, pass
``--output-dir dumps/``.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
BLAST_SOURCE = REPO_ROOT / "blast.cu"


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot be executed."""


def _ensure_sqlite3_available() -> str:
    sqlite3_cli = shutil.which("sqlite3")
    if not sqlite3_cli:
        raise BenchmarkError("sqlite3 CLI executable not found in PATH")
    return sqlite3_cli


def _discover_table(db_path: Path) -> str:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            LIMIT 1
            """
        )
        row = cur.fetchone()

    if not row or not row[0]:
        raise BenchmarkError(
            f"No user tables were found inside {db_path}. Specify --table explicitly."
        )
    return row[0]


def _build_blast_binary(target_path: Path) -> None:
    if not BLAST_SOURCE.exists():
        raise BenchmarkError(f"Unable to locate BLAST source file at {BLAST_SOURCE}")

    compile_cmd = [
        "gcc",
        "-O3",
        "-fopenmp",
        "-x",
        "c",
        str(BLAST_SOURCE),
        "-o",
        str(target_path),
    ]
    subprocess.run(
        compile_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _run_sqlite3_dump(
    sqlite3_cli: str,
    db_path: Path,
    table: str,
    output_path: Path,
) -> float:
    safe_table = table.replace('"', '""')
    script = (
        "\n".join(
            [
                ".mode csv",
                ".headers off",
                f".once {output_path}",
                f'SELECT rowid, * FROM "{safe_table}";',
            ]
        )
        + "\n"
    )

    start = time.perf_counter()
    subprocess.run(
        [sqlite3_cli, str(db_path)],
        input=script,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    end = time.perf_counter()

    if not output_path.exists():
        raise BenchmarkError("sqlite3 CLI did not produce the expected output file")

    return end - start


def _run_blast_dump(
    blast_binary: Path,
    db_path: Path,
    output_path: Path,
    use_cuda: bool,
) -> float:
    if not blast_binary.exists():
        raise BenchmarkError(f"BLAST binary not found at {blast_binary}")

    cmd: List[str] = [str(blast_binary)]
    if use_cuda:
        cmd.append("--cuda")
    cmd.extend([str(db_path), str(output_path)])

    start = time.perf_counter()
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    end = time.perf_counter()

    if not output_path.exists():
        raise BenchmarkError("BLAST did not produce the expected CSV output")

    return end - start


def benchmark(
    db_path: Path,
    *,
    table: Optional[str] = None,
    runs: int = 3,
    blast_binary: Optional[Path] = None,
    use_cuda: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Run the benchmark and return a summary dictionary."""

    db_path = Path(db_path)
    if not db_path.exists():
        raise BenchmarkError(f"Database file not found: {db_path}")
    if db_path.is_dir():
        raise BenchmarkError(f"Expected a database file, got directory: {db_path}")
    if runs <= 0:
        raise BenchmarkError("runs must be a positive integer")

    sqlite3_cli = _ensure_sqlite3_available()
    table_name = table or _discover_table(db_path)

    binary_tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None
    output_tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None

    try:
        if blast_binary is None:
            binary_tmpdir = tempfile.TemporaryDirectory()
            blast_binary_path = Path(binary_tmpdir.name) / "blast"
            _build_blast_binary(blast_binary_path)
        else:
            blast_binary_path = Path(blast_binary)

        if output_dir is None:
            output_tmpdir = tempfile.TemporaryDirectory()
            outputs_root = Path(output_tmpdir.name)
        else:
            outputs_root = Path(output_dir)
            outputs_root.mkdir(parents=True, exist_ok=True)

        sqlite_durations: List[float] = []
        blast_durations: List[float] = []
        sqlite_sizes: List[int] = []
        blast_sizes: List[int] = []

        for idx in range(runs):
            sqlite_output = outputs_root / f"sqlite3_run{idx}.csv"
            blast_output = outputs_root / f"blast_run{idx}.csv"

            sqlite_elapsed = _run_sqlite3_dump(
                sqlite3_cli, db_path, table_name, sqlite_output
            )
            sqlite_durations.append(sqlite_elapsed)
            sqlite_sizes.append(sqlite_output.stat().st_size)

            blast_elapsed = _run_blast_dump(
                blast_binary_path, db_path, blast_output, use_cuda
            )
            blast_durations.append(blast_elapsed)
            blast_sizes.append(blast_output.stat().st_size)

        sqlite_avg = mean(sqlite_durations)
        blast_avg = mean(blast_durations)
        speedup = sqlite_avg / blast_avg if blast_avg > 0 else float("inf")

        return {
            "table": table_name,
            "runs": runs,
            "output_dir": str(outputs_root),
            "sqlite3": {
                "times": sqlite_durations,
                "average": sqlite_avg,
                "sizes": sqlite_sizes,
            },
            "blast": {
                "times": blast_durations,
                "average": blast_avg,
                "sizes": blast_sizes,
            },
            "speedup": speedup,
        }
    finally:
        if binary_tmpdir is not None:
            binary_tmpdir.cleanup()
        if output_tmpdir is not None and output_dir is None:
            output_tmpdir.cleanup()


def _format_timings(values: Iterable[float]) -> str:
    return ", ".join(f"{v * 1000:.2f} ms" for v in values)


def _format_sizes(values: Iterable[int]) -> str:
    return ", ".join(f"{v / (1024 * 1024):.2f} MiB" for v in values)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark BLAST against sqlite3")
    parser.add_argument(
        "db",
        nargs="?",
        type=Path,
        help=(
            "Path to the SQLite database to benchmark. If omitted, a temporary "
            "toy database is generated using make_toy_db.py."
        ),
    )
    parser.add_argument(
        "--table",
        help="Name of the table to export. Defaults to the first user table.",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="How many times to run each dumper."
    )
    parser.add_argument(
        "--blast-binary",
        type=Path,
        help="Use an existing BLAST binary instead of compiling blast.cu.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where CSV outputs should be stored. Defaults to a temporary folder.",
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Run BLAST with the --cuda flag enabled."
    )

    args = parser.parse_args(argv)

    temp_db_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    db_path: Path

    if args.db is None:
        temp_db_dir = tempfile.TemporaryDirectory()
        db_path = Path(temp_db_dir.name) / "toy.db"
        toy_module = None
        previous = None
        try:
            import make_toy_db as toy_module  # type: ignore[import-not-found]

            previous = toy_module.DB_FILE
            toy_module.DB_FILE = str(db_path)
            toy_module.make_toy_db()
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive, should not happen in tests
            raise BenchmarkError(f"Failed to generate toy database: {exc}") from exc
        finally:
            if toy_module is not None and previous is not None:
                toy_module.DB_FILE = previous
        print(f"[+] Generated toy database at {db_path}")
    else:
        db_path = args.db

    try:
        summary = benchmark(
            db_path,
            table=args.table,
            runs=args.runs,
            blast_binary=args.blast_binary,
            use_cuda=args.cuda,
            output_dir=args.output_dir,
        )
    except BenchmarkError as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print("[!] A subprocess failed while running the benchmark", file=sys.stderr)
        print(exc, file=sys.stderr)
        return exc.returncode or 1
    finally:
        if temp_db_dir is not None:
            temp_db_dir.cleanup()

    sqlite_times = summary["sqlite3"]["times"]  # type: ignore[index]
    blast_times = summary["blast"]["times"]  # type: ignore[index]
    sqlite_sizes = summary["sqlite3"]["sizes"]  # type: ignore[index]
    blast_sizes = summary["blast"]["sizes"]  # type: ignore[index]

    print("\n=== Benchmark Summary ===")
    print(f"Database: {db_path}")
    print(f"Table: {summary['table']}")
    print(f"Runs: {summary['runs']}")
    print(f"Outputs stored in: {summary['output_dir']}")
    print()
    print("sqlite3 CLI:")
    print(f"  Times: {_format_timings(sqlite_times)}")
    print(f"  Sizes: {_format_sizes(sqlite_sizes)}")
    print(f"  Average: {summary['sqlite3']['average']:.4f} s")
    print()
    print("BLAST:")
    print(f"  Times: {_format_timings(blast_times)}")
    print(f"  Sizes: {_format_sizes(blast_sizes)}")
    print(f"  Average: {summary['blast']['average']:.4f} s")
    print()
    print(f"Speedup (sqlite3 / BLAST): {summary['speedup']:.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
