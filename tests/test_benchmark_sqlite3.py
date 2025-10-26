import shutil

import pytest

import make_toy_db
import benchmark_sqlite3


@pytest.mark.skipif(shutil.which("sqlite3") is None, reason="sqlite3 CLI is required for the benchmark")
def test_benchmark_produces_outputs(tmp_path, monkeypatch, blast_binary):
    db_path = tmp_path / "toy.db"
    outputs_dir = tmp_path / "outputs"

    monkeypatch.setattr(make_toy_db, "DB_FILE", str(db_path))
    make_toy_db.make_toy_db()

    summary = benchmark_sqlite3.benchmark(
        db_path,
        runs=1,
        blast_binary=blast_binary,
        output_dir=outputs_dir,
    )

    assert summary["runs"] == 1
    assert summary["table"] == "test_data"
    assert summary["sqlite3"]["times"] and summary["blast"]["times"]
    assert summary["sqlite3"]["sizes"][0] > 0
    assert summary["blast"]["sizes"][0] > 0

    sqlite_output = outputs_dir / "sqlite3_run0.csv"
    blast_output = outputs_dir / "blast_run0.csv"

    assert sqlite_output.exists()
    assert blast_output.exists()
