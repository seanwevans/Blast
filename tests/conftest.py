import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def blast_binary(tmp_path_factory, project_root):
    build_dir = tmp_path_factory.mktemp("blast-build")
    binary_path = build_dir / "blast"

    subprocess.run(
        [
            "gcc",
            "-O3",
            "-fopenmp",
            "-x",
            "c",
            str(project_root / "blast.cu"),
            "-o",
            str(binary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return binary_path
