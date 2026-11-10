"""Tests for the dataset generator CLI (generate_dataset.py)."""

import csv
import hashlib
import subprocess
import sys

import generate_dataset
from conftest import PROJECT_ROOT
from main import jl


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_dataset_choices_match_julia_registry():
    registered = sorted(str(k) for k in jl.seval("collect(keys(DATASETS))"))
    assert sorted(generate_dataset.DATASETS) == registered


def test_cli_generates_reproducible_csv(tmp_path):
    out = tmp_path / "houses.csv"
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "generate_dataset.py"), "houses", "--n", "200", "--seed", "1", "--out", str(out)],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "houses" in proc.stdout

    # Same seed & size from within the running Julia session → byte-identical file.
    again = tmp_path / "again.csv"
    jl.write_dataset("houses", str(again), n=200, seed=1)
    assert sha256(out) == sha256(again)

    with out.open() as f:
        assert next(csv.reader(f))[-1] == "price"


def test_different_seed_gives_different_data(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    jl.write_dataset("houses", str(a), n=200, seed=1)
    jl.write_dataset("houses", str(b), n=200, seed=2)
    assert sha256(a) != sha256(b)
