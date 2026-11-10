"""Generate a synthetic dataset for the demo.

Usage:
    uv run python generate_dataset.py houses
    uv run python generate_dataset.py houses --n 5000 --seed 7 --out my_houses.csv

The generators live in src/datasets.jl. The same seed always produces exactly
the same CSV, so demos are reproducible.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Kept in sync with the datasets registered in src/datasets.jl (checked by the tests).
DATASETS = ["houses"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset as CSV.")
    parser.add_argument("dataset", choices=DATASETS, help="which dataset to generate")
    parser.add_argument("--n", type=int, default=None, help="number of samples to draw (default: dataset-specific)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: dataset-specific)")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path (default: dataset-specific, in the project root)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Import Julia only after argument parsing, so `--help` is instant.
    from juliacall import Main as jl

    jl.include(str(ROOT / "src" / "datasets.jl"))
    spec = jl.dataset_spec(args.dataset)
    out = args.out or ROOT / str(spec.default_file)

    rows = jl.write_dataset(args.dataset, str(out), n=args.n, seed=args.seed)
    print(f"Wrote {rows} rows of '{args.dataset}' to {out}")


if __name__ == "__main__":
    main()
