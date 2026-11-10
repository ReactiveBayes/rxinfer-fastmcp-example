"""Generate a synthetic dataset for the demo.

Usage:
    uv run python generate_dataset.py houses
    uv run python generate_dataset.py bike --n 5000 --seed 7 --out my_rides.csv
    uv run python generate_dataset.py bike --show-formula   # reveal the hidden formula

The generators live in src/datasets.jl. The same seed always produces exactly
the same CSV, so demos are reproducible.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Kept in sync with the datasets registered in src/datasets.jl (checked by the tests).
DATASETS = ["houses", "bike"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset as CSV.")
    parser.add_argument("dataset", choices=DATASETS, help="which dataset to generate")
    parser.add_argument("--n", type=int, default=None, help="number of samples to draw (default: dataset-specific)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: dataset-specific)")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path (default: dataset-specific, in the project root)")
    parser.add_argument(
        "--show-formula", action="store_true",
        help="print the hidden formula behind the dataset instead of generating it",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # Import Julia only after argument parsing, so `--help` is instant.
    from juliacall import Main as jl

    jl.include(str(ROOT / "src" / "datasets.jl"))
    spec = jl.dataset_spec(args.dataset)

    if args.show_formula:
        show_formula(jl, spec)
        return

    out = args.out or ROOT / str(spec.default_file)

    rows = jl.write_dataset(args.dataset, str(out), n=args.n, seed=args.seed)
    print(f"Wrote {rows} rows of '{args.dataset}' to {out}")


def show_formula(jl, spec) -> None:
    print(f"Hidden formula behind the '{spec.name}' dataset:\n")
    print(str(spec.formula))
    if spec.true_coefficients is jl.nothing:
        return
    coefficients = {str(k): float(v) for k, v in spec.true_coefficients.items()}
    print(f"\nThe same formula in the model's basis (compare with the describe_model tool):\n")
    for term in jl.feature_function_names(spec.features):
        print(f"  {str(term):<24} {coefficients[str(term)]:>8.4f}")
    print(f"  {'noise std':<24} {float(spec.noise_std):>8.4f}")


if __name__ == "__main__":
    main()
