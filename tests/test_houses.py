"""Characterization tests for the original house-price demo.

These pin down the behaviour students saw in the original demo, so that
refactoring and adding new datasets cannot silently break it.
"""

import csv
import hashlib
import json
import os

import pytest

FULL_HOUSE = {
    "bedrooms": 2,
    "bathrooms": 1,
    "living_area": 120,
    "floors": 1,
    "basement_area": 30,
    "yr_built": 1980,
    "label": 5.0,
}
PARTIAL_HOUSE = {k: v for k, v in FULL_HOUSE.items() if k not in ("yr_built", "label")}
EMPTY_HOUSE: dict = {}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def estimate(client, house: dict) -> dict:
    result = await client.call_tool("estimate_house_price", house)
    return result.structured_content


# --- Dataset -----------------------------------------------------------------


def test_house_dataset_is_reproducible(houses_csv, golden_houses):
    golden_houses.check("csv_sha256", sha256(houses_csv))
    with houses_csv.open() as f:
        rows = list(csv.DictReader(f))
    golden_houses.check("csv_rows", len(rows))
    assert list(rows[0].keys()) == [
        "bedrooms", "bathrooms", "living_area", "floors",
        "basement_area", "yr_built", "label", "price",
    ]


# --- Before training ----------------------------------------------------------


async def test_estimate_before_training_is_an_error(untrained, client):
    out = await estimate(client, FULL_HOUSE)
    assert out["status"] == "error"


async def test_describe_before_training_is_an_error(untrained, client):
    out = (await client.call_tool("describe_data", {})).structured_content
    assert out["status"] == "error"


# --- Training -----------------------------------------------------------------


async def test_run_bayesian_analysis_tool(isolated_state, client, houses_csv, golden_houses):
    out = (
        await client.call_tool("run_bayesian_analysis", {"file_path": str(houses_csv), "iterations": 2})
    ).structured_content
    assert out["status"] == "success"
    golden_houses.check("csv_rows", out["num_houses_trained"])


@pytest.mark.parametrize("defect", ["missing_column", "empty_cell"])
async def test_incomplete_training_csv_reports_error(isolated_state, client, houses_csv, tmp_path, defect):
    # Known limitation: the model needs complete training data. It must fail
    # gracefully (status=error) instead of crashing the server.
    path = tmp_path / f"{defect}.csv"
    with houses_csv.open() as src, path.open("w", newline="") as dst:
        reader = csv.DictReader(src)
        fields = [f for f in reader.fieldnames if not (defect == "missing_column" and f == "bedrooms")]
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(reader):
            if i >= 500:
                break
            if defect == "empty_cell" and i == 10:
                row["bedrooms"] = ""
            writer.writerow({k: row[k] for k in fields})
    out = (
        await client.call_tool("run_bayesian_analysis", {"file_path": str(path), "iterations": 2})
    ).structured_content
    assert out["status"] == "error"
    assert "bedrooms" in out["message"]


# --- Predictions ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, house",
    [("full", FULL_HOUSE), ("partial", PARTIAL_HOUSE), ("empty", EMPTY_HOUSE)],
)
async def test_golden_predictions(trained_houses, client, golden_houses, name, house):
    out = await estimate(client, house)
    assert out["status"] == "success"
    golden_houses.check(f"{name}_price", out["estimated_price"])
    golden_houses.check(f"{name}_std", out["price_std"])
    assert sorted(out["specified_features"]) == sorted(house)
    assert sorted(out["missing_features"]) == sorted(set(FULL_HOUSE) - set(house))


async def test_less_information_means_more_uncertainty(trained_houses, client):
    full = await estimate(client, FULL_HOUSE)
    partial = await estimate(client, PARTIAL_HOUSE)
    empty = await estimate(client, EMPTY_HOUSE)
    assert full["price_std"] < partial["price_std"] < empty["price_std"]


async def test_knowing_nothing_predicts_the_population(trained_houses, client, houses_csv):
    # With every feature unknown, the law of total variance averages over all
    # training houses: the estimate is the average price, the spread their spread.
    with houses_csv.open() as f:
        prices = [float(r["price"]) for r in csv.DictReader(f)]
    avg = sum(prices) / len(prices)
    sd = (sum((p - avg) ** 2 for p in prices) / (len(prices) - 1)) ** 0.5
    out = await estimate(client, EMPTY_HOUSE)
    assert out["estimated_price"] == pytest.approx(avg, rel=0.01)
    assert out["price_std"] == pytest.approx(sd, rel=0.05)


async def test_in_sample_accuracy(trained_houses, client, houses_csv, golden_houses):
    with houses_csv.open() as f:
        rows = [r for _, r in zip(range(100), csv.DictReader(f))]
    errors = []
    for row in rows:
        house = {k: float(v) for k, v in row.items() if k != "price"}
        out = await estimate(client, house)
        errors.append(abs(out["estimated_price"] - float(row["price"])))
    mae = sum(errors) / len(errors)
    if os.environ.get("RECORD_GOLDEN") == "1":
        golden_houses.check("mae_100", mae)
    else:
        # One-sided with a little slack: the model may get better, not worse.
        assert mae <= golden_houses.get("mae_100") * 1.05


async def test_describe_data(trained_houses, client):
    out = (await client.call_tool("describe_data", {})).structured_content
    assert out["status"] == "success"
    assert "price" in out["columns"]
    assert out["column_details"]["price"]["known_field"] is True


# --- Tool contract ----------------------------------------------------------------


async def test_tool_schemas(client, golden_tools):
    tools = await client.list_tools()
    names = sorted(t.name for t in tools)
    for t in tools:
        schema = json.loads(json.dumps({"input": t.inputSchema, "output": t.outputSchema}))
        golden_tools.check(t.name, schema)
    golden_tools.check("_names", names)
