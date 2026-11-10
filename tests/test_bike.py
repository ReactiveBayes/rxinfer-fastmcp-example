"""Tests for the Eindhoven bike commute dataset.

The story for students: we *are* the world here. We know the hidden formula that
generates commute times, so we can check whether the Bayesian model recovers it,
and whether it gets honestly less certain when the weather is unknown.
"""

import csv
import hashlib
import subprocess
import sys

import pytest

from conftest import PROJECT_ROOT
from main import jl

FEATURES = [
    "distance_km", "wind_speed_ms", "rain_mm_per_hour", "traffic_lights",
    "temperature_c", "congestion", "bike_age_years",
]

# Demo conversation, step 1: "~5 km, ~5 traffic lights, the path is clear, perfect weather"
PERFECT_DAY = {
    "distance_km": 5.0, "traffic_lights": 5, "congestion": 0,
    "wind_speed_ms": 0.0, "rain_mm_per_hour": 0.0, "temperature_c": 20.0,
}
# Demo conversation, step 2: "it will start raining and the wind picks up to 5 m/s"
RAINY_WINDY_DAY = {**PERFECT_DAY, "rain_mm_per_hour": 2.0, "wind_speed_ms": 5.0}
# Planning tomorrow without a weather forecast
NO_FORECAST = {k: v for k, v in PERFECT_DAY.items() if k not in ("wind_speed_ms", "rain_mm_per_hour", "temperature_c")}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def true_coefficients() -> dict[str, float]:
    return {str(k): float(v) for k, v in jl.dataset_spec("bike").true_coefficients.items()}


def true_minutes(inputs: dict) -> float:
    """Noise-free commute time from the hidden formula (in basis form)."""
    c = true_coefficients()
    total = c["intercept"]
    for f in FEATURES:
        total += c[f] * inputs[f] + c[f + "^2"] * inputs[f] ** 2
    return total


async def estimate(client, inputs: dict) -> dict:
    return (await client.call_tool("estimate_bike_commute_time", inputs)).structured_content


@pytest.fixture(scope="session")
def bike_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "bike_commutes.csv"
    jl.write_dataset("bike", str(path))
    return path


@pytest.fixture(scope="session")
def bike_rows(bike_csv):
    with bike_csv.open() as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


@pytest.fixture(scope="session")
def trained_bike(bike_csv):
    result = jl.run_analysis(str(bike_csv), iterations=10)
    assert result["status"] == "success"
    return result


# --- Dataset -------------------------------------------------------------------


def test_bike_dataset_shape(bike_csv, bike_rows):
    with bike_csv.open() as f:
        assert next(csv.reader(f)) == FEATURES + ["commute_minutes"]
    assert len(bike_rows) == 20_000
    lo = {f: min(r[f] for r in bike_rows) for f in FEATURES}
    hi = {f: max(r[f] for r in bike_rows) for f in FEATURES}
    assert 0.5 <= lo["distance_km"] and hi["distance_km"] <= 15
    assert 0 <= lo["wind_speed_ms"] and hi["wind_speed_ms"] <= 12
    assert 0 <= lo["rain_mm_per_hour"] and hi["rain_mm_per_hour"] <= 10
    assert -5 <= lo["temperature_c"] and hi["temperature_c"] <= 30
    assert (lo["traffic_lights"], hi["traffic_lights"]) == (0, 15)
    assert (lo["congestion"], hi["congestion"]) == (0, 10)
    assert (lo["bike_age_years"], hi["bike_age_years"]) == (0, 20)
    dry = sum(r["rain_mm_per_hour"] == 0 for r in bike_rows) / len(bike_rows)
    assert dry == pytest.approx(0.6, abs=0.02)  # it's the Netherlands, but not *always* raining


def test_bike_dataset_is_reproducible(tmp_path):
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    jl.write_dataset("bike", str(a), n=300, seed=1)
    jl.write_dataset("bike", str(b), n=300, seed=1)
    jl.write_dataset("bike", str(c), n=300, seed=2)
    assert sha256(a) == sha256(b)
    assert sha256(a) != sha256(c)


def test_generator_follows_the_hidden_formula(bike_rows):
    residuals = [r["commute_minutes"] - true_minutes(r) for r in bike_rows]
    mean = sum(residuals) / len(residuals)
    sd = (sum((e - mean) ** 2 for e in residuals) / len(residuals)) ** 0.5
    assert mean == pytest.approx(0.0, abs=0.05)
    assert sd == pytest.approx(1.5, rel=0.03)


def test_cli_show_formula():
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "generate_dataset.py"), "bike", "--show-formula"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    assert "commute_minutes" in proc.stdout
    for term in ["intercept", "distance_km", "wind_speed_ms^2", "bike_age_years"]:
        assert term in proc.stdout


# --- Recovering the hidden formula ------------------------------------------------


def test_model_recovers_the_hidden_formula(trained_bike):
    state = jl.seval('name -> STATES[name]')("bike")
    names = [str(n) for n in jl.feature_function_names(jl.dataset_spec("bike").features)]
    means = list(state.parameters.omega_mean)
    sds = [float(v) ** 0.5 for v in jl.seval("diag")(state.parameters.omega_covariance)]
    truth = true_coefficients()
    for name, m, sd in zip(names, means, sds):
        assert abs(m - truth[name]) <= 5 * sd + 1e-3, f"{name}: learned {m:.4f} ± {sd:.4f}, true {truth[name]}"


def test_model_recovers_the_noise_level(trained_bike):
    state = jl.seval('name -> STATES[name]')("bike")
    precision = state.parameters.noise_shape * state.parameters.noise_scale
    assert float(precision) ** -0.5 == pytest.approx(1.5, rel=0.03)


# --- The demo conversation -------------------------------------------------------------


async def test_perfect_day_estimate(trained_bike, client):
    out = await estimate(client, PERFECT_DAY)
    assert out["status"] == "success"
    assert out["estimated_minutes"] == pytest.approx(true_minutes({**PERFECT_DAY, "bike_age_years": 0}), abs=0.3)
    assert out["minutes_std"] == pytest.approx(1.5, rel=0.1)
    assert out["missing_features"] == ["bike_age_years"]


async def test_rain_and_wind_make_the_ride_longer(trained_bike, client):
    perfect = await estimate(client, PERFECT_DAY)
    rainy = await estimate(client, RAINY_WINDY_DAY)
    expected_delta = true_minutes({**RAINY_WINDY_DAY, "bike_age_years": 0}) - true_minutes({**PERFECT_DAY, "bike_age_years": 0})
    assert expected_delta == pytest.approx(5.4)
    assert rainy["estimated_minutes"] - perfect["estimated_minutes"] == pytest.approx(expected_delta, abs=0.3)


async def test_unknown_weather_means_more_uncertainty(trained_bike, client):
    known = await estimate(client, PERFECT_DAY)
    unknown = await estimate(client, NO_FORECAST)
    assert unknown["minutes_std"] >= 2 * known["minutes_std"]
    assert sorted(unknown["missing_features"]) == sorted(
        ["wind_speed_ms", "rain_mm_per_hour", "temperature_c", "bike_age_years"]
    )


async def test_unknown_bike_age_does_not_matter(trained_bike, client):
    unknown_age = await estimate(client, PERFECT_DAY)
    known_age = await estimate(client, {**PERFECT_DAY, "bike_age_years": 10})
    assert unknown_age["minutes_std"] == pytest.approx(known_age["minutes_std"], rel=0.02)
    assert unknown_age["estimated_minutes"] == pytest.approx(known_age["estimated_minutes"], abs=0.1)


# --- Several datasets side by side -----------------------------------------------------


async def test_run_bayesian_analysis_detects_the_dataset(isolated_state, client, bike_csv):
    out = (
        await client.call_tool("run_bayesian_analysis", {"file_path": str(bike_csv), "iterations": 2})
    ).structured_content
    assert out["status"] == "success"
    assert out["dataset"] == "bike"
    assert out["num_rows_trained"] == 20_000
    assert out["num_features"] == len(FEATURES)


async def test_unknown_dataset_is_an_error(isolated_state, client, tmp_path):
    path = tmp_path / "mystery.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    out = (await client.call_tool("run_bayesian_analysis", {"file_path": str(path)})).structured_content
    assert out["status"] == "error"
    assert "commute_minutes" in out["message"] and "price" in out["message"]


async def test_both_models_live_side_by_side(trained_houses, trained_bike, client, golden_houses):
    from test_houses import FULL_HOUSE

    house = (await client.call_tool("estimate_house_price", FULL_HOUSE)).structured_content
    golden_houses.check("full_price", house["estimated_price"])
    bike = await estimate(client, PERFECT_DAY)
    assert bike["status"] == "success"


async def test_describe_data_per_dataset(trained_houses, trained_bike, client):
    bike = (await client.call_tool("describe_data", {"dataset": "bike"})).structured_content
    assert bike["status"] == "success"
    assert bike["dataset"] == "bike"
    assert bike["column_details"]["wind_speed_ms"]["known_field"] is True
    house = (await client.call_tool("describe_data", {"dataset": "houses"})).structured_content
    assert "price" in house["columns"]


async def test_estimate_before_training_bike_is_an_error(untrained, client):
    out = await estimate(client, PERFECT_DAY)
    assert out["status"] == "error"
