"""Unit tests for the dataset-agnostic model in src/model.jl.

Each test builds a tiny synthetic dataset whose answer is known in closed form.
"""

import math

import pytest

from main import jl

# Build data in Julia so the model receives exactly the types it expects.
make_data = jl.seval(
    """
    (f, n, seed, d) -> begin
        rng = StableRNG(seed)
        x = [Union{Float64, Missing}[10 * rand(rng) for _ in 1:d] for _ in 1:n]
        y = [f(xi) + 0.1 * randn(rng) for xi in x]
        return x, y
    end
    """
)
train = jl.seval("(x, y) -> learn_parameters(InferenceArguments(number_of_iterations = 10), x, y)[2]")
estimate = jl.seval(
    "(state, x) -> estimate(state, Union{Float64, Missing}[v === nothing ? missing : v for v in x])"
)


@pytest.fixture(scope="module")
def offset_state():
    # A constant offset can only be captured by an intercept term.
    x, y = make_data(jl.seval("x -> 10.0 + 0.5 * x[1]"), 2000, 1, 1)
    return train(x, y)


@pytest.fixture(scope="module")
def linear_state():
    # y = 3·x₁ + x₂ with x₁, x₂ ~ U(0, 10)
    x, y = make_data(jl.seval("x -> 3.0 * x[1] + x[2]"), 5000, 2, 2)
    return train(x, y)


def test_basis_has_linear_quadratic_and_intercept_terms():
    phis = jl.create_feature_functions(2)
    values = [float(phi(jl.seval("[2.0, 3.0]"))) for phi in phis]
    assert values == [2.0, 3.0, 4.0, 9.0, 1.0]


def test_intercept_is_learned(offset_state):
    omega = list(offset_state.parameters.omega_mean)
    assert omega[-1] == pytest.approx(10.0, abs=0.05)
    assert omega[0] == pytest.approx(0.5, abs=0.05)


def test_prediction_with_all_features_known(linear_state):
    out = estimate(linear_state, [2.0, 4.0])
    assert float(out.mean) == pytest.approx(10.0, abs=0.05)
    # Only the observation noise (0.1) remains.
    assert float(out.std) == pytest.approx(0.1, abs=0.02)


def test_missing_feature_uncertainty_follows_its_spread(linear_state):
    # x₁ unknown: by the law of total variance the prediction averages over the
    # training distribution of x₁ ~ U(0, 10): mean 3·5 + 4, std ≈ 3·10/√12.
    out = estimate(linear_state, [None, 4.0])
    assert float(out.mean) == pytest.approx(19.0, abs=0.3)
    assert float(out.std) == pytest.approx(3 * 10 / math.sqrt(12), rel=0.05)


def test_unknown_feature_with_small_effect_adds_little_uncertainty(linear_state):
    # x₂ has a 3x smaller effect than x₁, so it contributes 3x less spread.
    only_x1 = estimate(linear_state, [2.0, None])
    assert float(only_x1.std) == pytest.approx(10 / math.sqrt(12), rel=0.05)
