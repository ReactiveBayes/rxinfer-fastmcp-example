"""Shared fixtures for the test suite.

Julia is started once per test session (importing ``main`` loads RxInfer), so
the expensive part is paid only once. Tests talk to the server the same way an
LLM would: through the MCP tools, via FastMCP's in-memory ``Client``.

Golden values
-------------
Numbers that pin down current behaviour live in ``tests/golden/*.json``. To
(re-)record them after a *deliberate* behaviour change, run::

    RECORD_GOLDEN=1 uv run pytest

and review the resulting diff before committing.
"""

import json
import math
import os
from pathlib import Path

import pytest
from fastmcp import Client

from main import jl, mcp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
RECORDING = os.environ.get("RECORD_GOLDEN") == "1"


class Golden:
    """Compare observed values against a JSON file of recorded golden values."""

    def __init__(self, path: Path):
        self.path = path
        self.values = json.loads(path.read_text()) if path.exists() else {}
        self.dirty = False

    def check(self, key: str, observed, rtol: float = 1e-6):
        if RECORDING:
            self.values[key] = observed
            self.dirty = True
            return
        assert key in self.values, f"no golden value for {key!r}; run with RECORD_GOLDEN=1"
        expected = self.values[key]
        if isinstance(expected, float):
            assert math.isclose(observed, expected, rel_tol=rtol), (
                f"{key}: observed {observed!r} != golden {expected!r} (rtol={rtol})"
            )
        else:
            assert observed == expected, f"{key}: observed {observed!r} != golden {expected!r}"

    def get(self, key: str):
        assert key in self.values, f"no golden value for {key!r}; run with RECORD_GOLDEN=1"
        return self.values[key]

    def save(self):
        if self.dirty:
            self.path.write_text(json.dumps(self.values, indent=2, sort_keys=True) + "\n")


_goldens: dict[str, Golden] = {}


def golden_file(name: str) -> Golden:
    if name not in _goldens:
        _goldens[name] = Golden(GOLDEN_DIR / f"{name}.json")
    return _goldens[name]


def pytest_sessionfinish(session, exitstatus):
    for g in _goldens.values():
        g.save()


@pytest.fixture
def golden_houses() -> Golden:
    return golden_file("houses")


@pytest.fixture
def golden_tools() -> Golden:
    return golden_file("tools")


@pytest.fixture(scope="session")
def houses_csv() -> Path:
    """The house dataset used in the original demo (generated at server start)."""
    path = PROJECT_ROOT / "test_houses.csv"
    assert path.exists(), "test_houses.csv should be generated when the server starts"
    return path


@pytest.fixture(scope="session")
def trained_houses(houses_csv):
    """Train the house model once per session."""
    result = jl.run_analysis(str(houses_csv), iterations=10)
    assert result["status"] == "success"
    return result


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


def _save_state():
    return (
        jl.seval("global_inference_state"),
        jl.seval("global_dataframe"),
    )


def _restore_state(saved):
    state, df = saved
    jl.seval("(s, d) -> begin global global_inference_state = s; global global_dataframe = d; nothing end")(
        state, df
    )


@pytest.fixture
def isolated_state():
    """Run a test against whatever model state it creates, then put the previous state back."""
    saved = _save_state()
    yield
    _restore_state(saved)


@pytest.fixture
def untrained(isolated_state):
    """Pretend no model has been trained yet (restored after the test)."""
    _restore_state((None, None))
