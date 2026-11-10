# RxInfer FastMCP Example

An MCP server that exposes Bayesian regression powered by [RxInfer.jl](https://rxinfer.ml). Point it at a CSV, and it learns a probabilistic model that predicts with honest uncertainty -- when information is missing, the model says "I'm less sure" instead of silently guessing.

It comes with two synthetic datasets:

- **Bike commute in Eindhoven** -- how long does it take to bike to the university, given distance, wind, rain, traffic lights, temperature and how busy the bike path is?
- **House prices** -- the original demo: price a house from its bedrooms, living area, year built, …

Both are generated from a *hidden formula*, so you can check whether the model recovers it.

Built with [FastMCP](https://gofastmcp.com/), [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/), and `uv`.

## Tools

| Tool | Description |
|------|-------------|
| `run_bayesian_analysis` | Load a CSV and train the model. The dataset is recognized from its target column (`commute_minutes` or `price`) |
| `estimate_bike_commute_time` | Predict a commute time in minutes (any input can be omitted) |
| `estimate_house_price` | Predict the price of a house (any feature can be omitted) |
| `describe_model` | Show the formula the model learned: every weight with its uncertainty, plus the noise level |
| `describe_data` | Inspect a loaded dataset -- columns, stats, and what the model recognizes |

Each dataset keeps its own trained model, so you can train both and switch between them during a demo.

| Dataset | Target column | Feature columns |
|---------|---------------|-----------------|
| bike | `commute_minutes` | `distance_km`, `wind_speed_ms`, `rain_mm_per_hour`, `traffic_lights`, `temperature_c`, `congestion` (0 empty – 10 bike jam), `bike_age_years` |
| houses | `price` | `bedrooms`, `bathrooms`, `living_area`, `floors`, `basement_area`, `yr_built`, `label` |

All feature columns must be present and complete in the training CSV.

## Installation

```bash
uv sync
```

Julia itself and all Julia dependencies are managed automatically by `juliapkg`: they are pinned in `juliapkg.json` and installed into an isolated Julia environment under `.venv/julia_env` (independent of any global Julia setup). The **first** run downloads and precompiles the full stack (RxInfer, …), which can take several minutes; subsequent runs use the compiled cache. If you hit issues, see the [juliapkg docs](https://github.com/JuliaPy/pyjuliapkg).

## Generating datasets

Datasets are generated with a small CLI (the generators live in `src/datasets.jl`):

```bash
uv run python generate_dataset.py bike      # -> bike_commutes.csv (20,000 rides)
uv run python generate_dataset.py houses    # -> test_houses.csv (~19,000 houses)
```

- **Deterministic**: generated with a fixed seed (`StableRNG`), so the same seed always gives exactly the same CSV. Reproducible demos.
- **Options**: `--n` (number of samples), `--seed`, `--out` (output path).
- **Gitignored on purpose**: regenerate them whenever you need them.

### The hidden formula

The point of the demo: *we* are the world, so we know the formula that generated the data. Reveal it with:

```bash
uv run python generate_dataset.py bike --show-formula
```

```
commute_minutes = 2.0                                     (unlock and park the bike)
                + 3.0·distance_km + 0.03·distance_km²     (~20 km/h, a bit slower on long rides)
                + 0.1·wind_speed_ms + 0.1·wind_speed_ms²  (wind hurts more and more)
                + 1.2·rain_mm_per_hour                    (careful riding in the rain)
                + 0.5·traffic_lights                      (~30 s per traffic light)
                + 0.01·(temperature_c - 15)²              (15 °C is most comfortable)
                + 0.3·congestion + 0.03·congestion²       (busy bike paths)
                + 0·bike_age_years                        (an old bike is just as fast)
                + Normal(0, 1.5)                          (everything else)
```

It also prints the same formula expanded into the model's terms (`x`, `x²`, `intercept`), so you can compare it line by line with what the `describe_model` tool reports after training. The formula never leaves the generator -- the MCP server (and the LLM) only see the data.

The house prices use random per-house coefficients, so there the model can only recover the average effects (`--show-formula` explains this too).

## Usage

Generate the datasets (see above), then start the server in the background before connecting any MCP client:

```bash
uv run fastmcp run main.py:mcp --transport sse --port 8001
```

The first startup is slow because Julia compiles packages. Subsequent runs use the compiled cache and start in a few seconds. Wait until you see the server listening message before proceeding.

## The model

The model lives in `src/model.jl` and knows nothing about bikes or houses. It is Bayesian linear regression on linear, squared and constant terms of every feature,

```
target ≈ w₁·x₁ + … + w_d·x_d + v₁·x₁² + … + v_d·x_d² + b  +  noise
```

with Gaussian priors on the weights and a Gamma prior on the noise precision, learned with variational message passing in RxInfer.

**Missing inputs.** When an input is unknown (say, tomorrow's wind), the model averages its prediction over the values of that input seen in the training data -- the *law of total variance*:

```
mean = E_x[ μ(x) ]
var  = E_x[ σ²(x) ] + Var_x[ μ(x) ]
       (model)        (because x is unknown)
```

An unknown input that matters (large weight, wide spread) widens the prediction a lot; an unknown input that doesn't matter (like the bike's age) barely changes it.

## Example conversation

Ask Claude to train on the bike commutes (`bike_commutes.csv`), then:

> Hey, how long would it take me to bike to the university today? The weather is perfect, the uni is approximately 5 km away, there are maybe 5 traffic lights and the path is clear.

→ about **20 minutes ± 1.5**. Then:

> Okay, actually I just checked the forecast: by the time I go outside it will start raining and the wind will pick up to 5 m/s. Does that change the estimate?

→ about **26 minutes ± 1.5**. Things to try next:

- Ask about *tomorrow* without knowing the weather: the estimate gets much wider (± 6 minutes), because wind and rain matter a lot.
- Ask whether the age of the bike matters, or let Claude call `describe_model` and compare its answer with `--show-formula`.
- Ask for a what-if sweep: "plot the commute time for wind speeds from 0 to 12 m/s".

The original house price demo works the same way. Train on `test_houses.csv`, then ask Claude to estimate a price for some house, leaving some features out:

<img width="883" height="1026" alt="Screenshot 2026-03-26 at 11 20 36" src="https://github.com/user-attachments/assets/c9fe90f9-f221-48ec-a964-f73b63225c6e" />

<img width="865" height="1201" alt="Screenshot 2026-03-26 at 11 20 22" src="https://github.com/user-attachments/assets/6cec6da5-bb9b-4f1b-8172-afdb1c092ff3" />

Then imagine you found out the previously unknown features and check the new result:

<img width="860" height="1000" alt="Screenshot 2026-03-26 at 11 01 23" src="https://github.com/user-attachments/assets/e65f14e4-a80f-472e-8f77-6de122da1f16" />

(These screenshots predate the bike dataset; tool outputs have been extended since, but the flow is the same.)

## Claude Desktop

To connect via SSE in Claude Desktop, add the following to your [Claude Desktop MCP configuration](https://modelcontextprotocol.io/quickstart/user):

```json
{
  "mcpServers": {
    "rxinfer": {
      "command": "npx",
      "args": [
        "mcp-remote@latest",
        "http://localhost:8001/sse"
      ]
    }
  }
}
```

Make sure the server is running before launching Claude Desktop.

## Claude Code

A `.mcp.json` is included that connects to the server on `localhost:8001`. Start the server first in a separate terminal:

```bash
uv run fastmcp run main.py:mcp --transport sse --port 8001
```

Then launch Claude Code in the project directory. The server will be detected automatically — approve it when prompted.

## Project layout

| Path | What's inside |
|------|---------------|
| `main.py` | The MCP tools (FastMCP) |
| `src/model.jl` | The Bayesian regression model (RxInfer), independent of any dataset |
| `src/datasets.jl` | Dataset definitions, generators and hidden formulas |
| `src/server.jl` | Glue between the tools and the model: one trained model per dataset |
| `generate_dataset.py` | Dataset generator CLI |
| `tests/` | Test suite |

To add a dataset, register a `DatasetSpec` in `src/datasets.jl`, add its name to `generate_dataset.py`, and add an `estimate_…` tool in `main.py`.

## Running tests

```bash
uv run pytest
```

The tests start Julia once and drive the server through FastMCP's in-memory client, just like an LLM would. They cover the model on small problems with known answers, the house demo (pinned with golden values in `tests/golden/`), the bike dataset (including recovery of the hidden formula), and the generator CLI.

After a *deliberate* change in behaviour, re-record the golden values and review the diff:

```bash
RECORD_GOLDEN=1 uv run pytest
```
