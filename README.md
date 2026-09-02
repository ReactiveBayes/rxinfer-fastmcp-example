# RxInfer FastMCP Example

An MCP server that exposes Bayesian house price estimation powered by [RxInfer.jl](https://rxinfer.ml). Point it at a CSV of housing data, and it learns a probabilistic pricing model that predicts prices with honest uncertainty -- when information is missing, the model says "I'm less sure" instead of silently guessing.

Built with [FastMCP](https://gofastmcp.com/), [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/), and `uv`.

## Tools

| Tool | Description |
|------|-------------|
| `run_bayesian_analysis` | Load a CSV file and train the Bayesian regression model |
| `describe_data` | Inspect the loaded dataset -- columns, stats, and what the model recognizes |
| `estimate_house_price` | Predict price for a house (any features can be omitted) |

The CSV should have a `price` column and any of: `bedrooms`, `bathrooms`, `living_area`, `floors`, `basement_area`, `yr_built`, `label`.

## Installation

```bash
uv sync
```

Julia itself and all Julia dependencies are managed automatically by `juliapkg`: they are pinned in `juliapkg.json` and installed into an isolated Julia environment under `.venv/julia_env` (independent of any global Julia setup). The **first** server start downloads and precompiles the full stack (RxInfer, Plots, …), which can take several minutes; subsequent starts use the compiled cache. If you hit issues, see the [juliapkg docs](https://github.com/JuliaPy/pyjuliapkg).

## Usage

Start the server in the background before connecting any MCP client:

```bash
uv run fastmcp run main.py:mcp --transport sse --port 8001
```

The first startup is slow (~30s) because Julia compiles packages and generates a test CSV. Subsequent runs use the compiled cache and start much faster. Wait until you see the server listening message before proceeding.

## Test dataset

Starting the server writes a synthetic dataset to `test_houses.csv` in the project root (~20,000 houses). This is the dataset used in the demo/presentation — there's no need to keep it around, since it's regenerated on demand.

- **Deterministic**: it's generated with a fixed seed (`StableRNG(123)`), so every regeneration produces the exact same dataset. Reproducible demos.
- **Gitignored on purpose**: to get it back, just (re)start the server. Delete the file first if you want a guaranteed-fresh copy.
- **Columns**: `bedrooms`, `bathrooms`, `living_area`, `floors`, `basement_area`, `yr_built`, `label`, `price`.

To use it in the demo, point `run_bayesian_analysis` at `test_houses.csv`, or drag-and-drop the file into Claude Desktop.

The generation logic lives in `rxinfer.jl` and reuses the `random_house` / `price_house` functions from `example.jl`. To skip it on startup (faster boot when you already have a dataset), set `SKIP_TEST_CSV=true`:

```bash
SKIP_TEST_CSV=true uv run fastmcp run main.py:mcp --transport sse --port 8001
```

## Pluto Notebook

The Bayesian model is developed in `example.jl`, a [Pluto.jl](https://plutojl.org/) reactive notebook. To explore or modify the model interactively, install Pluto following the [instructions on the website](https://plutojl.org/), then open the notebook:

```julia
using Pluto; Pluto.run()
```

This opens a browser-based environment where you can open `example.jl`, tweak the model, and see results update in real time.

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

## Example conversation

Ask Claude to run Bayesian analysis on the test dataset (`test_houses.csv`, created automatically when starting the server — see [Test dataset](#test-dataset))

<img width="883" height="1026" alt="Screenshot 2026-03-26 at 11 20 36" src="https://github.com/user-attachments/assets/c9fe90f9-f221-48ec-a964-f73b63225c6e" />

Prompt it to estimate a price for some house, some features might be missing

<img width="865" height="1201" alt="Screenshot 2026-03-26 at 11 20 22" src="https://github.com/user-attachments/assets/6cec6da5-bb9b-4f1b-8172-afdb1c092ff3" />

Think of the situation where you actually got the missing data about the previously unknown features and check the new result

<img width="860" height="1000" alt="Screenshot 2026-03-26 at 11 01 23" src="https://github.com/user-attachments/assets/e65f14e4-a80f-472e-8f77-6de122da1f16" />



