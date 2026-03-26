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

Julia dependencies are managed automatically by `juliapkg`. If you hit issues, see the [juliapkg docs](https://github.com/JuliaPy/pyjuliapkg).

## Usage

Start the server in the background before connecting any MCP client:

```bash
uv run fastmcp run main.py:mcp --transport sse --port 8001
```

The first startup is slow (~30s) because Julia compiles packages and generates a test CSV. Subsequent runs use the compiled cache and start much faster. Wait until you see the server listening message before proceeding.

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
