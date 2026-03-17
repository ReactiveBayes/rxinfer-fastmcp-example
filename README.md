# RxInfer FastMCP example

This repository contains sample code to demonstrate how to call RxInfer 
from Python's code and return the result to an LLM with MCP protocol.

The project uses [FastMCP](https://gofastmcp.com/) to implement 
the MCP server. The python dependencies are managed with `uv` and the Julia-Python 
interaction is done with [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/#Installation).

## Installation

```bash
uv sync
```

## Usage

```bash
uv run python main.py
```

or 

```
uv run fastmcp run main.py:mcp --transport sse --port 8001
```

### Julia dependencies

The `juliacall` package uses `juliapkg` to manage Julia dependencies automatically.
In theory, you should be doing anything and all required dependencies should be 
installed automatically. In case of some Julia dependency problems refer to the 
documentation of `juliapkg` to see how to install all the required dependencies 
manually.

## MCP Client Config

Refer to your MCP client documentation to check how to add MCP servers there.
Typically it would imply some kind of `settings.json` and adding something like:
```json
{
  "mcpServers": {
    "rxinfer": {
      "command": "python",
      "args": ["main.py"]
    }
  }
}
```

