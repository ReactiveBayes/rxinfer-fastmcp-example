from fastmcp import FastMCP
from juliacall import Main as jl
from pydantic import BaseModel, Field

mcp = FastMCP("RxInfer")

jl.include("rxinfer.jl")

class InferenceResult(BaseModel):
    a: int = Field(description="Alpha parameter of the posterior")
    b: int = Field(description="Beta parameter of the posterior")


class InferenceError(BaseModel):
    error: str = Field(description="Error in the inference")


@mcp.tool(timeout=10)
async def run_inference(n: int):
    """
    Run inference with RxInfer
    """
    result = jl.run_inference(n)

    if "error" in result:
        return InferenceError(error=result["error"])

    return InferenceResult(a=result["a"], b=result["b"])

if __name__ == "__main__":
    mcp.run()
