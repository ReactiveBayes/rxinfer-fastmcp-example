import json
import os

from fastmcp import FastMCP
from juliacall import Main as jl
from pydantic import BaseModel, Field
from typing import Optional

mcp = FastMCP("RxInfer")

jl.include("rxinfer.jl")


def _resolve_data_path(file_path: str) -> str:
    """Resolve and validate a CSV path, restricting reads to a trusted data directory.

    The data directory defaults to the current working directory and can be overridden
    with the RXINFER_DATA_DIR environment variable. Absolute paths outside that directory
    and any ../ traversal are rejected.
    """
    data_dir = os.path.realpath(os.environ.get("RXINFER_DATA_DIR", os.getcwd()))
    p = os.path.realpath(file_path)
    if p != data_dir and not p.startswith(data_dir + os.sep):
        raise ValueError(f"file_path must be inside the trusted data directory: {data_dir}")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"No such file: {file_path}")
    return p


class AnalysisResult(BaseModel):
    status: str = Field(description="Status of the analysis ('success' or 'error')")
    num_houses_trained: Optional[int] = Field(
        default=None, description="Number of houses used for training"
    )
    num_features: Optional[int] = Field(
        default=None, description="Number of features used"
    )
    message: Optional[str] = Field(
        default=None, description="Error message if status is 'error'"
    )


class PriceEstimate(BaseModel):
    status: str = Field(description="Status of the prediction ('success' or 'error')")
    estimated_price: Optional[float] = Field(
        default=None, description="Estimated house price"
    )
    price_std: Optional[float] = Field(
        default=None,
        description="Standard deviation of the price estimate (uncertainty)",
    )
    specified_features: Optional[list[str]] = Field(
        default=None, description="Features that were provided"
    )
    missing_features: Optional[list[str]] = Field(
        default=None, description="Features that were missing and imputed"
    )
    message: Optional[str] = Field(
        default=None, description="Error message if status is 'error'"
    )


class DataDescription(BaseModel):
    status: str = Field(description="Status ('success' or 'error')")
    num_rows: Optional[int] = Field(
        default=None, description="Number of rows in the dataset"
    )
    columns: Optional[list[str]] = Field(default=None, description="Column names")
    column_details: Optional[dict] = Field(
        default=None, description="Per-column statistics and descriptions"
    )
    message: Optional[str] = Field(
        default=None, description="Error message if status is 'error'"
    )


@mcp.tool(timeout=120)
async def run_bayesian_analysis(file_path: str, iterations: int = 10) -> AnalysisResult:
    """
    Load a CSV file containing housing data and run Bayesian regression analysis.
    The CSV must have a 'price' column and can have any of these feature columns:
    bedrooms, bathrooms, living_area, floors, basement_area, yr_built, label.
    After training, the model can estimate prices for new houses via estimate_house_price.
    """
    try:
        resolved = _resolve_data_path(file_path)
    except (ValueError, FileNotFoundError, OSError) as e:
        return AnalysisResult(status="error", message=f"Invalid file path: {e}")
    result = jl.run_analysis(resolved, iterations=iterations)
    return AnalysisResult(**{k: result.get(k) for k in AnalysisResult.model_fields})


@mcp.tool(timeout=30)
async def describe_data() -> DataDescription:
    """
    Describe the structure of the currently loaded dataset. Shows column names,
    types, statistics (min/max/mean), and which columns the Bayesian model uses.
    Must call run_bayesian_analysis first to load data.
    """
    result = jl.get_data_description()
    result = json.loads(jl.JSON.json(result))
    return DataDescription(**{k: result.get(k) for k in DataDescription.model_fields})


@mcp.tool(timeout=60)
async def estimate_house_price(
    bedrooms: Optional[float] = None,
    bathrooms: Optional[float] = None,
    living_area: Optional[float] = None,
    floors: Optional[float] = None,
    basement_area: Optional[float] = None,
    yr_built: Optional[float] = None,
    label: Optional[float] = None,
) -> PriceEstimate:
    """
    Estimate the price of a house given its features. Not all features need to be specified -
    missing features will be imputed from the training data mean, and the model will report
    higher uncertainty for those. Must call run_bayesian_analysis first.

    Parameters:
    - bedrooms: Number of bedrooms
    - bathrooms: Number of bathrooms
    - living_area: Living area in square meters
    - floors: Number of floors
    - basement_area: Basement size in square meters
    - yr_built: Year the house was built
    - label: Quality label from 1.0 to 5.0
    """
    house_dict = {}
    if bedrooms is not None:
        house_dict["bedrooms"] = bedrooms
    if bathrooms is not None:
        house_dict["bathrooms"] = bathrooms
    if living_area is not None:
        house_dict["living_area"] = living_area
    if floors is not None:
        house_dict["floors"] = floors
    if basement_area is not None:
        house_dict["basement_area"] = basement_area
    if yr_built is not None:
        house_dict["yr_built"] = yr_built
    if label is not None:
        house_dict["label"] = label

    result = jl.predict_price(house_dict)
    result = json.loads(jl.JSON.json(result))
    return PriceEstimate(**{k: result.get(k) for k in PriceEstimate.model_fields})


if __name__ == "__main__":
    mcp.run()
