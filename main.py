import json
from pathlib import Path

from fastmcp import FastMCP
from juliacall import Main as jl
from pydantic import BaseModel, Field
from typing import Optional

mcp = FastMCP("RxInfer")

jl.include(str(Path(__file__).resolve().parent / "src" / "server.jl"))


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
    result = json.loads(jl.JSON.json(jl.run_analysis(file_path, iterations=iterations)))
    result["num_houses_trained"] = result.get("num_rows_trained")
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
    Estimate the price of a house given its features. Must call run_bayesian_analysis first.
    Every feature is optional, so choose which to pass based on your use case:

    1) Pricing a real house with incomplete information. Provide only the features you
       actually know and OMIT any feature whose value is unknown (leave it unset / null) -
       do NOT fill an unknown feature with a single guessed value as if it were known. The
       Bayesian model imputes each omitted feature from the training-data mean and reports
       higher uncertainty (a larger price_std) for it, which is the correct, honest result.
       Substituting a fabricated value instead produces a falsely confident, misleading
       estimate. Example: if you don't know when the house was built, omit yr_built rather
       than picking some plausible year.

    2) Exploring or comparing scenarios (what-if analysis). It is perfectly fine - and
       intended - to call this tool multiple times with different explicit values for a
       feature to see how the estimate changes: e.g. sweeping yr_built across a range to plot
       price vs. year, or comparing a 2-bedroom vs. a 4-bedroom version of the same house.
       Here the varied values are deliberate inputs you are testing, not guesses standing in
       for missing knowledge.

    Rule of thumb: omit a feature when its value is genuinely unknown; supply a value only
    when you know it, or when you are deliberately testing that specific value.

    Parameters (all optional):
    - bedrooms: Number of bedrooms - omit if unknown
    - bathrooms: Number of bathrooms - omit if unknown
    - living_area: Living area in square meters - omit if unknown
    - floors: Number of floors - omit if unknown
    - basement_area: Basement size in square meters - omit if unknown
    - yr_built: Year the house was built - omit if unknown
    - label: Quality label from 1.0 to 5.0 - omit if unknown
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

    result = json.loads(jl.JSON.json(jl.predict("houses", house_dict)))
    result["estimated_price"] = result.get("estimate")
    result["price_std"] = result.get("std")
    return PriceEstimate(**{k: result.get(k) for k in PriceEstimate.model_fields})


if __name__ == "__main__":
    mcp.run()
