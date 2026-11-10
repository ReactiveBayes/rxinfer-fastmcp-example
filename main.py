import json
from pathlib import Path
from typing import Literal, Optional

from fastmcp import FastMCP
from juliacall import Main as jl
from pydantic import BaseModel, Field

mcp = FastMCP("RxInfer")

jl.include(str(Path(__file__).resolve().parent / "src" / "server.jl"))

Dataset = Literal["houses", "bike"]


def call_julia(fn, *args, **kwargs) -> dict:
    """Call a Julia function returning a Dict and convert the result to plain Python."""
    return json.loads(jl.JSON.json(fn(*args, **kwargs)))


class AnalysisResult(BaseModel):
    status: str = Field(description="Status of the analysis ('success' or 'error')")
    dataset: Optional[str] = Field(
        default=None, description="Which dataset was recognized and trained ('houses' or 'bike')"
    )
    num_rows_trained: Optional[int] = Field(
        default=None, description="Number of rows used for training"
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


class CommuteEstimate(BaseModel):
    status: str = Field(description="Status of the prediction ('success' or 'error')")
    estimated_minutes: Optional[float] = Field(
        default=None, description="Estimated door-to-door commute time in minutes"
    )
    minutes_std: Optional[float] = Field(
        default=None,
        description="Standard deviation of the estimate in minutes (uncertainty)",
    )
    specified_features: Optional[list[str]] = Field(
        default=None, description="Features that were provided"
    )
    missing_features: Optional[list[str]] = Field(
        default=None, description="Features that were unknown and averaged over"
    )
    message: Optional[str] = Field(
        default=None, description="Error message if status is 'error'"
    )


class DataDescription(BaseModel):
    status: str = Field(description="Status ('success' or 'error')")
    dataset: Optional[str] = Field(default=None, description="Which dataset is described")
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
    Load a CSV file and train a Bayesian regression model on it. The dataset is
    recognized from its target column:

    - 'price' -> house prices. Feature columns: bedrooms, bathrooms, living_area,
      floors, basement_area, yr_built, label. Then use estimate_house_price.
    - 'commute_minutes' -> bike commute times in Eindhoven. Feature columns:
      distance_km, wind_speed_ms, rain_mm_per_hour, traffic_lights, temperature_c,
      congestion, bike_age_years. Then use estimate_bike_commute_time.

    All feature columns must be present and complete. Each dataset keeps its own
    trained model, so you can train both and switch between them.
    """
    result = call_julia(jl.run_analysis, file_path, iterations=iterations)
    return AnalysisResult(**{k: result.get(k) for k in AnalysisResult.model_fields})


@mcp.tool(timeout=30)
async def describe_data(dataset: Optional[Dataset] = None) -> DataDescription:
    """
    Describe the structure of a loaded dataset. Shows column names, types,
    statistics (min/max/mean), and which columns the Bayesian model uses.
    `dataset` is 'houses' or 'bike'; defaults to the most recently trained one.
    Must call run_bayesian_analysis first to load data.
    """
    result = call_julia(jl.get_data_description, dataset)
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

    result = call_julia(jl.predict, "houses", house_dict)
    result["estimated_price"] = result.get("estimate")
    result["price_std"] = result.get("std")
    return PriceEstimate(**{k: result.get(k) for k in PriceEstimate.model_fields})


@mcp.tool(timeout=60)
async def estimate_bike_commute_time(
    distance_km: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    rain_mm_per_hour: Optional[float] = None,
    traffic_lights: Optional[float] = None,
    temperature_c: Optional[float] = None,
    congestion: Optional[float] = None,
    bike_age_years: Optional[float] = None,
) -> CommuteEstimate:
    """
    Estimate how long a bike commute in Eindhoven takes, in minutes, with uncertainty.
    Must call run_bayesian_analysis on the bike commute dataset first.
    Every input is optional, so choose which to pass based on your use case:

    1) Estimating a real ride with incomplete information. Provide only what you
       actually know and OMIT anything whose value is unknown (leave it unset / null) -
       do NOT fill an unknown input with a single guessed value as if it were known.
       The Bayesian model averages over the values it has seen for every omitted input
       and reports a larger minutes_std when that input matters, which is the correct,
       honest result. Example: if you don't know tomorrow's wind, omit wind_speed_ms
       rather than picking some plausible value.

    2) Exploring or comparing scenarios (what-if analysis). It is perfectly fine - and
       intended - to call this tool multiple times with different explicit values, e.g.
       comparing a dry and a rainy ride, or sweeping wind_speed_ms from 0 to 12 to see
       how much headwind costs. Here the varied values are deliberate inputs you are
       testing, not guesses standing in for missing knowledge.

    When the user describes conditions in words, translating them is fine: "perfect
    weather" means no rain and little wind, "the path is clear" means congestion 0.

    Parameters (all optional):
    - distance_km: Length of the ride in km (e.g. 0.5-15) - omit if unknown
    - wind_speed_ms: Wind speed in m/s (0 calm, ~3 light breeze, ~5 moderate, ~8 strong,
      12 very strong) - omit if unknown
    - rain_mm_per_hour: Rain intensity in mm/hour (0 dry, ~1 light rain, ~4 moderate,
      8+ heavy) - omit if unknown
    - traffic_lights: Number of traffic lights on the route (e.g. 0-15) - omit if unknown
    - temperature_c: Air temperature in degrees Celsius (e.g. -5 to 30) - omit if unknown
    - congestion: How busy the bike path is, 0 (empty) to 10 (rush-hour bike jam) -
      omit if unknown
    - bike_age_years: Age of the bike in years (e.g. 0-20) - omit if unknown
    """
    inputs = {
        "distance_km": distance_km,
        "wind_speed_ms": wind_speed_ms,
        "rain_mm_per_hour": rain_mm_per_hour,
        "traffic_lights": traffic_lights,
        "temperature_c": temperature_c,
        "congestion": congestion,
        "bike_age_years": bike_age_years,
    }
    inputs = {k: v for k, v in inputs.items() if v is not None}

    result = call_julia(jl.predict, "bike", inputs)
    result["estimated_minutes"] = result.get("estimate")
    result["minutes_std"] = result.get("std")
    return CommuteEstimate(**{k: result.get(k) for k in CommuteEstimate.model_fields})


if __name__ == "__main__":
    mcp.run()
