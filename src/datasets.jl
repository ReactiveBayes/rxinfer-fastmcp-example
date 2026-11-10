# Synthetic datasets for the demo.
#
# Each dataset is described by a `DatasetSpec`: which column is the target, which
# columns are features (with human/LLM-readable descriptions), and a generator
# that produces a reproducible `DataFrame` from a seed. This file deliberately does
# not load RxInfer, so generating data (see `generate_dataset.py`) is fast.

using CSV, DataFrames, Distributions, Random, StableRNGs, ConcreteStructs

"""
    DatasetSpec

Everything the server and the generator need to know about one dataset.
"""
Base.@kwdef struct DatasetSpec
    name               :: String
    target             :: String
    target_description :: String
    features           :: Vector{String}
    descriptions       :: Dict{String, String}
    default_n          :: Int
    default_seed       :: Int
    default_file       :: String
    generate           :: Function  # (rng, n) -> DataFrame with `features` + `target` columns
    # The hidden formula, for `generate_dataset.py --show-formula`. Never exposed to the LLM.
    formula            :: String = ""
    # The hidden formula in the model's basis (`feature_function_names` order), if the
    # model can represent it exactly. `nothing` if there is no exact formula to recover.
    true_coefficients  :: Union{Nothing, Dict{String, Float64}} = nothing
    noise_std          :: Union{Nothing, Float64} = nothing
end

"""
    feature_function_names(features)

Human-readable names of the model's basis functions `[x₁, …, x_d, x₁², …, x_d², 1]`,
in the same order as `create_feature_functions` in `model.jl`.
"""
function feature_function_names(features::AbstractVector{<:AbstractString})
    return vcat(collect(features), [f * "^2" for f in features], ["intercept"])
end

const DATASETS = Dict{String, DatasetSpec}()

register_dataset!(spec::DatasetSpec) = (DATASETS[spec.name] = spec)

function dataset_spec(name::AbstractString)
    haskey(DATASETS, name) || error("Unknown dataset '$name'. Available: $(join(sort(collect(keys(DATASETS))), ", "))")
    return DATASETS[name]
end

"""
    generate_dataset(name; n, seed)

Generate the dataset `name` with `n` samples using a `StableRNG(seed)`, so the same
seed always produces exactly the same data.
"""
function generate_dataset(name::AbstractString; n = nothing, seed = nothing)
    spec = dataset_spec(name)
    rng = StableRNG(something(seed, spec.default_seed))
    return spec.generate(rng, something(n, spec.default_n))
end

"""
    write_dataset(name, path; n, seed)

Generate the dataset `name` and write it to `path` as CSV. Returns the number of rows.
"""
function write_dataset(name::AbstractString, path::AbstractString; n = nothing, seed = nothing)
    df = generate_dataset(name; n = n, seed = seed)
    CSV.write(path, df)
    return nrow(df)
end

# ==============================================================================
# Houses
# ==============================================================================

"""
    House

A house described by its key features: number of bedrooms and bathrooms, living area in
square meters, number of floors, basement size, year built, and a quality label (1.0–5.0).
"""
Base.@kwdef @concrete struct House
    bedrooms
    bathrooms
    living_area
    floors
    basement_area
    yr_built
    label
end

"""
    random_house(rng::AbstractRNG)

Generate a random house with realistic-ish feature ranges. Single-story homes are more common
(weighted sampling), and quality labels range from 1.0 to 5.0 in 0.1 increments.
"""
function random_house(rng::AbstractRNG)
    return House(
        bedrooms = rand(rng, 1:4),
        bathrooms = rand(rng, 1:3),
        living_area = rand(rng, 40:220),
        floors = rand(rng, [1, 1, 1, 1, 2, 2, 3]),
        basement_area = rand(rng, 0:10),
        yr_built = rand(rng, 1930:2025),
        label = rand(rng, 1.0:0.1:5.0)
    )
end

"""
    price_house(rng, house)

The hidden pricing formula. Each coefficient is itself random per house, so the noise is
multiplicative and the formula is not exactly recoverable by the model.
"""
function price_house(rng::AbstractRNG, house::House)
    return (
        (house.yr_built - 1920) * rand(rng, Normal(1_000, 100)) +
        (house.label)^2 * rand(rng, Normal(3_500, 1_000)) +
        house.bedrooms * rand(rng, Normal(7_000, 1_000)) +
        house.bathrooms * rand(rng, Normal(4_000, 1_000)) +
        (house.living_area - 80) * rand(rng, Normal(4_000, 500)) +
        house.floors * rand(rng, Normal(25_000, 3_000)) +
        house.basement_area * rand(rng, Normal(2500, 100))
    ) + rand(rng, 40_000:1_000:80_000)
end

function generate_houses(rng::AbstractRNG, n::Int)
    houses = map(1:n) do _
        house = random_house(rng)
        (house = house, price = price_house(rng, house))
    end
    # only consider houses that are above a certain threshold
    houses = filter(h -> h.price > 100_000, houses)
    return DataFrame(
        bedrooms      = map(h -> h.house.bedrooms, houses),
        bathrooms     = map(h -> h.house.bathrooms, houses),
        living_area   = map(h -> h.house.living_area, houses),
        floors        = map(h -> h.house.floors, houses),
        basement_area = map(h -> h.house.basement_area, houses),
        yr_built      = map(h -> h.house.yr_built, houses),
        label         = map(h -> h.house.label, houses),
        price         = map(h -> h.price, houses),
    )
end

register_dataset!(DatasetSpec(
    name               = "houses",
    target             = "price",
    target_description = "House price (target variable)",
    features           = ["bedrooms", "bathrooms", "living_area", "floors", "basement_area", "yr_built", "label"],
    descriptions       = Dict(
        "bedrooms"      => "Number of bedrooms (e.g. 1-4)",
        "bathrooms"     => "Number of bathrooms (e.g. 1-3)",
        "living_area"   => "Living area in square meters (e.g. 40-220)",
        "floors"        => "Number of floors (e.g. 1-3)",
        "basement_area" => "Basement size in square meters (e.g. 0-10)",
        "yr_built"      => "Year the house was built (e.g. 1930-2025)",
        "label"         => "Quality label from 1.0 to 5.0",
    ),
    default_n          = 20_000,
    default_seed       = 123,
    default_file       = "test_houses.csv",
    generate           = generate_houses,
    formula            = """
price = (yr_built - 1920)   · N(1000, 100)
      + label²              · N(3500, 1000)
      + bedrooms            · N(7000, 1000)
      + bathrooms           · N(4000, 1000)
      + (living_area - 80)  · N(4000, 500)
      + floors              · N(25000, 3000)
      + basement_area       · N(2500, 100)
      + uniform{40000, 41000, …, 80000}
(only houses with price > 100000 are kept)

Every coefficient is drawn anew for each house (multiplicative noise), so there is
no exact formula for the model to recover - only the average effects.""",
))

# ==============================================================================
# Bike commute in Eindhoven
# ==============================================================================
#
# How long does it take to bike to the university? The hidden formula below is
# *additive* on purpose: real physics is multiplicative (time = distance / speed,
# and wind slows you down more on a longer ride), but an additive formula with
# linear and squared terms is exactly what our model can represent, so we can
# check whether it recovers the formula.

const BIKE_FEATURES = [
    "distance_km", "wind_speed_ms", "rain_mm_per_hour", "traffic_lights",
    "temperature_c", "congestion", "bike_age_years",
]

const BIKE_NOISE_STD = 1.5

"""
    bike_commute_minutes(; distance_km, wind_speed_ms, rain_mm_per_hour, traffic_lights,
                           temperature_c, congestion, bike_age_years)

The hidden formula: noise-free door-to-door commute time in minutes.
"""
function bike_commute_minutes(; distance_km, wind_speed_ms, rain_mm_per_hour, traffic_lights,
                                temperature_c, congestion, bike_age_years)
    return 2.0 +                                              # unlock and park the bike
        3.0 * distance_km + 0.03 * distance_km^2 +            # ~20 km/h, a bit slower on long rides
        0.1 * wind_speed_ms + 0.1 * wind_speed_ms^2 +         # wind hurts more and more
        1.2 * rain_mm_per_hour +                              # careful riding in the rain
        0.5 * traffic_lights +                                # ~30 s per traffic light
        0.01 * (temperature_c - 15)^2 +                       # 15 °C is most comfortable
        0.3 * congestion + 0.03 * congestion^2 +              # busy bike paths
        0.0 * bike_age_years                                  # an old bike is just as fast
end

const BIKE_FORMULA = """
commute_minutes = 2.0                                     (unlock and park the bike)
                + 3.0·distance_km + 0.03·distance_km²     (~20 km/h, a bit slower on long rides)
                + 0.1·wind_speed_ms + 0.1·wind_speed_ms²  (wind hurts more and more)
                + 1.2·rain_mm_per_hour                    (careful riding in the rain)
                + 0.5·traffic_lights                      (~30 s per traffic light)
                + 0.01·(temperature_c - 15)²              (15 °C is most comfortable)
                + 0.3·congestion + 0.03·congestion²       (busy bike paths)
                + 0·bike_age_years                        (an old bike is just as fast)
                + Normal(0, 1.5)                          (everything else)"""

# The same formula expanded into the model's basis [x, x², 1]:
# 0.01·(T - 15)² = 0.01·T² - 0.3·T + 2.25, so the intercept is 2.0 + 2.25.
const BIKE_TRUE_COEFFICIENTS = Dict{String, Float64}(
    "distance_km"        => 3.0,  "distance_km^2"        => 0.03,
    "wind_speed_ms"      => 0.1,  "wind_speed_ms^2"      => 0.1,
    "rain_mm_per_hour"   => 1.2,  "rain_mm_per_hour^2"   => 0.0,
    "traffic_lights"     => 0.5,  "traffic_lights^2"     => 0.0,
    "temperature_c"      => -0.3, "temperature_c^2"      => 0.01,
    "congestion"         => 0.3,  "congestion^2"         => 0.03,
    "bike_age_years"     => 0.0,  "bike_age_years^2"     => 0.0,
    "intercept"          => 4.25,
)

function random_bike_ride(rng::AbstractRNG)
    return (
        distance_km      = round(0.5 + 14.5 * rand(rng), digits = 1),
        wind_speed_ms    = round(12 * rand(rng), digits = 1),
        rain_mm_per_hour = rand(rng) < 0.6 ? 0.0 : round(0.1 + 9.9 * rand(rng), digits = 1),
        traffic_lights   = rand(rng, 0:15),
        temperature_c    = round(-5 + 35 * rand(rng), digits = 1),
        congestion       = rand(rng, 0:10),
        bike_age_years   = rand(rng, 0:20),
    )
end

function generate_bike_commutes(rng::AbstractRNG, n::Int)
    rides = map(1:n) do _
        ride = random_bike_ride(rng)
        minutes = bike_commute_minutes(; ride...) + BIKE_NOISE_STD * randn(rng)
        (; ride..., commute_minutes = round(minutes, digits = 2))
    end
    return DataFrame(rides)
end

register_dataset!(DatasetSpec(
    name               = "bike",
    target             = "commute_minutes",
    target_description = "Door-to-door bike commute time in minutes (target variable)",
    features           = BIKE_FEATURES,
    descriptions       = Dict(
        "distance_km"      => "Length of the ride in kilometers (e.g. 0.5-15)",
        "wind_speed_ms"    => "Wind speed in m/s (0 calm, ~3 light breeze, ~5 moderate, ~8 strong, 12 very strong)",
        "rain_mm_per_hour" => "Rain intensity in mm/hour (0 dry, ~1 light rain, ~4 moderate, 8+ heavy)",
        "traffic_lights"   => "Number of traffic lights on the route (e.g. 0-15)",
        "temperature_c"    => "Air temperature in degrees Celsius (e.g. -5 to 30)",
        "congestion"       => "How busy the bike path is, from 0 (empty) to 10 (rush-hour bike jam)",
        "bike_age_years"   => "Age of the bike in years (e.g. 0-20)",
    ),
    default_n          = 20_000,
    default_seed       = 123,
    default_file       = "bike_commutes.csv",
    generate           = generate_bike_commutes,
    formula            = BIKE_FORMULA,
    true_coefficients  = BIKE_TRUE_COEFFICIENTS,
    noise_std          = BIKE_NOISE_STD,
))
