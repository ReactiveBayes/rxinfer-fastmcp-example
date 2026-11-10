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
))
