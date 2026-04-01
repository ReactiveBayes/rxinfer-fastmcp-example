using LoggingExtras, Logging, Dates, CSV, DataFrames, JSON

logger = TransformerLogger(ConsoleLogger()) do log
    return merge(log, (message = string("Julia [", Dates.now(UTC), "] ", log.message), ))
end

Logging.global_logger(logger)

@info "Including example.jl — this may take a while (loading packages, generating data, training)..."
include("example.jl")
@info "Finished including example.jl"

# --- Generate test CSV ---

if get(ENV, "SKIP_TEST_CSV", "false") != "true"
let
    test_file = "test_houses.csv"
    @info "Generating test CSV with 20,000 entries..."
    test_rng = StableRNG(123)
    test_houses = map(_ -> price_house(test_rng, random_house(test_rng)), 1:20_000)
    test_houses = filter(h -> h.price > 100_000, test_houses)
    df = DataFrame(
        bedrooms      = map(h -> h.house.bedrooms, test_houses),
        bathrooms     = map(h -> h.house.bathrooms, test_houses),
        living_area   = map(h -> h.house.living_area, test_houses),
        floors        = map(h -> h.house.floors, test_houses),
        basement_area = map(h -> h.house.basement_area, test_houses),
        yr_built      = map(h -> h.house.yr_built, test_houses),
        label         = map(h -> h.house.label, test_houses),
        price         = map(h -> h.price, test_houses),
    )
    CSV.write(test_file, df)
    @info "Test CSV available at: $test_file ($(nrow(df)) rows)"
end
else
    @info "Skipping test CSV generation (SKIP_TEST_CSV=true)"
end

# --- Global State ---

global_inference_state = nothing
global_dataframe = nothing

const HOUSE_FIELDS = [:bedrooms, :bathrooms, :living_area, :floors, :basement_area, :yr_built, :label]

const HOUSE_FIELD_DESCRIPTIONS = Dict(
    :bedrooms      => "Number of bedrooms (e.g. 1-4)",
    :bathrooms     => "Number of bathrooms (e.g. 1-3)",
    :living_area   => "Living area in square meters (e.g. 40-220)",
    :floors        => "Number of floors (e.g. 1-3)",
    :basement_area => "Basement size in square meters (e.g. 0-10)",
    :yr_built      => "Year the house was built (e.g. 1930-2025)",
    :label         => "Quality label from 1.0 to 5.0",
)

function run_analysis(file_path::String; iterations::Int=10)
    try
        @info "Starting Bayesian analysis on $file_path"
        global global_dataframe = CSV.read(file_path, DataFrame)

        priced_houses = PricedHouse[]
        for row in eachrow(global_dataframe)
            d = Dict{String, Any}()
            for field in HOUSE_FIELDS
                col = string(field)
                if col in names(global_dataframe) && !ismissing(row[col])
                    d[col] = Float64(row[col])
                end
            end
            house = convert(House, d)
            price = Float64(row["price"])
            push!(priced_houses, PricedHouse(house=house, price=price))
        end

        @info "Loaded $(length(priced_houses)) houses, running inference with $iterations iterations"

        arguments = InferenceArguments(
            number_of_iterations = iterations,
            compute_free_energy = true
        )

        _, state = learn_parameters(arguments, priced_houses)
        global global_inference_state = state

        @info "Bayesian analysis completed successfully"
        return Dict(
            "status" => "success",
            "num_houses_trained" => length(priced_houses),
            "num_features" => length(HOUSE_FIELDS),
        )
    catch e
        @error "Exception in analysis" error = (e, catch_backtrace())
        return Dict("status" => "error", "message" => "Analysis failed: $(e)")
    end
end

function predict_price(house_dict)
    try
        if global_inference_state === nothing
            return Dict("status" => "error", "message" => "No model trained yet. Run analysis first.")
        end

        @info "Predicting price for house: $house_dict"
        house = convert(House, Dict{String,Any}(house_dict))
        result = estimate_price(global_inference_state, house)

        specified = String[]
        unspecified = String[]
        for field in HOUSE_FIELDS
            if getfield(result.house, field) === missing
                push!(unspecified, string(field))
            else
                push!(specified, string(field))
            end
        end

        @info "Prediction complete: price=$(result.price), std=$(result.price_std)"
        return Dict(
            "status" => "success",
            "estimated_price" => round(result.price, digits=2),
            "price_std" => round(result.price_std, digits=2),
            "specified_features" => specified,
            "missing_features" => unspecified,
        )
    catch e
        @error "Exception in prediction" error = (e, catch_backtrace())
        return Dict("status" => "error", "message" => "Prediction failed: $(e)")
    end
end

function get_data_description()
    try
        if global_dataframe === nothing
            return Dict("status" => "error", "message" => "No data loaded yet. Run analysis first.")
        end

        df = global_dataframe
        col_info = Dict{String, Any}()
        for col in names(df)
            col_sym = Symbol(col)
            desc = get(HOUSE_FIELD_DESCRIPTIONS, col_sym, nothing)
            non_missing = count(!ismissing, df[!, col])
            entry = Dict{String, Any}(
                "non_missing_count" => non_missing,
                "total_count" => nrow(df),
                "known_field" => col_sym in HOUSE_FIELDS || col == "price",
                "description" => desc !== nothing ? desc : (col == "price" ? "House price (target variable)" : "Unknown field (will be ignored by the model)"),
            )
            if non_missing > 0
                vals = collect(skipmissing(df[!, col]))
                if eltype(vals) <: Number
                    entry["min"] = minimum(vals)
                    entry["max"] = maximum(vals)
                    entry["mean"] = round(mean(vals), digits=2)
                end
            end
            col_info[col] = entry
        end

        @info "Data description generated for $(nrow(df)) rows, $(ncol(df)) columns"
        return Dict(
            "status" => "success",
            "num_rows" => nrow(df),
            "columns" => names(df),
            "column_details" => col_info,
        )
    catch e
        @error "Exception in data description" error = (e, catch_backtrace())
        return Dict("status" => "error", "message" => "Description failed: $(e)")
    end
end
