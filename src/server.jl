# Glue between the MCP server (main.py) and the Julia model.
#
# Keeps one trained model per dataset, so you can train on several datasets and
# switch between them during a demo without retraining.

using LoggingExtras, Logging, Dates, CSV, DataFrames, JSON

logger = TransformerLogger(ConsoleLogger()) do log
    return merge(log, (message = string("Julia [", Dates.now(UTC), "] ", log.message), ))
end

Logging.global_logger(logger)

@info "Loading model and datasets (first start compiles packages, this may take a while)..."
include(joinpath(@__DIR__, "model.jl"))
include(joinpath(@__DIR__, "datasets.jl"))
@info "Model and datasets loaded"

# --- Global State ---

const STATES       = Dict{String, InferenceState}()
const DATAFRAMES   = Dict{String, DataFrame}()
const LAST_DATASET = Ref{Union{Nothing, String}}(nothing)

function reset_state!()
    empty!(STATES)
    empty!(DATAFRAMES)
    LAST_DATASET[] = nothing
    return nothing
end

"""
    detect_dataset(df)

Find the dataset whose target column is present in `df`.
"""
function detect_dataset(df::DataFrame)
    for name in sort(collect(keys(DATASETS)))
        DATASETS[name].target in names(df) && return name
    end
    return nothing
end

function run_analysis(file_path::String; iterations::Int=10)
    try
        @info "Starting Bayesian analysis on $file_path"
        df = CSV.read(file_path, DataFrame)

        dataset = detect_dataset(df)
        if dataset === nothing
            targets = join(["'$(s.target)' ($(s.name))" for s in values(DATASETS)], ", ")
            return Dict("status" => "error", "message" => "Could not recognize the dataset: the CSV needs one of these target columns: $targets.")
        end
        spec = DATASETS[dataset]

        # The model needs complete training data. Missing values would make
        # inference blow the stack, which crashes the whole process under juliacall.
        for col in vcat(spec.features, spec.target)
            if !(col in names(df))
                return Dict("status" => "error", "message" => "Column '$col' is missing from the CSV. Required columns: $(join(spec.features, ", ")), $(spec.target).")
            end
            if any(ismissing, df[!, col])
                return Dict("status" => "error", "message" => "Column '$col' has empty cells. The model needs complete training data.")
            end
        end

        x = [Union{Float64, Missing}[Float64(row[f]) for f in spec.features] for row in eachrow(df)]
        y = Float64.(df[!, spec.target])

        @info "Loaded $(length(y)) rows of '$dataset', running inference with $iterations iterations"

        arguments = InferenceArguments(
            number_of_iterations = iterations,
            compute_free_energy = true
        )

        _, state = learn_parameters(arguments, x, y)
        STATES[dataset] = state
        DATAFRAMES[dataset] = df
        LAST_DATASET[] = dataset

        @info "Bayesian analysis completed successfully"
        return Dict(
            "status" => "success",
            "dataset" => dataset,
            "num_rows_trained" => length(y),
            "num_features" => length(spec.features),
        )
    catch e
        @error "Exception in analysis" error = (e, catch_backtrace())
        return Dict("status" => "error", "message" => "Analysis failed: $(e)")
    end
end

"""
    predict(dataset, inputs)

Predict the target of `dataset` for `inputs`, a dictionary from feature name to value.
Features not present in `inputs` are treated as unknown.
"""
function predict(dataset::String, inputs)
    try
        spec = dataset_spec(dataset)
        if !haskey(STATES, dataset)
            return Dict("status" => "error", "message" => "No model trained yet for the '$dataset' dataset. Run analysis first.")
        end

        inputs = Dict{String, Any}(inputs)
        @info "Predicting '$(spec.target)' for $dataset: $inputs"
        x = Union{Float64, Missing}[haskey(inputs, f) ? Float64(inputs[f]) : missing for f in spec.features]
        result = estimate(STATES[dataset], x)

        specified = [f for f in spec.features if haskey(inputs, f)]
        unspecified = [f for f in spec.features if !haskey(inputs, f)]

        @info "Prediction complete: mean=$(result.mean), std=$(result.std)"
        return Dict(
            "status" => "success",
            "dataset" => dataset,
            "estimate" => round(result.mean, digits=2),
            "std" => round(result.std, digits=2),
            "specified_features" => specified,
            "missing_features" => unspecified,
        )
    catch e
        @error "Exception in prediction" error = (e, catch_backtrace())
        return Dict("status" => "error", "message" => "Prediction failed: $(e)")
    end
end

function get_data_description(dataset = nothing)
    try
        dataset = something(dataset, LAST_DATASET[], Some(nothing))
        if dataset === nothing || !haskey(DATAFRAMES, dataset)
            return Dict("status" => "error", "message" => "No data loaded yet. Run analysis first.")
        end

        spec = DATASETS[dataset]
        df = DATAFRAMES[dataset]
        col_info = Dict{String, Any}()
        for col in names(df)
            desc = col == spec.target ? spec.target_description :
                get(spec.descriptions, col, "Unknown field (will be ignored by the model)")
            non_missing = count(!ismissing, df[!, col])
            entry = Dict{String, Any}(
                "non_missing_count" => non_missing,
                "total_count" => nrow(df),
                "known_field" => col in spec.features || col == spec.target,
                "description" => desc,
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
            "dataset" => dataset,
            "num_rows" => nrow(df),
            "columns" => names(df),
            "column_details" => col_info,
        )
    catch e
        @error "Exception in data description" error = (e, catch_backtrace())
        return Dict("status" => "error", "message" => "Description failed: $(e)")
    end
end
