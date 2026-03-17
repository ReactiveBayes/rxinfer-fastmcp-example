using RxInfer, Random, LoggingExtras, Logging, Dates

logger = TransformerLogger(ConsoleLogger()) do log 
    return merge(log, (message = string("Julia [", Dates.now(UTC), "] ", log.message), ))
end

Logging.global_logger(logger)

# GraphPPL.jl export `@model` macro for model specification
# It accepts a regular Julia function and builds a factor graph under the hood
@model function coin_model(y, a, b)
    # We endow θ parameter of our model with some prior
    θ ~ Beta(a, b)
    # We assume that outcome of each coin flip 
    # is governed by the Bernoulli distribution
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)
    end
end

function run_inference(n)
    try
        @info "Starting inference"
        p = 0.75 # Bias of a coin

        distribution = Bernoulli(p)
        dataset = rand(distribution, n)

        result = infer(
            model=coin_model(a=2.0, b=7.0),
            data=(y=dataset,)
        )
        @info "Inference finished"

        posterior = result.posteriors[:θ]

        return Dict("a" => posterior.α, "b" => posterior.β)
    catch e
        @error "Exception occured in the inference procedure" error = (e, catch_backtrace())
        return Dict("error" => "Exception occured during inference. $(e)")
    end
end


