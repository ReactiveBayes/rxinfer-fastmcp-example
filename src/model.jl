# Bayesian regression with linear + quadratic basis functions, learned with RxInfer.jl.
#
# Nothing in this file knows about houses or bikes: the model works on plain
# feature vectors `x::Vector{Union{Float64, Missing}}` and a target `y::Float64`.
# Dataset-specific code (column names, generators) lives in `datasets.jl`.

using RxInfer, LinearAlgebra, ConcreteStructs

# --- Feature engineering -----------------------------------------------------
#
# A linear model can only learn straight-line relationships. By adding
# *quadratic* basis functions alongside the linear ones, we give the model the
# vocabulary to express curved, non-linear effects (e.g. sweet spots).

"""
    linear_functions(x_dim)

Create a vector of functions that extract each individual feature from the input vector `x`.
"""
function linear_functions(x_dim)
    return [(x) -> getindex(x, i) for i in 1:x_dim]
end

"""
    quadratic_functions(x_dim)

Create a vector of functions that compute the *square* of each feature from the input vector `x`.
"""
function quadratic_functions(x_dim)
    return [(x) -> getindex(x, i)^2 for i in 1:x_dim]
end

"""
    create_feature_functions(x_dim)

Combine linear and quadratic basis functions plus a constant (intercept) into a single
feature map. For `x_dim` input features, this produces `2 * x_dim + 1` basis functions:
`[x₁, …, x_d, x₁², …, x_d², 1]`.

The intercept matters: a sweet spot like `-(x - 15)²` expands to `-x² + 30x - 225`,
and that constant `-225` can only be expressed by an intercept term.
"""
function create_feature_functions(x_dim)
    return vcat(linear_functions(x_dim), quadratic_functions(x_dim), [(x) -> 1.0])
end

# --- The probabilistic model -------------------------------------------------
#
# 1. Training model: observes all the data and learns the weight vector `ω`
#    and noise precision `s`. Each target is a noisy linear combination of the
#    basis-expanded features.
# 2. Predictive model: takes the learned posteriors and a *single* new input,
#    and produces a predictive distribution for its target.
#
# Both use variational message passing under mean-field constraints.

@model function feature_regression_unknown_noise(ϕs, x, y, priors)
    ω ~ priors[:ω]
    s ~ priors[:s]

    if length(x) != length(y)
        throw(DimensionMismatch("x and y must have the same length"))
    end

    M = length(x)

    for i in 1:M
        ϕx = [ϕ(x[i]) for ϕ in ϕs]
        y[i] ~ Normal(mean = dot(ϕx, ω), precision = s)
    end
end

@model function feature_regression_predictive(ϕs, x, y, priors)
    ω ~ priors[:ω]
    s ~ priors[:s]
    ϕx = [ϕ(x) for ϕ in ϕs]
    μ := dot(ϕx, ω)
    y ~ Normal(mean = μ, precision = s)
end

@constraints function feature_regressions_predictive_constraints()
    q(μ, s, y) = q(μ, y)q(s)
end

# --- Inference state -----------------------------------------------------------

"""
    InferenceArguments

Configuration for the inference engine: how many message-passing iterations to run
and whether to track the variational free energy (a convergence diagnostic).
"""
Base.@kwdef @concrete struct InferenceArguments
    number_of_iterations = 10
    compute_free_energy = true
end

"""
    InferenceParameters

The learned posterior parameters: mean and covariance of the weight vector `ω`,
plus the shape and scale of the Gamma-distributed noise precision `s`.
"""
Base.@kwdef mutable struct InferenceParameters
    omega_mean       :: Any
    omega_covariance :: Any
    noise_shape      :: Any
    noise_scale      :: Any
end

"""
    InferenceState

The full state of a trained model: inference settings, learned parameters, and the
training data (needed later to impute missing values during prediction).
"""
Base.@kwdef mutable struct InferenceState
    arguments  :: InferenceArguments
    parameters :: InferenceParameters
    x_train    :: Any
    y_train    :: Any
end

"""
    initial_state(arguments, phi_dim)

Create a blank `InferenceState` with vague priors: zero-mean weights with very large variance
(we know nothing yet) and a near-flat Gamma prior on noise precision (shape 0.01, rate 0.01,
i.e. scale 100).
"""
function initial_state(arguments::InferenceArguments, phi_dim::Int)::InferenceState
    return InferenceState(
        arguments = arguments,
        parameters = InferenceParameters(
            omega_mean       = zeros(phi_dim),
            omega_covariance = 1e8 * Diagonal(ones(phi_dim)),
            noise_shape      = 1e-2,
            noise_scale      = 1e2
        ),
        x_train = nothing,
        y_train = nothing
    )
end

# --- Training ------------------------------------------------------------------

"""
    learn_parameters(arguments, x, y)

The main training loop. Takes feature vectors `x` (one per row) and targets `y`,
builds the basis-expanded design and runs variational inference to learn posterior
distributions over the regression weights and observation noise. Returns both the raw
inference result (for diagnostics) and the `InferenceState` (for making predictions).
"""
function learn_parameters(arguments::InferenceArguments, x::AbstractVector, y::AbstractVector)
    x_dim   = length(x[1])
    phi_s   = create_feature_functions(x_dim)
    phi_dim = length(phi_s)

    state = initial_state(arguments, phi_dim)

    omega_distribution = MvNormalMeanCovariance(
        state.parameters.omega_mean,
        state.parameters.omega_covariance
    )
    noise_distribution = GammaShapeScale(
        state.parameters.noise_shape,
        state.parameters.noise_scale
    )

    # We run loopy belief propagation, which requires an initial state.
    # We simply use our priors as the initial state.
    init = @initialization begin
        μ(ω) = omega_distribution
        q(s) = noise_distribution
    end

    priors = Dict(
        :ω => omega_distribution,
        :s => noise_distribution
    )

    result = infer(
        model = feature_regression_unknown_noise(
            ϕs = phi_s, priors = priors, x = x
        ),
        data           = (y = y,),
        initialization = init,
        returnvars     = (ω = KeepLast(), s = KeepLast()),
        iterations     = state.arguments.number_of_iterations,
        constraints    = MeanField(),
        options        = (limit_stack_depth = 300,),
        free_energy    = state.arguments.compute_free_energy,
    )

    state.x_train = x
    state.y_train = y
    state.parameters = InferenceParameters(
        omega_mean       = mean(result.posteriors[:ω]),
        omega_covariance = cov(result.posteriors[:ω]),
        noise_shape      = shape(result.posteriors[:s]),
        noise_scale      = scale(result.posteriors[:s])
    )

    return result, state
end

# --- Prediction ------------------------------------------------------------------

"""
    predict_known(state, x)

Run the RxInfer predictive model for a fully specified feature vector `x`.
Returns the predictive `(mean, var)`.
"""
function predict_known(state::InferenceState, x::AbstractVector{<:Real})
    phi_s = create_feature_functions(length(x))

    omega_distribution = MvNormalMeanCovariance(
        state.parameters.omega_mean,
        state.parameters.omega_covariance
    )
    noise_distribution = GammaShapeScale(
        state.parameters.noise_shape,
        state.parameters.noise_scale
    )

    priors = Dict(
        :ω => omega_distribution,
        :s => noise_distribution
    )

    init = @initialization begin
        μ(ω) = omega_distribution
        q(s) = noise_distribution
    end

    result = infer(
        model = feature_regression_predictive(
            ϕs = phi_s, priors = priors, x = x
        ),
        data           = (y = UnfactorizedData(missing),),
        predictvars    = (y = KeepLast(), ),
        returnvars     = (μ = KeepLast(), ),
        initialization = init,
        iterations     = state.arguments.number_of_iterations,
        constraints    = feature_regressions_predictive_constraints()
    )

    return (mean = mean(result.posteriors[:μ]), var = var(result.predictions[:y]))
end

"""
    estimate(state, x)

Predict the target for a feature vector `x` given the learned `state`. Entries of `x`
may be `missing` — for example, tomorrow's wind speed when you don't know it yet.

Unknown features are handled with the *law of total variance*. We don't know the
value of a missing feature, but the training data tells us which values are plausible.
So we average the prediction over the values seen in training:

    mean = E_x[ μ(x) ]
    var  = E_x[ σ²(x) ] + Var_x[ μ(x) ]
           ╰─ model ─╯   ╰─ because x is unknown ─╯

where `μ(x) = ϕ(x)ᵀ E[ω]` and `σ²(x)` is the predictive variance from RxInfer. To keep
it cheap, `σ²(x)` is evaluated once, at the training mean of the missing features
(it barely changes with `x` once the weights are well determined).

The result: unknown features that matter a lot (large weight, wide spread in the data)
widen the prediction a lot; unknown features that don't matter barely change it.

Returns a named tuple `(mean, std)` of the predictive distribution.
"""
function estimate(state::InferenceState, x::AbstractVector)
    missing_idx = findall(ismissing, x)

    # Fill missing features with their training mean and ask RxInfer for the
    # predictive distribution at that point.
    x̄ = Float64[ismissing(x[i]) ? mean(getindex.(state.x_train, i)) : x[i] for i in eachindex(x)]
    known = predict_known(state, x̄)

    isempty(missing_idx) && return (mean = known.mean, std = sqrt(known.var))

    # Spread of the mean prediction when the missing features take the values
    # seen in the training data (the rest of the input stays fixed).
    phi_s = create_feature_functions(length(x))
    m = state.parameters.omega_mean
    μs = map(state.x_train) do row
        xk = copy(x̄)
        xk[missing_idx] .= row[missing_idx]
        dot([ϕ(xk) for ϕ in phi_s], m)
    end

    return (mean = mean(μs), std = sqrt(known.var + var(μs)))
end
