function optimize_hyperparams(
    clusters::AbstractVector{<:AbstractCluster{T, D, E}},
    hyperparams0::AbstractFCHyperparams{T, D};
    verbose=true
    ) where {T, D, E}

    ax = getfield(hyperparams0._, :axes)
    opt_options = Options(iterations=1000, show_trace=verbose)

    # We do not include nn_scale in the optimization
    if hasnn(hyperparams0) && hyperparams0.nn.prior.scale == one(T)
        x0 = transform(hyperparams0._)[1:end-1]
    else
        x0 = transform(hyperparams0._)
    end

    function objfun(x)
        if hasnn(hyperparams0) && hyperparams0.nn.prior.scale == one(T)
            transformed_hparray = ComponentArray([x[:]; 0.0], ax)
        else
            transformed_hparray = ComponentArray(x[:], ax)
        end
        params = backtransform!(transformed_hparray)
        logprob = -logprobgenerative(clusters, params)
        if !isfinite(logprob)
            @warn println(params)
        end
        return logprob
    end

    optres = optimize(objfun, x0, LBFGS(), opt_options)

    if hasnn(hyperparams0) && hyperparams0.nn.prior.scale == one(T)
        opt_hp = backtransform(ComponentArray([minimizer(optres)[:]; 0.0], ax))
    else
        opt_hp = backtransform(ComponentArray(minimizer(optres)[:], ax))
    end

    hyperparams = deepcopy(hyperparams0)
    hyperparams._ .= opt_hp

    return (optres=optres, hyperparams=hyperparams)

end

function optimize_hyperparams!(
    clusters::AbstractVector{<:AbstractCluster{T, D, E}},
    hyperparams0::AbstractFCHyperparams{T, D};
    verbose=true
    ) where {T, D, E}

    optres, hyperparams = optimize_hyperparams(clusters, hyperparams0, verbose=verbose)
    hyperparams0._ .= hyperparams._

    return hyperparams0

end