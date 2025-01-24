function predictive_distribution(
    clusters::AbstractVector{<:AbstractCluster{T, D, E}},
    hyperparams::AbstractFCHyperparams{T, D};
    ignore_weights=false,
    ignore_empty=false
    ) where {T, D, E}

    alpha = hyperparams._.pyp.alpha
    mu, lambda, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.nu
    psi = foldpsi(hyperparams._.niw.flatL)

    if ignore_weights
        weights = ones(T, length(clusters) + 1)
        if ignore_empty
            weights[end] = zero(T)
        end
    else
        weights = T.(length.(clusters))
        if ignore_empty
            push!(weights, zero(T))
        else
            push!(weights, alpha)
        end
    end

    weights ./= sum(weights)

    updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)
    # println("in predictive_distribution(clusters, hyperparams):    #cluster=$(length(updated_mvstudent_degs_mus_sigs))")
    return predictive_distribution(weights, updated_mvstudent_degs_mus_sigs)
end

function predictive_distribution(
    component_weights::AbstractVector{T},
    mvstudent_degs_mus_sigs::AbstractVector{@NamedTuple{df_c::T, mu_c::Vector{T}, sigma_c::Matrix{T}}}
    ) where T

    @assert isapprox(sum(component_weights), one(T)) "sum(component_weights) = $(sum(component_weights)) !~= 1"

    d = length(first(mvstudent_degs_mus_sigs).mu_c)

    # println("in predictive_distribution(weights, mvparams):    #weights=$(length(component_weights)) #mvstclusters=$(length(mvstudent_degs_mus_sigs))")

    return MixtureModel(
            [MvTDist(df, d, mu, PDMat((sig + sig')/2)) for (df, mu, sig) in mvstudent_degs_mus_sigs],
            Categorical(component_weights)
        )

end

function tail_probability(
    component_weights::AbstractVector{T},
    mvstudent_degs_mus_sigs::AbstractVector{@NamedTuple{df_c::T, mu_c::Vector{T}, sigma_c::Matrix{T}}};
    nb_rejection_samples=10000,
    per_cluster=false,
    rng=default_rng()
    ) where T

    if !per_cluster
        dist = predictive_distribution(component_weights, mvstudent_degs_mus_sigs)

        sample_draw = rand(rng, dist, nb_rejection_samples)
        logpdf_samples = logpdf(dist, sample_draw)

        function tailprob_func(coordinate::AbstractVector{T})
            return tailprob_func(reshape(coordinate, :, 1))[1]
        end

        function tailprob_func(coordinates::AbstractArray{T})
            isocontours = logpdf(dist, coordinates)
            in_tail = logpdf_samples .<= reshape(isocontours, 1, size(isocontours)...)
            tail_probs = dropdims(mean(in_tail, dims=1), dims=1)
            return tail_probs
        end

        return tailprob_func
    else

        cluster_dists = predictive_distribution.(fill([1.0], length(component_weights)), (x->[x]).(mvstudent_degs_mus_sigs))
        cluster_draws = rand.(rng, cluster_dists, nb_rejection_samples)
        cluster_logpdf_samples = logpdf.(cluster_dists, cluster_draws)

        function clusters_tailprob_func(coordinate::AbstractVector{T})
            return clusters_tailprob_func(reshape(coordinate, :, 1))[1]
        end

        function clusters_tailprob_func(coordinates::AbstractArray{T, D}) where D
            cluster_isocontours = logpdf.(cluster_dists, Ref(coordinates))
            cluster_in_tails = [logpdf_samples .<= reshape(isocontours, 1, size(isocontours)...) for (isocontours, logpdf_samples) in zip(cluster_isocontours, cluster_logpdf_samples)]
            cluster_tail_probs = [mean(in_tail, dims=1) for in_tail in cluster_in_tails]
            max_cluster_tail_probs = dropdims(maximum(cat(cluster_tail_probs..., dims=1), dims=1), dims=1)
            return max_cluster_tail_probs
        end

        return clusters_tailprob_func

    end


end

function tail_probability(
    clusters::AbstractVector{<:AbstractCluster{T, D, E}},
    hyperparams::AbstractFCHyperparams{T, D};
    nb_rejection_samples=10000,
    rng=default_rng(),
    per_cluster=false,
    ignore_weights=false,
    ignore_empty=false
    ) where {T, D, E}

    alpha = hyperparams._.pyp.alpha
    mu, lambda, nu = hyperparams._.niw.mu, hyperparams._.niw.lambda, hyperparams._.niw.nu
    psi = foldpsi(hyperparams._.niw.flatL)

    if ignore_weights
        weights = ones(T, length(clusters) + 1)
        if ignore_empty
            weights[end] = zero(T)
        end
    else
        weights = T.(length.(clusters))
        if ignore_empty
            push!(weights, zero(T))
        else
            push!(weights, T(alpha))
        end
    end

    weights ./= sum(weights)

    updated_mvstudent_degs_mus_sigs = updated_mvstudent_params(clusters, mu, lambda, psi, nu, add_empty=true)
    # println("in tail_probability(clusters, hyperparams):       #clusters=$(length(clusters)) #mvstclusters=$(length(updated_mvstudent_degs_mus_sigs))")

    if hasnn(hyperparams)
        basetailprob_func = tail_probability(weights, updated_mvstudent_degs_mus_sigs, nb_rejection_samples=nb_rejection_samples, per_cluster=per_cluster, rng=rng)
        function tailprob_func(coordinates::AbstractArray)
            # Project elements to the base space
            _, _, base_elements = forwardffjord(rng, coordinates, hyperparams)
            return basetailprob_func(base_elements)
        end
        return tailprob_func
    else
        return tail_probability(weights, updated_mvstudent_degs_mus_sigs, nb_rejection_samples=nb_rejection_samples, per_cluster=per_cluster, rng=rng)
    end

end

function tail_probability(
    clusters_samples::AbstractVector{<:AbstractVector{<:AbstractCluster{T, D, E}}},
    hyperparams_samples::AbstractVector{<:AbstractFCHyperparams{T, D}};
    nb_rejection_samples=10000,
    rng=default_rng(),
    per_cluster=false,
    ignore_weights=false,
    ignore_empty=false
    ) where {T, D, E}

    tailprob_funcs = tail_probability.(clusters_samples, hyperparams_samples, nb_rejection_samples=nb_rejection_samples, rng=rng, per_cluster=per_cluster, ignore_weights=ignore_weights, ignore_empty=ignore_empty)

    function tailprob(coordinates::AbstractArray)
        tailprob_samples = [tailprob_func(coordinates) for tailprob_func in tailprob_funcs]
        if coordinates isa AbstractVector
            return tailprob_samples
        else
            return stack(tailprob_samples, dims=ndims(coordinates))
        end
    end

    return tailprob

end

function tail_probability_summary(
    clusters_samples::AbstractVector{<:AbstractVector{<:AbstractCluster{T, D, E}}},
    hyperparams_samples::AbstractVector{<:AbstractFCHyperparams{T, D}};
    nb_rejection_samples=10000,
    rng=default_rng(),
    per_cluster=false,
    ignore_weights=false,
    ignore_empty=false
    ) where {T, D, E}

    tailprob_func = tail_probability(clusters_samples, hyperparams_samples, nb_rejection_samples=nb_rejection_samples, rng=rng, per_cluster=per_cluster, ignore_weights=ignore_weights, ignore_empty=ignore_empty)

    function _CI95(x::AbstractArray)
        return quantile(x, 0.975) - quantile(x, 0.025)
    end

    function _CI90(x::AbstractArray)
        return quantile(x, 0.95) - quantile(x, 0.05)
    end

    # Mode from Freedman-Diaconis rule
    function _modefd(x::AbstractArray)
        fdw = freedmandiaconis(x)
        fdbins = nothing
        fdbins = LinRange(minimum(x), maximum(x), ceil(Int64, (maximum(x) - minimum(x))/fdw) + 1)
        histfd = fit(Histogram, x, fdbins)
        modefdidx = sortperm(histfd.weights)[end]
        return mean(histfd.edges[1][modefdidx:modefdidx+1])

    end

    # Mode from Doane's rule
    function _modedoane(x::AbstractArray)
        dk = ceil(Int64, doane(x))
        dbins = LinRange(minimum(x), maximum(x), dk + 1)
        histd = fit(Histogram, x, dbins)
        modedidx = sortperm(histd.weights)[end]
        return mean(histd.edges[1][modedidx:modedidx+1])
    end

    function summaries(coordinates::AbstractArray)
        tps = tailprob_func(coordinates)

        if coordinates isa AbstractVector

            return (median=median(tps),
                    mean=mean(tps),
                    std=std(tps),
                    iqr=iqr(tps),
                    CI95=_CI95(tps),
                    CI90=_CI90(tps),
                    quantile=q -> quantile(tps, q),
                    modefd=_modefd(tps),
                    modedoane=_modedoane(tps)
                    )
        else
            return (median=dropdims(mapslices(median, tps, dims=ndims(tps)), dims=ndims(tps)),
                    mean=dropdims(mapslices(mean, tps, dims=ndims(tps)), dims=ndims(tps)),
                    std=dropdims(mapslices(std, tps, dims=ndims(tps)), dims=ndims(tps)),
                    iqr=dropdims(mapslices(iqr, tps, dims=ndims(tps)), dims=ndims(tps)),
                    CI95=dropdims(mapslices(_CI95, tps, dims=ndims(tps)), dims=ndims(tps)),
                    CI90=dropdims(mapslices(_CI90, tps, dims=ndims(tps)), dims=ndims(tps)),
                    quantile=q -> dropdims(mapslices(sl -> quantile(sl, q), tps, dims=ndims(tps)), dims=ndims(tps)),
                    modefd=dropdims(mapslices(_modefd, tps, dims=ndims(tps)), dims=ndims(tps)),
                    modedoane=dropdims(mapslices(_modedoane, tps, dims=ndims(tps)), dims=ndims(tps))
                    )
        end
    end

end