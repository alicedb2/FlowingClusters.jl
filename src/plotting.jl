function _deco!(axis; hide=(:t, :r, :l, :b), grid=true, minorgrid=true, ticks=false, ticklabels=false, label=false)
    hidespines!(axis, hide...)
    hidedecorations!(axis, label=label, grid=grid, minorgrid=minorgrid, ticks=ticks, ticklabels=ticklabels)
    return axis
end

function _scale_lims(lower, upper, scaling_factor)
    lower = min(lower, upper)
    upper = max(lower, upper)
    center = (lower + upper) / 2
    new_width = scaling_factor * (upper - lower)
    new_lower = center - new_width / 2
    new_upper = center + new_width / 2
    return (new_lower, new_upper)
end

function plot!(ax, clusters::AbstractVector{<:AbstractCluster{T, D, E}}; proj=[1, 2], rev=false, nbclusters=nothing, orig=false, plot_kw...) where {T, D, E}

    if D >= 2
        size(proj, 1) == 2 || error("Can only support plotting 1 or 2 dimensions for now")
        clusters = project_clusters(sort(clusters, by=length, rev=!rev), proj, orig=orig)
    else
        clusters = [[first(el) for el in cl] for cl in Vector.(sort(clusters, by=length, rev=!rev), orig=orig)]
    end

    if nbclusters === nothing || nbclusters < 0
        nbclusters = length(clusters)
    end

    for (cluster, i) in zip(clusters, 1:nbclusters)
        if !isempty(cluster)
            if D >= 2
                x = getindex.(clusters[i], 1)
                y = getindex.(clusters[i], 2)
                scatter!(ax, x, y, label="$(length(cluster))", color=Cycled(i), plot_kw...)
            else
                hist!(ax, cluster, label="$(length(cluster))", color=Cycled(i), plot_kw...)
            end
        end
    end
    _deco!(ax, hide=(:t, :r))
    axislegend(ax, nbanks=2)

    return ax
end

function plot(clusters::AbstractVector{<:AbstractCluster}; proj=[1, 2], rev=false, nbclusters=nothing, orig=false, plot_kw...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, clusters, proj=proj, rev=rev, nbclusters=nbclusters, orig=orig, plot_kw...)
    return fig
end

function plot(chain::FCChain; proj=[1, 2], burn=0, rev=false, nbclusters=nothing)

    N = length(chain)

    burn = _burnlength(N, burn)

    map_idx = chain.map_idx

    if hasnn(chain.hyperparams)
        ysize = 2800
        # ysize = 2400
    else
        ysize = 1500
    end

    fig = Figure(size=(1400, ysize))

    p_map_axis = Axis(fig[1:2, 1], title="MAP state ($(length(chain.map_clusters)) clusters)")
    plot!(p_map_axis, chain.map_clusters; proj=proj, orig=true, rev=rev, nbclusters=nbclusters)

    p_current_axis = Axis(fig[1:2, 2], title="Current state ($(length(chain.clusters)) clusters)")
    plot!(p_current_axis, chain.clusters; proj=proj, orig=true, rev=rev, nbclusters=nbclusters)

    offset = 0
    if hasnn(chain.hyperparams)
        offset = 2
        p_basemap_axis = Axis(fig[3:4, 1], title="MAP base state ($(length(chain.map_clusters)) clusters)")
        plot!(p_basemap_axis, chain.map_clusters; proj=proj, orig=false, rev=rev, nbclusters=nbclusters)

        p_basecurrent_axis = Axis(fig[3:4, 2], title="Current base state ($(length(chain.clusters)) clusters)")
        plot!(p_basecurrent_axis, chain.clusters; proj=proj, orig=false, rev=rev, nbclusters=nbclusters)
    end

    lpc = logprob_chain(chain, burn)
    logprob_axis = Axis(fig[offset + 3, 1], title="log probability", aspect=3)
    _deco!(logprob_axis, hide=(:t, :r))
    lines!(logprob_axis, burn+1:N, lpc, label=nothing)
    hlines!(logprob_axis, [maximum(chain.logprob_chain)], label=nothing, color=:black)
    # hlines!(logprob_axis, [chain.map_logprob], label=nothing, color=:green)
    if map_idx > 0
        vlines!(logprob_axis, [map_idx], label=nothing, color=:green)
    end

    nbc = nbclusters_chain(chain, burn)
    nbc_axis = Axis(fig[offset + 3, 2], title="#cluster", aspect=3)
    _deco!(nbc_axis, hide=(:t, :r))
    lines!(nbc_axis, burn+1:N, nbc, label=nothing)
    if map_idx > 0
        vlines!(nbc_axis, [map_idx], label=nothing, color=:black)
    end

    lcc = largestcluster_chain(chain, burn)
    lcc_axis = Axis(fig[offset + 4, 1], title="Largest cluster", aspect=3)
    _deco!(lcc_axis, hide=(:t, :r))
    lines!(lcc_axis, burn+1:N, lcc, label=nothing)
    if map_idx > 0
        vlines!(lcc_axis, [map_idx], label=nothing, color=:black)
    end

    ac = alpha_chain(chain, burn)
    alpha_axis = Axis(fig[offset + 4, 2], title="α", aspect=3)
    _deco!(alpha_axis, hide=(:t, :r))
    lines!(alpha_axis, burn+1:N, ac, label=nothing)
    if map_idx > 0
        vlines!(alpha_axis, [map_idx], label=nothing, color=:black)
    end

    muc = mu_chain(Matrix, chain, burn)
    mu_axis = Axis(fig[offset + 5, 1], title="μ", aspect=3)
    _deco!(mu_axis, hide=(:t, :r))
    for mucomponent in eachrow(muc)
        lines!(mu_axis, burn+1:N, mucomponent, label=nothing)
    end
    if map_idx > 0
        vlines!(mu_axis, [map_idx], label=nothing, color=:black)
    end

    lc = lambda_chain(chain, burn)
    lambda_axis = Axis(fig[offset + 5, 2], title="λ", aspect=3)
    _deco!(lambda_axis, hide=(:t, :r))
    lines!(lambda_axis, burn+1:N, lc, label=nothing)
    if map_idx > 0
        vlines!(lambda_axis, [map_idx], label=nothing, color=:black)
    end

    pc = psi_chain(Matrix, chain, burn)
    psi_axis = Axis(fig[offset + 6, 1], title="Ψ", aspect=3)
    _deco!(psi_axis, hide=(:t, :r))
    for psicomponent in eachrow(pc)
        lines!(psi_axis, burn+1:N, psicomponent, label=nothing)
    end
    if map_idx > 0
        vlines!(psi_axis, [map_idx], label=nothing, color=:black)
    end

    nc = nu_chain(chain, burn)
    nu_axis = Axis(fig[offset + 6, 2], title="ν", aspect=3)
    _deco!(nu_axis, hide=(:t, :r))
    lines!(nu_axis, burn+1:N, nc, label=nothing)
    if map_idx > 0
        vlines!(nu_axis, [map_idx], label=nothing, color=:black)
    end

    if hasnn(chain.hyperparams)
        nnc = nn_params_chain(Matrix, chain, burn)
        nn_axis = Axis(fig[9:10, 1:2], title="FFJORD neural network prior")
        _deco!(nn_axis, hide=(:t, :r))
        for p in eachrow(nnc)
            # scatter!(nn_axis, burn+1:N, collect(p), label=nothing, markersize=2, alpha=0.5)
            barplot!(nn_axis, burn+1:N, collect(p), fillto=collect(p) .- 0.01, gap=-0.01, strokewidth=0, label=nothing)
            # lines!(nn_axis, burn+1:N, collect(p), label=nothing, linewidth=1, alpha=0.5)
        end
        if map_idx > 0
            vlines!(nn_axis, [map_idx], label=nothing, color=:black)
        end

        # nnhpc = nn_hyperparams_chain(Matrix, chain, burn)

        # mu_axis = Axis(fig[10, 1], title="FFJORD hyperprior μ")
        # _deco!(mu_axis, hide=(:t, :r))
        # lines!(mu_axis, burn+1:N, nnhpc[1, :], label=nothing)
        # if map_idx > 0
        #     vlines!(mu_axis, [map_idx], label=nothing, color=:black)
        # end

        # lambda_axis = Axis(fig[10, 2], title="FFJORD hyperprior log λ")
        # _deco!(lambda_axis, hide=(:t, :r))
        # lines!(lambda_axis, burn+1:N, nnhpc[2, :], label=nothing)
        # if map_idx > 0
        #     vlines!(lambda_axis, [map_idx], label=nothing, color=:black)
        # end

        # alpha_axis = Axis(fig[11, 1], title="FFJORD hyperprior log α")
        # _deco!(alpha_axis, hide=(:t, :r))
        # lines!(alpha_axis, burn+1:N, nnhpc[3, :], label=nothing)
        # if map_idx > 0
        #     vlines!(alpha_axis, [map_idx], label=nothing, color=:black)
        # end

        # beta_axis = Axis(fig[11, 2], title="FFJORD hyperprior log β")
        # _deco!(beta_axis, hide=(:t, :r))
        # lines!(beta_axis, burn+1:N, nnhpc[4, :], label=nothing)
        # if map_idx > 0
        #     vlines!(beta_axis, [map_idx], label=nothing, color=:black)
        # end

    end

    return fig
end

function deformation_plot(hyperparams; proj=[1, 2], gridlims=((-4, 4), (-4, 4)), realzs=nothing, basezs=realzs, rng=default_rng(), t=1.0, nbpoints=300, nblines=10, bounds_scaling_factor=1.5)

    if !hasnn(hyperparams)
        @error("Hyperparameters do not contain a FFJORD neural network")
        return nothing
    end

    d = datadimension(hyperparams)

    @assert length(Set(proj)) == 2 "Specify at least 2 different dimensions"
    @assert length(proj) + (isnothing(realzs) ? 0 : length(realzs)) == d "You must specify as many (real) z values as the number of dimensions minus 2"
    @assert length(proj) + (isnothing(basezs) ? 0 : length(basezs)) == d "You must specify as many (base) z values as the number of dimensions minus 2"

    baseprobgridhoriz = reduce(hcat, [[x, y] for y in LinRange(gridlims[2]..., nblines), x in LinRange(gridlims[1]..., nbpoints)])
    baseprobgridvert = reduce(hcat, [[x, y] for x in LinRange(gridlims[1]..., nblines), y in LinRange(gridlims[2]..., nbpoints)])
    baseprobgrid = hcat(baseprobgridhoriz, baseprobgridvert)

    realprobgridhoriz = reduce(hcat, [[x, y] for y in LinRange(gridlims[2]..., nblines), x in LinRange(gridlims[1]..., nbpoints)])
    realprobgridvert = reduce(hcat, [[x, y] for x in LinRange(gridlims[1]..., nblines), y in LinRange(gridlims[2]..., nbpoints)])
    realprobgrid = hcat(realprobgridhoriz, realprobgridvert)

    if !isnothing(realzs)
        zsproj = setdiff(1:d, proj)
        _realprobgrid = zeros(d, 2 * nblines * nbpoints)
        _realprobgrid[proj, :] .= realprobgrid
        _realprobgrid[zsproj, :] .= realzs
        realprobgrid = _realprobgrid
    end
    if !isnothing(realzs)
        zsproj = setdiff(1:d, proj)
        _baseprobgrid = zeros(d, 2 * nblines * nbpoints)
        _baseprobgrid[proj, :] .= baseprobgrid
        _baseprobgrid[zsproj, :] .= realzs
        baseprobgrid = _baseprobgrid
    end

    _, basespace = forwardffjord(rng, realprobgrid, hyperparams, t=t)
    realspace = backwardffjord(rng, baseprobgrid, hyperparams, t=t)

    fig = Figure(size=(900, 500));
    ax1 = Axis(fig[1, 1], xlabel="Real axis $(proj[1]) -> Base axis $(proj[1])", ylabel="Real axis $(proj[2]) -> Base axis $(proj[2])")
    scatter!(ax1, basespace[proj[1], :], basespace[proj[2], :], markersize=2, label=nothing);
    scatter!(ax1, realprobgrid[proj[1], :], realprobgrid[proj[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);

    scatter!(ax1, Float64[], Float64[], color=:grey, markersize=15, label="Real/environmental space");
    scatter!(ax1, Float64[], Float64[], color=Cycled(1), markersize=15, label="Base space");

    # ylims!(ax1, nothing, 7);
    xlims!(ax1, _scale_lims(first(gridlims)..., bounds_scaling_factor));
    ylims!(ax1, _scale_lims(first(gridlims)..., bounds_scaling_factor));

    axislegend(ax1, framecolor=:white);
    _deco!(ax1, hide=(:t, :r))

    ax2 = Axis(fig[1, 2], xlabel="Base axis $(proj[1]) -> Real axis $(proj[1])", ylabel="Base axis $(proj[2]) -> Real axis $(proj[2])")
    scatter!(ax2, realspace[proj[1], :], realspace[proj[2], :], markersize=2, label=nothing);
    scatter!(ax2, baseprobgrid[proj[1], :], baseprobgrid[proj[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    scatter!(ax2, Float64[], Float64[], color=:grey, markersize=15, label="Base space");
    scatter!(ax2, Float64[], Float64[], color=Cycled(1), markersize=15, label="Real/environmental space");
    # ylims!(ax2, nothing, 7);
    xlims!(ax2, _scale_lims(last(gridlims)..., bounds_scaling_factor));
    ylims!(ax2, _scale_lims(last(gridlims)..., bounds_scaling_factor));

    axislegend(ax2, framecolor=:white);
    _deco!(ax2, hide=(:t, :r));

    return fig
end

# fig = Figure();
# ax = Axis(fig[1, 1]);
# streamplot!(ax,(x,y)->Point2(chain.hyperparams.ffjord.nn(Float64[x, y], chain.map_hyperparams._.nn.params, chain.hyperparams.ffjord.nns.model)[1]...), -6..6, -6..6);
# scatter!(Tuple.(eachcol(Matrix(chain.clusters, orig=true))));
# scatter!(Tuple.(eachcol(Matrix(chain.clusters, orig=false))));
# fig


function flow_plot(hyperparams; proj=[1, 2], gridlims=((-4, 4), (-4, 4)), realzs=nothing, basezs=realzs, rng=default_rng(), t=1.0, nbpoints=50, nblines=20, bounds_scaling_factor=1.02)

    # streamplot((x,y)->Point2(chain.hyperparams.ffjord.nn(Float64[x, y], chain.map_hyperparams._.nn.params, chain.hyperparams.ffjord.nns.model)[1]...), -2..2, -2..2)

    if !hasnn(hyperparams)
        @error("Hyperparameters do not contain a FFJORD neural network")
        return nothing
    end

    d = datadimension(hyperparams)

    @assert length(Set(proj)) == 2 "Specify at least 2 different dimensions"
    @assert length(proj) + (isnothing(realzs) ? 0 : length(realzs)) == d "You must specify as many (real) z values as the number of dimensions minus 2"
    @assert length(proj) + (isnothing(basezs) ? 0 : length(basezs)) == d "You must specify as many (base) z values as the number of dimensions minus 2"

    baseprobgridhoriz = reduce(hcat, [[x, y] for y in LinRange(gridlims[2]..., nblines), x in LinRange(gridlims[1]..., nblines)])
    baseprobgridvert = reduce(hcat, [[x, y] for x in LinRange(gridlims[1]..., nblines), y in LinRange(gridlims[2]..., nblines)])
    baseprobgrid = hcat(baseprobgridhoriz, baseprobgridvert)

    realprobgridhoriz = reduce(hcat, [[x, y] for y in LinRange(gridlims[2]..., nblines), x in LinRange(gridlims[1]..., nblines)])
    realprobgridvert = reduce(hcat, [[x, y] for x in LinRange(gridlims[1]..., nblines), y in LinRange(gridlims[2]..., nblines)])
    realprobgrid = hcat(realprobgridhoriz, realprobgridvert)

    if !isnothing(realzs)
        zsproj = setdiff(1:d, proj)
        _realprobgrid = zeros(d, 2 * nblines * nblines)
        _realprobgrid[proj, :] .= realprobgrid
        _realprobgrid[zsproj, :] .= realzs
        realprobgrid = _realprobgrid
    end
    if !isnothing(realzs)
        zsproj = setdiff(1:d, proj)
        _baseprobgrid = zeros(d, 2 * nblines * nblines)
        _baseprobgrid[proj, :] .= baseprobgrid
        _baseprobgrid[zsproj, :] .= realzs
        baseprobgrid = _baseprobgrid
    end

    fig = Figure(size=(900, 500));
    ax1 = Axis(fig[1, 1], xlabel="Real axis $(proj[1]) -> Base axis $(proj[1])", ylabel="Real axis $(proj[2]) -> Base axis $(proj[2])")
    basespace = nothing
    scatter!(ax1, realprobgrid[proj[1], :], realprobgrid[proj[2], :], color=:grey, markersize=6, alpha=0.2, label=nothing);
    for _t in LinRange(0, 1, 50)
        _, basespace = forwardffjord(rng, realprobgrid, hyperparams, t=_t)
        scatter!(ax1, basespace[proj[1], :], basespace[proj[2], :], markersize=2, color=Cycled(1), label=nothing);
    end
    scatter!(ax1, basespace[proj[1], :], basespace[proj[2], :], markersize=6, color=Cycled(1), label=nothing);

    scatter!(ax1, Float64[], Float64[], color=:grey, markersize=15, label="Real space");
    scatter!(ax1, Float64[], Float64[], color=Cycled(1), markersize=15, label="Flow from real to base space");

    # ylims!(ax1, nothing, 7);
    xlims!(ax1, _scale_lims(first(gridlims)..., bounds_scaling_factor));
    ylims!(ax1, _scale_lims(first(gridlims)..., bounds_scaling_factor));

    axislegend(ax1, framecolor=:white);
    _deco!(ax1, hide=(:t, :r))

    ax2 = Axis(fig[1, 2], xlabel="Base axis $(proj[1]) -> Real axis $(proj[1])", ylabel="Base axis $(proj[2]) -> Real axis $(proj[2])")

    scatter!(ax2, baseprobgrid[proj[1], :], baseprobgrid[proj[2], :], color=:grey, alpha=0.2, markersize=6, label=nothing);
    realspace = nothing
    for _t in LinRange(0, 1, nbpoints)
        realspace = backwardffjord(rng, baseprobgrid, hyperparams, t=_t)
        scatter!(ax2, realspace[proj[1], :], realspace[proj[2], :], markersize=2, color=Cycled(1), label=nothing);
    end
    scatter!(ax2, realspace[proj[1], :], realspace[proj[2], :], markersize=6, color=Cycled(1), label=nothing);

    scatter!(ax2, Float64[], Float64[], color=:grey, markersize=15, label="Base space");
    scatter!(ax2, Float64[], Float64[], color=Cycled(1), markersize=15, label="Flow from base to real space");
    # ylims!(ax2, nothing, 7);
    xlims!(ax2, _scale_lims(last(gridlims)..., bounds_scaling_factor));
    ylims!(ax2, _scale_lims(last(gridlims)..., bounds_scaling_factor));

    axislegend(ax2, framecolor=:white);
    _deco!(ax2, hide=(:t, :r));

    return fig
end