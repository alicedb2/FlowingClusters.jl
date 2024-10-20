function deco!(axis)
    hidespines!(axis, :t, :r, :l, :b)
    hidedecorations!(axis, grid=true, minorgrid=true, ticks=true, label=true, ticklabels=true)
    return axis
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
    axislegend(ax)

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

    if burn >= N
        @error("Can't burn the whole chain, burn must be smaller than $N")
        return nothing
    end

    if burn >= 0
        if 0 < burn < 1
            burn = round(Int, burn * N)
        end
    else
        if -1 < burn < 0
            burn = round(Int, -burn * N)
        else
            burn = N + burn
        end
    end

    map_idx = chain.map_idx

    function deco!(axis)
        hidespines!(axis, :t, :r)
        hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
        return axis
    end

    if hasnn(chain.hyperparams)
        ysize = 2400
    else
        ysize = 1500
    end

    fig = Figure(size=(1200, ysize))

    p_map_axis = Axis(fig[1:2, 1], title="MAP state ($(length(chain.map_clusters)) clusters)")
    deco!(p_map_axis)
    plot!(p_map_axis, chain.map_clusters; proj=proj, orig=true, rev=rev, nbclusters=nbclusters)
    axislegend(p_map_axis)

    p_current_axis = Axis(fig[1:2, 2], title="Current state ($(length(chain.clusters)) clusters)")
    deco!(p_current_axis)
    plot!(p_current_axis, chain.clusters; proj=proj, orig=true, rev=rev, nbclusters=nbclusters)
    # axislegend(p_current_axis, framecolor=:white)

    offset = 0
    if hasnn(chain.hyperparams)
        offset = 2
        p_basemap_axis = Axis(fig[3:4, 1], title="MAP base state ($(length(chain.map_clusters)) clusters)")
        deco!(p_basemap_axis)
        plot!(p_basemap_axis, chain.map_clusters; proj=proj, orig=false, rev=rev, nbclusters=nbclusters)
        # axislegend(p_basemap_axis, framecolor=:white)

        p_basecurrent_axis = Axis(fig[3:4, 2], title="Current base state ($(length(chain.clusters)) clusters)")
        deco!(p_basecurrent_axis)
        plot!(p_basecurrent_axis, chain.clusters; proj=proj, orig=false, rev=rev, nbclusters=nbclusters)
        # axislegend(p_basecurrent_axis, framecolor=:white)
    end

    lpc = logprob_chain(chain, burn)
    logprob_axis = Axis(fig[offset + 3, 1], title="log probability", aspect=3)
    deco!(logprob_axis)
    lines!(logprob_axis, burn+1:N, lpc, label=nothing)
    hlines!(logprob_axis, [maximum(chain.logprob_chain)], label=nothing, color=:black)
    # hlines!(logprob_axis, [chain.map_logprob], label=nothing, color=:green)
    if map_idx > 0
        vlines!(logprob_axis, [map_idx], label=nothing, color=:green)
    end

    nbc = nbclusters_chain(chain, burn)
    nbc_axis = Axis(fig[offset + 3, 2], title="#cluster", aspect=3)
    deco!(nbc_axis)
    lines!(nbc_axis, burn+1:N, nbc, label=nothing)
    if map_idx > 0
        vlines!(nbc_axis, [map_idx], label=nothing, color=:black)
    end

    lcc = largestcluster_chain(chain, burn)
    lcc_axis = Axis(fig[offset + 4, 1], title="Largest cluster", aspect=3)
    deco!(lcc_axis)
    lines!(lcc_axis, burn+1:N, lcc, label=nothing)
    if map_idx > 0
        vlines!(lcc_axis, [map_idx], label=nothing, color=:black)
    end

    ac = alpha_chain(chain, burn)
    alpha_axis = Axis(fig[offset + 4, 2], title="α", aspect=3)
    deco!(alpha_axis)
    lines!(alpha_axis, burn+1:N, ac, label=nothing)
    if map_idx > 0
        vlines!(alpha_axis, [map_idx], label=nothing, color=:black)
    end

    muc = mu_chain(Matrix, chain, burn)
    mu_axis = Axis(fig[offset + 5, 1], title="μ", aspect=3)
    deco!(mu_axis)
    for mucomponent in eachrow(muc)
        lines!(mu_axis, burn+1:N, mucomponent, label=nothing)
    end
    if map_idx > 0
        vlines!(mu_axis, [map_idx], label=nothing, color=:black)
    end

    lc = lambda_chain(chain, burn)
    lambda_axis = Axis(fig[offset + 5, 2], title="λ", aspect=3)
    deco!(lambda_axis)
    lines!(lambda_axis, burn+1:N, lc, label=nothing)
    if map_idx > 0
        vlines!(lambda_axis, [map_idx], label=nothing, color=:black)
    end

    pc = psi_chain(Matrix, chain, burn)
    psi_axis = Axis(fig[offset + 6, 1], title="Ψ", aspect=3)
    deco!(psi_axis)
    for psicomponent in eachrow(pc)
        lines!(psi_axis, burn+1:N, psicomponent, label=nothing)
    end
    if map_idx > 0
        vlines!(psi_axis, [map_idx], label=nothing, color=:black)
    end

    nc = nu_chain(chain, burn)
    nu_axis = Axis(fig[offset + 6, 2], title="ν", aspect=3)
    deco!(nu_axis)
    lines!(nu_axis, burn+1:N, nc, label=nothing)
    if map_idx > 0
        vlines!(nu_axis, [map_idx], label=nothing, color=:black)
    end

    if hasnn(chain.hyperparams)
        nnc = nn_chain(Matrix, chain, burn)
        nn_axis = Axis(fig[9, 1:2], title="FFJORD neural network prior")
        deco!(nn_axis)
        for p in eachrow(nnc)
            lines!(nn_axis, burn+1:N, collect(p), label=nothing, linewidth=1, alpha=0.5)
        end
        if map_idx > 0
            vlines!(nn_axis, [map_idx], label=nothing, color=:black)
        end

        nnalphac = nn_alpha_chain(chain, burn)
        nnscale_axis = Axis(fig[10, 1], title="FFJORD hyperprior log α")
        deco!(nnscale_axis)
        lines!(nnscale_axis, burn+1:N, log.(nnalphac), label=nothing)
        if map_idx > 0
            vlines!(nnscale_axis, [map_idx], label=nothing, color=:black)
        end

        nnscalec = nn_scale_chain(chain, burn)
        nnscale_axis = Axis(fig[10, 2], title="FFJORD hyperprior scale")
        deco!(nnscale_axis)
        lines!(nnscale_axis, burn+1:N, nnscalec, label=nothing)
        if map_idx > 0
            vlines!(nnscale_axis, [map_idx], label=nothing, color=:black)
        end
    end

    return fig
end

function deformation_figure_2d(hyperparams; lims=((-4, 4), (-4, 4)), proj=[1, 2], realzs=nothing, basezs=realzs, rng=default_rng(), t=1.0, nbpoints=300, nblines=10)

    if !hasnn(hyperparams)
        @error("Hyperparameters do not contain a FFJORD neural network")
        return nothing
    end

    d = datadimension(hyperparams)

    @assert length(Set(proj)) == 2 "Specify at least 2 different dimensions"
    @assert length(proj) + (isnothing(realzs) ? 0 : length(realzs)) == d "You must specify as many (real) z values as the number of dimensions minus 2"
    @assert length(proj) + (isnothing(basezs) ? 0 : length(basezs)) == d "You must specify as many (base) z values as the number of dimensions minus 2"

    function _deco!(axis)
        hidespines!(axis, :t, :r)
        hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
        return axis
    end

    baseprobgridhoriz = reduce(hcat, [[x, y] for y in LinRange(lims[2]..., nblines), x in LinRange(lims[1]..., nbpoints)])
    baseprobgridvert = reduce(hcat, [[x, y] for x in LinRange(lims[1]..., nblines), y in LinRange(lims[2]..., nbpoints)])
    baseprobgrid = hcat(baseprobgridhoriz, baseprobgridvert)

    realprobgridhoriz = reduce(hcat, [[x, y] for y in LinRange(lims[2]..., nblines), x in LinRange(lims[1]..., nbpoints)])
    realprobgridvert = reduce(hcat, [[x, y] for x in LinRange(lims[1]..., nblines), y in LinRange(lims[2]..., nbpoints)])
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

    _, _, basespace = forwardffjord(rng, realprobgrid, hyperparams, t=t)
    realspace = backwardffjord(rng, baseprobgrid, hyperparams, t=t)

    fig = Figure(size=(900, 500));
    ax1 = Axis(fig[1, 1], xlabel="Real axis 1 -> Base axis 1", ylabel="Real axis 2 -> Base axis 2")
    scatter!(ax1, basespace[proj[1], :], basespace[proj[2], :], markersize=2, label=nothing); 
    scatter!(ax1, realprobgrid[proj[1], :], realprobgrid[proj[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);

    scatter!(ax1, Float64[], Float64[], color=:grey, markersize=15, label="Real/environmental space"); 
    scatter!(ax1, Float64[], Float64[], color=Cycled(1), markersize=15, label="Base space"); 
    
    ylims!(ax1, nothing, 7);
    axislegend(ax1, framecolor=:white); 
    _deco!(ax1)

    ax2 = Axis(fig[1, 2], xlabel="Base axis 1 -> Real axis 1", ylabel="Base axis 2 -> Real axis 2")
    scatter!(ax2, realspace[proj[1], :], realspace[proj[2], :], markersize=2, label=nothing); 
    scatter!(ax2, baseprobgrid[proj[1], :], baseprobgrid[proj[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    scatter!(ax2, Float64[], Float64[], color=:grey, markersize=15, label="Base space");
    scatter!(ax2, Float64[], Float64[], color=Cycled(1), markersize=15, label="Real/environmental space");
    ylims!(ax2, nothing, 7);
    axislegend(ax2, framecolor=:white);
    _deco!(ax2);

    return fig
end