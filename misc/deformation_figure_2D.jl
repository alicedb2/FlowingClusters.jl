function deformation_figure_2d(chain)

    function _deco!(axis)
        hidespines!(axis, :t, :r)
        hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
        return axis
    end
    
    basexlims, baseylims = extrema(Matrix(chain.map_clusters, orig=false), dims=2)
    realxlims, realylims = extrema(Matrix(chain.map_clusters, orig=true), dims=2)

    _nbpoints = 300

    baseprobgridxs = reduce(hcat, [[x, y] for y in LinRange(baseylims..., 10), x in LinRange(basexlims..., _nbpoints)])
    baseprobgridys = reduce(hcat, [[x, y] for x in LinRange(basexlims..., 10), y in LinRange(baseylims..., _nbpoints)])
    baseprobgrid = hcat(baseprobgridxs, baseprobgridys)

    realprobgridys = reduce(hcat, [[x, y] for x in LinRange(realxlims..., 10), y in LinRange(realylims..., _nbpoints)])
    realprobgridxs = reduce(hcat, [[x, y] for y in LinRange(realylims..., 10), x in LinRange(realxlims..., _nbpoints)])
    realprobgrid = hcat(realprobgridxs, realprobgridys)

    _, _, basespace = forwardffjord(chain.rng, realprobgrid, chain.map_hyperparams)
    realspace = backwardffjord(chain.rng, baseprobgrid, chain.map_hyperparams)
    # basespace = ffjord_mdl(realprobgrid, ps, st)[1].z
    # realspace = DiffEqFlux.__backward_ffjord(ffjord_mdl, baseprobgrid, ps, st)

    fig = Figure(size=(900, 500));
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel="Real axis 1 -> Base axis 1", ylabel="Real axis 2 -> Base axis 2")
    scatter!(ax1, basespace[1, :], basespace[2, :], markersize=2, label=nothing); 
    scatter!(ax1, realprobgrid[1, :], realprobgrid[2, :], color=:grey, alpha=0.5, markersize=2, label=nothing);

    scatter!(ax1, Float64[], Float64[], color=:grey, markersize=15, label="Real/environmental space"); 
    scatter!(ax1, Float64[], Float64[], color=Cycled(1), markersize=15, label="Base space"); 
    ylims!(ax1, nothing, 7); axislegend(ax1, framecolor=:white); 
    _deco!(ax1)

    ax2 = CairoMakie.Axis(fig[1, 2], xlabel="Base axis 1 -> Real axis 1", ylabel="Base axis 2 -> Real axis 2")
    scatter!(ax2, realspace[1, :], realspace[2, :], markersize=2, label=nothing); 
    scatter!(ax2, baseprobgrid[1, :], baseprobgrid[2, :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    scatter!(ax2, Float64[], Float64[], color=:grey, markersize=15, label="Base space");
    scatter!(ax2, Float64[], Float64[], color=Cycled(1), markersize=15, label="Real/environmental space");
    ylims!(ax2, nothing, 7); axislegend(ax2, framecolor=:white);
    _deco!(ax2);

    return fig
end