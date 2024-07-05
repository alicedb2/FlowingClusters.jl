function deco!(axis)
    hidespines!(axis, :t, :r, :l, :b)
    hidedecorations!(axis, grid=true, minorgrid=true, ticks=true, label=true, ticklabels=true)
    return axis
end

nn, ps, st = (presence_chain.hyperparams.nn,
              presence_chain.map_hyperparams.nn_params,
              presence_chain.map_hyperparams.nn_state)

ffjord_mdl = FFJORD(nn, (0.0f0, 1.0f0), (dimension(presence_chain.hyperparams),), Tsit5(), ad=AutoForwardDiff())

basexlims = minimum(getindex.(keys(presence_chain.map_base2original), dims[1])), maximum(getindex.(keys(presence_chain.map_base2original), dims[1]))
baseylims = minimum(getindex.(keys(presence_chain.map_base2original), dims[2])), maximum(getindex.(keys(presence_chain.map_base2original), dims[2]))
basezlims = minimum(getindex.(keys(presence_chain.map_base2original), dims[3])), maximum(getindex.(keys(presence_chain.map_base2original), dims[3]))
realxlims = minimum(getindex.(values(presence_chain.map_base2original), dims[1])), maximum(getindex.(values(presence_chain.map_base2original), dims[1]))
realylims = minimum(getindex.(values(presence_chain.map_base2original), dims[2])), maximum(getindex.(values(presence_chain.map_base2original), dims[2]))
realzlims = minimum(getindex.(values(presence_chain.map_base2original), dims[3])), maximum(getindex.(values(presence_chain.map_base2original), dims[3]))

_nbpoints = 300
_nblines = 10

baseprobgridhorizontal = reduce(hcat, [[x, y] for y in LinRange(baseylims..., _nblines), x in LinRange(basexlims..., _nbpoints)])
baseprobgridvertical = reduce(hcat, [[x, y] for x in LinRange(basexlims..., _nblines), y in LinRange(baseylims..., _nbpoints)])
realprobgridhorizontal = reduce(hcat, [[x, y] for y in LinRange(realylims..., _nblines), x in LinRange(realxlims..., _nbpoints)])
realprobgridvertical = reduce(hcat, [[x, y] for x in LinRange(realxlims..., _nblines), y in LinRange(realylims..., _nbpoints)])

dims = [2, 3, 1]

for z in LinRange(basezlims..., _nblines)

    baseprobgrid = fill(0.0, 3, 2 * _nblines * _nbpoints)
    baseprobgrid[[dims[1], dims[2]], :] .= hcat(baseprobgridhorizontal, baseprobgridvertical)
    baseprobgrid[dims[3], :] .= fill(z, 2 * _nblines * _nbpoints)
    realprobgrid = fill(0.0, 3, 2 * _nblines * _nbpoints)
    realprobgrid[[dims[1], dims[2]], :] .= hcat(realprobgridhorizontal, realprobgridvertical)
    realprobgrid[dims[3], :] .= fill(z, 2 * _nblines * _nbpoints)
    baseprobgrid
    realprobgrid

    basespace = ffjord_mdl(realprobgrid, ps, st)[1].z
    realspace = __backward_ffjord(ffjord_mdl, baseprobgrid, ps, st)

    fig = Figure(size=(800, 400));
    ax1 = Axis(fig[1, 1], xlabel="Real axis $(dims[1]) -> Base axis $(dims[1])", ylabel="Real axis $(dims[2]) -> Base axis $(dims[2])")
    bases1 = scatter!(ax1, basespace[dims[1], :], basespace[dims[2], :], markersize=2, label=nothing);
    reals1 = scatter!(ax1, realprobgrid[dims[1], :], realprobgrid[dims[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    leg1 = Legend(fig[2, 1], [bases1, reals1], ["Base space", "Real/environmental space"], framevisible=false, markersize=30)
    for i in 1:2
        leg1.entrygroups[][1][2][i].elements[1].attributes[:markersize] = Observable(10)
    end
    notify(leg1.entrygroups)
    xlims!(ax1, [-3, 4]); ylims!(ax1, [-3, 3])

    ax2 = Axis(fig[1, 2], xlabel="Base axis $(dims[2]) -> Real axis $(dims[2])", ylabel="Base axis $(dims[2]) -> Real axis $(dims[2])")
    reals2 = scatter!(ax2, realspace[dims[1], :], realspace[dims[2], :], markersize=2, label=nothing);
    bases2 = scatter!(ax2, baseprobgrid[dims[1], :], baseprobgrid[dims[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    leg2 = Legend(fig[2, 2], [reals2, bases2], ["Real/environmental space", "Base space"], framevisible=false, markersize=30)
    for i in 1:2
        leg2.entrygroups[][1][2][i].elements[1].attributes[:markersize] = Observable(10)
    end
    notify(leg2.entrygroups)
    xlims!(ax2, [-3, 4]); ylims!(ax2, [-3, 3])

    deco!(ax1); deco!(ax2);
    rowsize!(fig.layout, 2, 30)
    colsize!(fig.layout, 1, 300)
    colsize!(fig.layout, 2, 300)

    display(fig)
end

#############################
#############################
#############################

baseprobgrid = fill(0.0, 3, 2 * _nblines * _nbpoints)
baseprobgrid[[dims[1], dims[2]], :] .= hcat(baseprobgridhorizontal, baseprobgridvertical)
realprobgrid = fill(0.0, 3, 2 * _nblines * _nbpoints)
realprobgrid[[dims[1], dims[2]], :] .= hcat(realprobgridhorizontal, realprobgridvertical)

fig = Figure(size=(650, 325 * _nblines));

for (i, z) in enumerate(LinRange(basezlims..., _nblines))

    baseprobgrid[dims[3], :] .= fill(z, 2 * _nblines * _nbpoints)
    realprobgrid[dims[3], :] .= fill(z, 2 * _nblines * _nbpoints)

    basespace = ffjord_mdl(realprobgrid, ps, st)[1].z
    realspace = __backward_ffjord(ffjord_mdl, baseprobgrid, ps, st)

    ax1 = Axis(fig[i, 1], xlabel="Real axis $(dims[1]) -> Base axis $(dims[1])", 
        ylabel="Real axis $(dims[2]) -> Base axis $(dims[2])"
    )
    bases1 = scatter!(ax1, basespace[dims[1], :], basespace[dims[2], :], markersize=2, label=nothing);
    reals1 = scatter!(ax1, realprobgrid[dims[1], :], realprobgrid[dims[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    xlims!(ax1, [-3, 4.5]); ylims!(ax1, [-3, 3])

    ax2 = Axis(fig[i, 2], 
        xlabel="Base axis $(dims[2]) -> Real axis $(dims[2])", ylabel="Base axis $(dims[2]) -> Real axis $(dims[2])"
    )
    reals2 = scatter!(ax2, realspace[dims[1], :], realspace[dims[2], :], markersize=2, label=nothing);
    bases2 = scatter!(ax2, baseprobgrid[dims[1], :], baseprobgrid[dims[2], :], color=:grey, alpha=0.5, markersize=2, label=nothing);
    xlims!(ax2, [-3, 4.5]); ylims!(ax2, [-3, 3])
    deco!(ax1); deco!(ax2);

    rowsize!(fig.layout, i, 300)
end
colsize!(fig.layout, 1, 300)
colsize!(fig.layout, 2, 300)

ax = Axis(fig[_nblines + 1, 1])
text!(ax, [(0, 0)], 
    text=["Deformation from real to base space"],
    align=(:center, :baseline),
    font=:bold)
deco!(ax)
ax = Axis(fig[_nblines + 1, 2])
text!(ax, [(0, 0)], 
    text=["Deformation from base to real space"],
    align=(:center, :baseline),
    font=:bold)
deco!(ax)
display(fig)
