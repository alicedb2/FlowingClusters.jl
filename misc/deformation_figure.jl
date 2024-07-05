function deco!(axis)
    hidespines!(axis, :t, :r)
    hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
    return axis
end

nn, ps, st = (presence_chain.hyperparams.nn, 
              presence_chain.map_hyperparams.nn_params, 
              presence_chain.map_hyperparams.nn_state)

ffjord_mdl = FFJORD(nn, (0.0f0, 1.0f0), (2,), Tsit5(), ad=AutoForwardDiff())

basexlims = minimum(getindex.(keys(presence_chain.map_base2original), 1)), maximum(getindex.(keys(presence_chain.map_base2original), 1))
baseylims = minimum(getindex.(keys(presence_chain.map_base2original), 2)), maximum(getindex.(keys(presence_chain.map_base2original), 2))
realxlims = minimum(getindex.(values(presence_chain.map_base2original), 1)), maximum(getindex.(values(presence_chain.map_base2original), 1))
realylims = minimum(getindex.(values(presence_chain.map_base2original), 2)), maximum(getindex.(values(presence_chain.map_base2original), 2))

_nbpoints = 300
    
baseprobgridys = reduce(hcat, [[x, y] for x in LinRange(basexlims..., 10), y in LinRange(baseylims..., _nbpoints)])
baseprobgridxs = reduce(hcat, [[x, y] for y in LinRange(baseylims..., 10), x in LinRange(basexlims..., _nbpoints)])
baseprobgrid = hcat(baseprobgridys, baseprobgridxs)

realprobgridys = reduce(hcat, [[x, y] for x in LinRange(realxlims..., 10), y in LinRange(realylims..., _nbpoints)])
realprobgridxs = reduce(hcat, [[x, y] for y in LinRange(realylims..., 10), x in LinRange(realxlims..., _nbpoints)])
realprobgrid = hcat(realprobgridys, realprobgridxs)

basespace = ffjord_mdl(realprobgrid, ps, st)[1].z
realspace = __backward_ffjord(ffjord_mdl, baseprobgrid, ps, st)

fig = Figure(size=(900, 500));
ax1 = Axis(fig[1, 1], xlabel="Real axis 1 -> Base axis 1", ylabel="Real axis 2 -> Base axis 2")
scatter!(ax1, basespace[1, :], basespace[2, :], markersize=2, label=nothing); 
scatter!(ax1, realprobgrid[1, :], realprobgrid[2, :], color=:grey, alpha=0.5, markersize=2, label=nothing);

scatter!(ax1, Float64[], Float64[], color=:grey, markersize=15, label="Real/environmental space"); 
scatter!(ax1, Float64[], Float64[], color=Cycled(1), markersize=15, label="Base space"); 
ylims!(ax1, nothing, 7); axislegend(ax1, framecolor=:white); 

ax2 = Axis(fig[1, 2], xlabel="Base axis 1 -> Real axis 1", ylabel="Base axis 2 -> Real axis 2")
scatter!(ax2, realspace[1, :], realspace[2, :], markersize=2, label=nothing); 
scatter!(ax2, baseprobgrid[1, :], baseprobgrid[2, :], color=:grey, alpha=0.5, markersize=2, label=nothing);
scatter!(ax2, Float64[], Float64[], color=:grey, markersize=15, label="Base space");
scatter!(ax2, Float64[], Float64[], color=Cycled(1), markersize=15, label="Real/environmental space");
ylims!(ax2, nothing, 7); axislegend(ax2, framecolor=:white);
deco!(ax1); deco!(ax2);

display(fig)