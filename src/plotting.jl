function plot!(ax, clusters::AbstractVector{<:AbstractCluster}; proj_or_dims=[1, 2], rev=false, nb_clusters=nothing, orig=false, plot_kw...)

    size(proj_or_dims, 1) == 2 || error("Can only support plotting 2 dimesions for now")

    clusters = project_clusters(sort(clusters, by=length, rev=!rev), proj_or_dims)

    if nb_clusters === nothing || nb_clusters < 0
        nb_clusters = length(clusters)
    end

    for (cluster, i) in zip(clusters, 1:nb_clusters)
        if !isempty(cluster)
            x = getindex.(clusters[i], 1)
            y = getindex.(clusters[i], 2)
            scatter!(ax, x, y, label="$(length(cluster))", color=Cycled(i), plot_kw...)
        end
    end
    axislegend(ax)

    return ax
end

function plot(clusters::AbstractVector{<:AbstractCluster}; proj_or_dims=[1, 2], rev=false, nb_clusters=nothing, orig=false, plot_kw...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    plot!(ax, clusters, proj_or_dims=proj_or_dims, rev=rev, nb_clusters=nb_clusters, orig=orig, plot_kw...)
    return fig
end

# function plot(chain::FCChain; proj_or_dims::Vector{Int}=[1, 2], burn=0, rev=false, nb_clusters=nothing, plot_kw...)

#     @assert length(dims) == 2 "We can only plot in 2 dimensions for now, dims must be a vector of length 2."

#     d = dimension(chain.hyperparams)

#     proj = dims_to_proj(dims, d)

#     return plot(chain, proj, burn=burn, rev=rev, nb_clusters=nb_clusters; plot_kw...)
# end

# function plot(chain::FCChain, proj::Matrix{Float64}; burn=0, rev=false, nb_clusters=nothing)

#     @assert size(proj, 1) == 2 "The projection matrix should have 2 rows"

#     N = length(chain)

#     if burn >= N
#         @error("Can't burn the whole chain, burn must be smaller than $N")
#     end

#     map_idx = chain.map_idx - burn

#     map_marginals = project_clusters(realspace_clusters(Cluster, chain.map_clusters, chain.map_base2original), proj)
#     current_marginals = project_clusters(realspace_clusters(Cluster, chain.clusters, chain.base2original), proj)

#     basemap_marginals = project_clusters(chain.map_clusters, proj)
#     basecurrent_marginals = project_clusters(chain.clusters, proj)

#     function deco!(axis)
#         hidespines!(axis, :t, :r)
#         hidedecorations!(axis, grid=true, minorgrid=true, ticks=false, label=false, ticklabels=false)
#         return axis
#     end

#     if hasnn(chain.hyperparams)
#         ysize = 2400
#     else
#         ysize = 1500
#     end

#     fig = Figure(size=(1200, ysize))

#     p_map_axis = Axis(fig[1:2, 1], title="MAP state ($(length(chain.map_clusters)) clusters)")
#     deco!(p_map_axis)
#     plot!(p_map_axis, map_marginals; rev=rev, nb_clusters=nb_clusters)
#     axislegend(p_map_axis)

#     p_current_axis = Axis(fig[1:2, 2], title="Current state ($(length(chain.clusters)) clusters)")
#     deco!(p_current_axis)
#     plot!(p_current_axis, current_marginals; rev=rev, nb_clusters=nb_clusters)
#     # axislegend(p_current_axis, framecolor=:white)

#     offset = 0
#     if hasnn(chain.hyperparams)
#         offset = 2
#         p_basemap_axis = Axis(fig[3:4, 1], title="MAP base state ($(length(chain.map_clusters)) clusters)")
#         deco!(p_basemap_axis)
#         plot!(p_basemap_axis, basemap_marginals; rev=rev, nb_clusters=nb_clusters)
#         # axislegend(p_basemap_axis, framecolor=:white)

#         p_basecurrent_axis = Axis(fig[3:4, 2], title="Current base state ($(length(chain.clusters)) clusters)")
#         deco!(p_basecurrent_axis)
#         plot!(p_basecurrent_axis, basecurrent_marginals; rev=rev, nb_clusters=nb_clusters)
#         # axislegend(p_basecurrent_axis, framecolor=:white)
#     end

#     lpc = logprob_chain(chain, burn)
#     logprob_axis = Axis(fig[offset + 3, 1], title="log probability", aspect=3)
#     deco!(logprob_axis)
#     lines!(logprob_axis, burn+1:N, lpc, label=nothing)
#     hlines!(logprob_axis, [maximum(chain.logprob_chain)], label=nothing, color=:black)
#     hlines!(logprob_axis, [chain.map_logprob], label=nothing, color=:green)
#     if map_idx > 0
#         vlines!(logprob_axis, [map_idx], label=nothing, color=:green)
#     end

#     nbc = nbclusters_chain(chain, burn)
#     nbc_axis = Axis(fig[offset + 3, 2], title="#cluster", aspect=3)
#     deco!(nbc_axis)
#     lines!(nbc_axis, burn+1:N, nbc, label=nothing)
#     if map_idx > 0
#         vlines!(nbc_axis, [map_idx], label=nothing, color=:black)
#     end

#     lcc = largestcluster_chain(chain, burn)
#     lcc_axis = Axis(fig[offset + 4, 1], title="Largest cluster", aspect=3)
#     deco!(lcc_axis)
#     lines!(lcc_axis, burn+1:N, lcc, label=nothing)
#     if map_idx > 0
#         vlines!(lcc_axis, [map_idx], label=nothing, color=:black)
#     end

#     ac = alpha_chain(chain, burn)
#     alpha_axis = Axis(fig[offset + 4, 2], title="α", aspect=3)
#     deco!(alpha_axis)
#     lines!(alpha_axis, burn+1:N, ac, label=nothing)
#     if map_idx > 0
#         vlines!(alpha_axis, [map_idx], label=nothing, color=:black)
#     end

#     muc = mu_chain(Matrix, chain, burn)
#     mu_axis = Axis(fig[offset + 5, 1], title="μ", aspect=3)
#     deco!(mu_axis)
#     for mucomponent in eachrow(muc)
#         lines!(mu_axis, burn+1:N, mucomponent, label=nothing)
#     end
#     if map_idx > 0
#         vlines!(mu_axis, [map_idx], label=nothing, color=:black)
#     end

#     lc = lambda_chain(chain, burn)
#     lambda_axis = Axis(fig[offset + 5, 2], title="λ", aspect=3)
#     deco!(lambda_axis)
#     lines!(lambda_axis, burn+1:N, lc, label=nothing)
#     if map_idx > 0
#         vlines!(lambda_axis, [map_idx], label=nothing, color=:black)
#     end

#     pc = psi_chain(Matrix, chain, burn)
#     psi_axis = Axis(fig[offset + 6, 1], title="Ψ", aspect=3)
#     deco!(psi_axis)
#     for psicomponent in eachrow(pc)
#         lines!(psi_axis, burn+1:N, psicomponent, label=nothing)
#     end
#     if map_idx > 0
#         vlines!(psi_axis, [map_idx], label=nothing, color=:black)
#     end

#     nc = nu_chain(chain, burn)
#     nu_axis = Axis(fig[offset + 6, 2], title="ν", aspect=3)
#     deco!(nu_axis)
#     lines!(nu_axis, burn+1:N, nc, label=nothing)
#     if map_idx > 0
#         vlines!(nu_axis, [map_idx], label=nothing, color=:black)
#     end

#     if hasnn(chain.hyperparams)
#         nnc = nn_chain(Matrix, chain, burn)
#         nn_axis = Axis(fig[9, 1:2], title="FFJORD neural network prior")
#         deco!(nn_axis)
#         for p in eachrow(nnc)
#             lines!(nn_axis, burn+1:N, collect(p), label=nothing, linewidth=1, alpha=0.5)
#         end
#         if map_idx > 0
#             vlines!(nn_axis, [map_idx], label=nothing, color=:black)
#         end

#         nnscalec = nn_alpha_chain(chain, burn)
#         nnscale_axis = Axis(fig[10, 1], title="FFJORD hyperprior log α")
#         deco!(nnscale_axis)
#         lines!(nnscale_axis, burn+1:N, log.(nnscalec), label=nothing)
#         if map_idx > 0
#             vlines!(nnscale_axis, [map_idx], label=nothing, color=:black)
#         end

#         nnscalec = nn_scale_chain(chain, burn)
#         nnscale_axis = Axis(fig[10, 2], title="FFJORD hyperprior scale")
#         deco!(nnscale_axis)
#         lines!(nnscale_axis, burn+1:N, nnscalec, label=nothing)
#         if map_idx > 0
#             vlines!(nnscale_axis, [map_idx], label=nothing, color=:black)
#         end
#     end

#     return fig
# end