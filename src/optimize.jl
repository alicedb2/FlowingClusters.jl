# function optimize_hyperparams(
#     clusters::AbstractVector{<:AbstractCluster},
#     hyperparams0::AbstractFCHyperparams;
#     verbose=false
#     )

#     objfun(x) = -logprobgenerative(clusters, x)

#     x0 = unpack(hyperparams0)

#     if verbose
#         function callback(x)
#             print(" * Iter $(x.iteration),   objfun $(-round(x.value, digits=2)),   g_norm $(round(x.g_norm, digits=8))\r")
#             return false
#         end
#     else
#         callback =  nothing
#     end
#     opt_options = Options(iterations=50000,
#                           x_tol=1e-8,
#                           f_tol=1e-6,
#                           g_tol=2e-2,
#                           callback=callback)

#     optres = optimize(objfun, x0, NelderMead(), opt_options)

#     if verbose
#         println()
#     end

#     opt_hp = pack(minimizer(optres))

#     return MNCRPHyperparams(opt_hp..., deepcopy(hyperparams0.diagnostics), hyperparams0.nn, hyperparams0.nn_params, hyperparams0.nn_state)

# end

# # function optimize_hyperparams!(clusters::Vector{Cluster}, hyperparams::MNCRPHyperparams; jacobian=false, verbose=false)

# #     opt_res = optimize_hyperparams(clusters, hyperparams, jacobian=jacobian, verbose=verbose)

# #     hyperparams.alpha = opt_res.alpha
# #     hyperparams.mu = opt_res.mu
# #     hyperparams.lambda = opt_res.lambda
# #     hyperparams.flatL = opt_res.flatL
# #     hyperparams.L = opt_res.L
# #     hyperparams.psi = opt_res.psi
# #     hyperparams.nu = opt_res.nu

# #     return hyperparams

# # end
