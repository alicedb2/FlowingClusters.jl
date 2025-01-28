using FlowingClusters, SplitMaskStandardize, CairoMakie, Distributions, CSV, Random

include(joinpath(@__DIR__, "misc", "brt_predictions.jl"))
include(joinpath(@__DIR__, "misc", "bioclim_predictions.jl"))
include(joinpath(@__DIR__, "misc", "flowingclusters_predictions.jl"))

twomoons = reduce(hcat, [[a, b] for (a, b) in CSV.File("/Users/alice/Downloads/cluster_moons.csv")])

wpa_df = DataFrame(["BIO$i" for i in 1:19] .=> Vector{Float64}.(collect.(CSV.File("data/Sitta_whiteheadi/features.csv", header=false))))
insertcols!(wpa_df, 1, :sp => [first(x) for x in CSV.File("data/Sitta_whiteheadi/labels.csv", header=false)])
insertcols!.(Ref(wpa_df), [1, 1], [:lat, :lon] .=> Vector{Float64}.(eachrow(reduce(hcat, collect.(CSV.File("data/Sitta_whiteheadi/coordinates_latlon.csv", header=false))))))
wpa = SMSDataset(wpa_df, splits=[2, 1, 1])#, seed=4242)
training_dataset = wpa.training.presence(:sp).standardize(predictors...)(predictors...)

scatter(Tuple.(eachcol(wpa.presence(:sp)(:lon, :lat))));
scatter!(Tuple.(eachcol(wpa.absence(:sp)(:lon, :lat))));
current_figure()
scatter(Tuple.(eachcol(wpa.presence(:sp)(:BIO1, :BIO12))));
scatter!(Tuple.(eachcol(wpa.absence(:sp)(:BIO1, :BIO12))));
current_figure()

# "BIO1" => "Annual Mean Temperature",
# "BIO2" => "Mean Diurnal Range (Mean of monthly (max temp - min temp))",
# "BIO3" => "Isothermality (BIO2/BIO7) (×100)",
# "BIO4" => "Temperature Seasonality (standard deviation ×100)",
# "BIO5" => "Max Temperature of Warmest Month",
# "BIO6" => "Min Temperature of Coldest Month",
# "BIO7" => "Temperature Annual Range (BIO5-BIO6)",
# "BIO8" => "Mean Temperature of Wettest Quarter",
# "BIO9" => "Mean Temperature of Driest Quarter",
# "BIO10" => "Mean Temperature of Warmest Quarter",
# "BIO11" => "Mean Temperature of Coldest Quarter",
# "BIO12" => "Annual Precipitation",
# "BIO13" => "Precipitation of Wettest Month",
# "BIO14" => "Precipitation of Driest Month",
# "BIO15" => "Precipitation Seasonality (Coefficient of Variation)",
# "BIO16" => "Precipitation of Wettest Quarter",
# "BIO17" => "Precipitation of Driest Quarter",
# "BIO18" => "Precipitation of Warmest Quarter",
# "BIO19" => "Precipitation of Coldest Quarter",

dataset = SMSDataset("data/ebird_data/ebird_bioclim_landcover.csv", 
                subsample=1000, splits=[1, 1, 1], seed=99);
perfstat = :MCC
species = :sp1
predictors = (:BIO1, :BIO3, :BIO12)
training_dataset = dataset.training.presence(species).standardize(predictors...)(predictors...)
chain = FCChain(training_dataset, ffjord_nn=
dense_nn(length(predictors), 20, 20, length(predictors), act=tanhshrink), strategy=:1)

advance_chain!(chain, 1000,
    nb_ffjord_am=1, ffjord_am_temperature=1.0,
    nb_amwg=0, nb_gibbs=0, nb_splitmerge=0,
    attempt_map=false, sample_every=nothing, stop_criterion=:sample_ess)

advance_chain!(chain, 1000, nb_splitmerge=300, nb_ffjord_am=1)



# predictors = (:BIO1, :BIO12)
predictors = (:BIO1, :BIO3, :BIO12)
# predictors = (:BIO1, :BIO3, :BIO14, :BIO11, :BIO2, :BIO7)


allpresences = zeros(length(predictors), 0)
for i in 1:16
    allpresences = hcat(allpresences, eb.training.presence(Symbol("sp$i")).standardize(predictors...)(predictors...))
end



# chain = FCChain(train_presences, SetCluster)
# advance_chain!(chain, Inf, nb_splitmerge=150, stop_chain=:sample_ess)

function initbias(lower, upper)
    fun(rng, outdims, _) = rand(rng, Uniform(lower, upper), outdims, 1)
    return fun
end
# initbias(lims...) = (rng, outdims, _) -> rand(rng, Product.(Uniform(lw...)), outdims, 1)

# zero biases on the last layer appears to be a good idea
# softsign appears to be a good idea as well instead
# of tanh, the rational being that it behaves as
# 1/x for large x, which is a good property
# to capture signal far from the origin
nn32 = Chain(
 Dense(length(predictors), 32, softsign, init_bias=initbias(-4, 4), init_weight=kaiming_uniform(gain=0.1)),
 Dense(32, length(predictors), softsign, init_bias=zeros64, init_weight=kaiming_uniform(gain=0.1))
#  Dense(length(predictors), 32, softsign, init_bias=initbias(-4, 4), init_weight=orthogonal(Float64)),
#  Dense(32, length(predictors), softsign, init_bias=zeros64, init_weight=orthogonal(Float64))
)
nn48 = Chain(
 Dense(length(predictors), 48, softsign, init_bias=initbias(-4, 4), init_weight=orthogonal(Float64)),
 Dense(48, length(predictors), softsign, init_bias=zeros64, init_weight=orthogonal(Float64))
)

# Not good lol
nlrelu(T::Type, beta=1) = x -> log.(T(beta) .* max.(zero(T), x) .+ one(T))
logsign(T::Type, beta=1) = x -> sign(x) .* log.(T(beta) .* abs.(x) .+ one(T))

# chainnn = FCChain(training_presence_predictors, SetCluster, ffjord_nn=nn2d, seed=4041)
# chain32 = FCChain(training_presence_predictors, SetCluster, strategy=:1, ffjord_nn=nn32, seed=777)
chain48 = FCChain(training_presence_predictors, SetCluster, strategy=:1, ffjord_nn=nn48, seed=7)
advance_chain!(chain48, 5000,
       nb_ffjord_am=1, ffjord_am_temperature=1.0,
       nb_amwg=0, nb_gibbs=0, nb_splitmerge=0,
       attempt_map=false, sample_every=nothing, stop_criterion=:sample_ess)

plot(chainnn)

# Pre-training
advance_chain!(chainnn, 1000, 
nb_ffjord_am=2, nb_hyperparams=1,
nb_gibbs=0, nb_splitmerge=0, 
sample_every=nothing, stop_criterion=nothing)

advance_chain!(chainnn2, Inf, nb_splitmerge=200, stop_criterion=:sample_ess)