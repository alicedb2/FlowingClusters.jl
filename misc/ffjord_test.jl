using FlowingClusters, DiffEqFlux, SplitMaskStandardize, CairoMakie, Distributions


wpa = DataFrame(["BIO$i" for i in 1:19] .=> Vector{Float64}.(collect.(CSV.File("data/Sitta_whiteheadi/features.csv", header=false))))
insertcols!(wpa, 1, :sp => [first(x) for x in CSV.File("data/Sitta_whiteheadi/labels.csv", header=false)])
insertcols!.(Ref(wpa), [1, 1], [:lat, :lon] .=> Vector{Float64}.(eachrow(reduce(hcat, collect.(CSV.File("data/Sitta_whiteheadi/coordinates_latlon.csv", header=false))))))
wpa = SMSDataset(wpa, splits=[1, 1, 1], seed=4242)
scatter(Tuple.(eachcol(wpa.presence(:sp)(:lon, :lat))));
scatter!(Tuple.(eachcol(wpa.absence(:sp)(:lon, :lat))));
current_figure()
scatter(Tuple.(eachcol(wpa.presence(:sp)(:BIO1, :BIO12))));
scatter!(Tuple.(eachcol(wpa.absence(:sp)(:BIO1, :BIO12))));
current_figure()

eb = SMSDataset("data/ebird_data/ebird_bioclim_landcover.csv", 
                subsample=3000, splits=[1, 1, 1], seed=4242);

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

species = :sp2
predictors = (:BIO1, :BIO12)
# predictors = (:BIO1, :BIO3, :BIO12)
# predictors = (:BIO1, :BIO3, :BIO14, :BIO11, :BIO2, :BIO7)
training_presence_predictors = eb.training.presence(species).standardize(predictors...)(predictors...)


# chain = FCChain(train_presences, SetCluster)
# advance_chain!(chain, Inf, nb_splitmerge=150, stop_chain=:sample_ess)

initbias(lower, upper) = (rng, outdims, _) -> rand(rng, Uniform(lower, upper), outdims, 1)
# initbias(lims...) = (rng, outdims, _) -> rand(rng, Product.(Uniform(lw...)), outdims, 1)

g = 0.1
nn2d = Chain(
 Dense(length(predictors), 32, softsign, init_bias=initbias(-4, 4), init_weight=kaiming_uniform(Float64)),
 Dense(32, length(predictors), softsign, init_bias=zeros64, init_weight=kaiming_uniform(Float64))
)
           
# chainnn = FCChain(training_presence_predictors, SetCluster, ffjord_nn=nn2d, seed=4041)
chainnn = FCChain(training_presence_predictors, SetCluster, strategy=:2, ffjord_nn=nn2d, seed=4042)
plot(chainnn)

# Pre-training
advance_chain!(chainnn, 1000, 
nb_ffjord_am=2, nb_hyperparams=1,
nb_gibbs=0, nb_splitmerge=0, 
sample_every=nothing, stop_criterion=nothing)

advance_chain!(chainnn2, Inf, nb_splitmerge=200, stop_criterion=:sample_ess)