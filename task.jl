using JLD2, InlineStrings, DataFrames, FileIO
using MultivariateNormalCRP
using Random

dataset = load("data/aedes_albopictus_dataset.jld2")

Random.seed!()

chain = initiate_chain(dataset["chain_input"])

advance_chain!(chain, nb_steps=200000, 
nb_splitmerge=400, splitmerge_t=3, nb_gibbs=1, 
checkpoint_every=2000, checkpoint_prefix="tiger7k")

