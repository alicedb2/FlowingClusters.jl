using JLD2, InlineStrings, DataFrames, FileIO
using MultivariateNormalCRP
using Random

dataset = load("../data/aedes_albopictus_dataset.jld2")

Random.seed!()

chain = initiate_chain(dataset["chain_input"])

advance_chain!(chain, nb_steps=2000, 
nb_splitmerge=300, nb_gibbs=1, fullseq_prob=0.0,
checkpoint_every=-1, checkpoint_prefix="tiger7k",
pretty_progress=false)

burn!(chain, 2000)

advance_chain!(chain, nb_steps=20000, 
nb_splitmerge=300, nb_gibbs=1, fullseq_prob=0.0,
checkpoint_every=5000, checkpoint_prefix="tiger7k",
pretty_progress=false)
