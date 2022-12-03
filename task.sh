#!/bin/bash

#SBATCH --time 96:00:00
#SBATCH --array=1-16
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=12G
#SBATCH --account=ctb-tpoisot

module load StdEnv/2020
module load julia/1.8.1

cd /home/alicedb/MultivariateNormalCRP
julia --project=. task.jl