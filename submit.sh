#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 4

# Request 4 GPU
#$ -l gpus=2

# Request at least capability 3.5/6.0/7.0
#$ -l gpu_c=7.0

#specify a project
#$ -P aclab

#merge the error and output
# -j y

#send email at the end
#$ -m e

# Set maximum time to 1 days
#$ -l h_rt=24:00:00



module load python3
module load pytorch
module load tensorflow

source env/bin/activate

python main.py
