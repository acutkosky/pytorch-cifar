#!/bin/bash -l

# Request 4 CPUs
#$ -pe omp 4

# Request 4 GPU
#$ -l gpus=1

# Request at least capability 3.5/6.0/7.0
#$ -l gpu_c=7.0

#specify a project
#$ -P aclab

#merge the error and output
#$ -j y

#send email at the end
#$ -m b

# Set maximum time to 12 days
#$# -l h_rt=12:00:00



module load python3
module load pytorch
module load tensorflow

source env/bin/activate
cmd="python main.py --arch=$1 --scale=$2 --stagewise=$3 --wd=$4 --patience=$5 --threshold=$6 --layer_count=$7 --retrain=$8 --tag=$9 --retrain_addition=${10} --lr ${11} --residual ${12}"
echo $cmd
$cmd
