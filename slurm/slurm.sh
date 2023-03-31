#!/bin/bash

# Manpage: https://slurm.schedmd.com/sbatch.html

##################################
######### Configuration ##########
##################################
# Configure the job name
#SBATCH --job-name IRI_single_gpu

##################################
####### Resources Request ########
##################################

# Use GPU partition (gpu1 and gpu2) or other partition (e.g.: short)
# Find more usable partitions with 'sinfo -a'
#SBATCH --partition=gpu1

# Configure the number of nodes (in partition above)
# Do NOT use --ntasks-per-node > 1 if you are not using MPI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Configure the number of GPUs
#SBATCH --gres=gpu:4
# Configure the number of CPUs
#SBATCH --cpus-per-task=24

# Pipe stdout and stderr to files <jobname>.<jobid>.<nodeid>
#SBATCH --output=%x.%J.%N.out

# (optional) list all gpus (check GPU available)
# nvidia-smi
echo `date`

# load conda environment
. ./env.sh

start=`date +%s`

# load main script
./train.sh

end=`date +%s`

runtime=$((end-start))
echo "Runtime: $runtime"