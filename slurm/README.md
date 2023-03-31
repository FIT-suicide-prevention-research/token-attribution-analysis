# SLURM scripts for running on the cluster

## Preparing the environment (`env.sh`)
- Install `mambaforge` or `miniconda` on `aipanther` login node ([Guide](https://github.com/conda-forge/miniforge#install))
- If you install `miniconda`, or install `conda` in different paths, you need to change the path in `env.sh`
- Create a new environment using `mamba create -n dl` or any other name (if you do change the activated environment name in `env.sh`)


## Preparing SLURM scripts (`slurm.sh`)
In `slurm.sh`:
- Change the `--job-name` to preferred job name
- Change `--partition` to available partition (e.g. `--partition=gpu1`)
- Read the comments in the script for more information

## Preparing training script (`train.sh`)
Create a new `train.sh` script for your training script. You can use current `train.sh` as an example. All commands in `train.sh` will be executed on the node similarly to current login node, so I highly suggest to test run the script on the login node first.

## Submitting jobs
- `sbatch slurm.sh` to submit the job

## Optional: Resource checker