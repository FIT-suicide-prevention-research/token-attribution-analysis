#!/bin/bash
source ~/.bashrc
source ~/mambaforge/etc/profile.d/mamba.sh
eval "$(conda shell.bash hook)"
conda activate dl
"$@"
