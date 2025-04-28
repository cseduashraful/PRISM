#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH --mem=32G
#SBATCH -G 1  # Number of GPUs

#SBATCH --constraint=a100
#SBATCH -t 1-12:00:00  # Job time limit

#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -q long


module load conda/latest
conda activate pyg

python main.py --bs 8192 --data tgbl-review