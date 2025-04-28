#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH --mem=64G
#SBATCH -G 1  # Number of GPUs

#SBATCH --constraint=a100
#SBATCH -t 1-12:00:00  # Job time limit

#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -q long


module load conda/latest
conda activate pyg

# python main.py --data tgbl-review --bs 65536 
#33331144 256 33331147 512 33331148 1024 33331150 2048 33331181 4096 33331185 8192 33331189 16384 33331245 32768 33331250 65536

# python train.py --data tgbl-review --bs 65536
#33331283 256 33331303 512 33331311 1024 33331312 2048 33331314 4096 33331319 8192 33331321 16384 33331324 32768 33331325 65536