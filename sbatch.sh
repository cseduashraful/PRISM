#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH --mem=64G
#SBATCH -G 1  # Number of GPUs

#SBATCH --constraint=2080ti
#SBATCH -t 3-00:00:00  # Job time limit

#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -q long


module load conda/latest
conda activate pyg

python main.py --bs 8192 -num_epoch 200 --patience 200 --num_run 1