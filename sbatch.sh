#!/bin/bash
#SBATCH -c 8  # Number of Cores per Task
#SBATCH -p gpu  # Partition
#SBATCH --mem=128G
#SBATCH -G 1  # Number of GPUs

#SBATCH --constraint=a100
#SBATCH -t 5-00:00:00  # Job time limit

#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -q long


module load conda/latest
conda activate pyg

# python apan.py --bs 64
# python main.py --bs 2048 --lr 0.001 --debug False
# python tncn.py --bs 64 --lr 0.0001 #--debug False

# python main.py --data tgbl-review --bs 65536 
#33331144 256 33331147 512 33331148 1024 33331150 2048 33331181 4096 33331185 8192 33331189 16384 33331245 32768 33331250 65536

# python train.py --data tgbl-review --bs 65536
#33331283 256 33331303 512 33331311 1024 33331312 2048 33331314 4096 33331319 8192 33331321 16384 33331324 32768 33331325 65536


# python main.py --data reddit --bs 2048
#33396176 128 33396179 256 33409975  33421660 2048 33412770 4096  33413012 8192 ??? 16384 33413255 32768

# python tgn.py --data reddit --bs 16384
#33396405 128 33396503 256 33396506 512 33396507 1024 33396509 2048 33396515 4096 33396528 8192 33407329 16384 33396555 32768


# python main.py --data lastfm --bs 32768
#33452671 256 33452692 512 33453158 1024 33453162 2048 33453164 4096 33453634 8192 33453637 16384 33453884 32768
# python tgn.py --data lastfm --bs 32768
#33454161 256 33454163 512 33454588 1024 33454591 2048 33454935 4096 33455387 8192 33455792 16384 33456099 32768



#7-1-25
# TGN-wiki --- re-run with patience 500
# python main.py --data tgbl-wiki --bs 64 --num_epoch 500 --mxtt 48 --lr 0.001 --num_run 1 --patience 500 --deliver_to self
# python tgn.py --bs 64 --lr 0.001 --num_epoch 500 --num_run 1 --patience 500


#TGN lastfm
# python main.py --data lastfm --bs 64 --num_epoch 500 --mxtt 48 --lr 0.001 --num_run 1 --patience 500 --deliver_to self
# python tgn.py --data lastfm --bs 64 --lr 0.001 --num_epoch 500 --num_run 1 --patience 500

#TGN reddit
# python main.py --data reddit --bs 64 --num_epoch 500 --mxtt 48 --lr 0.001 --num_run 1 --patience 500 --deliver_to self
# python tgn.py --data reddit --bs 64 --lr 0.001 --num_epoch 500 --num_run 1 --patience 500


#7-5-2025
# python main.py --decoder NCN --data tgbl-wiki --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001
# [64-16384]
# python tncn.py --data tgbl-wiki --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.0001

#7-6-2026
# python main.py --decoder NCN --data tgbl-flight --bs 8192 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001


#7-9-25
# python main.py --decoder NCN --data reddit --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001 --mxtt 72 --mxet 96
# python tncn.py --data reddit --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001

#7-22-2026
# python main.py --data tgbl-wiki --bs 16384 --lr 0.001 --deliver_to neighbor --num_epoch 200 --patience 200 --num_run 1
# python main.py --decoder NCN --data lastfm --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001 --mxtt 72 --mxet 96
# python main.py --decoder NCN --data lastfm --bs 64 --num_epoch 200 --patience 200 --num_run 1 --lr 0.0001 --mxtt 72 --mxet 96 --val_neg 5
# python tncn.py --data lastfm --bs 8192 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001 --val_neg 5

#8-17
# python apan.py --data lastfm --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001 --val_neg 5
# python apan.py --data reddit --bs 16384 --num_epoch 200 --patience 200 --num_run 1 --lr 0.001
