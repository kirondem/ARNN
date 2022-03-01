#! /bin/bash

#SBATCH --job-name="Train"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpagi.kironde@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output ./logs/job%J.output
#SBATCH --error ./logs/jo%J.err
#SBATCH --gres=gpu:1
#SBATCH --partition=normal


module load cuda/10.0

cd ..
python3 train.py --env=camber --time_steps=1000

python3 train.py --env=camber --time_steps=1000
