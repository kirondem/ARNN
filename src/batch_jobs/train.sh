#!/bin/bash

#SBATCH -D /users/adbg238                    # Working directory
#SBATCH --job-name train_ANN                 # Job name
#SBATCH --partition=nodes                    # Select the correct partition. 
#SBATCH --nodes=1                            # Run on 1 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=8                  # Use 8 cores, most of the procesing happens on the GPU
#SBATCH --time=24:00:00                      # Expected ammount of time to Rrun Time limit hrs:min:sec
#SBATCH --mem=1GB                            # Expected memory usage (0 means use all available memory)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mpagi.kironde@city.ac.uk
#SBATCH --output=/users/adbg238/logs/train_ANN_%j.txt        # Standard output and error log [%j is replaced with the jobid]
#SBATCH --error=/users/adbg238/logs/train_ANN_%j.err

#Enable modules command
source /opt/flight/etc/setup.sh
flight env activate gridware

#Remove any unwanted modules
module purge

#Modules required

#Run your script.
cd /users/adbg238/Work/PhD/src

python3 train.py
