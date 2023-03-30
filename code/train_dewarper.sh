#!/bin/bash

#SBATCH -J dewarper
#SBATCH -p gpu
#SBATCH -o logs/dewarper.txt
#SBATCH -e logs/dewarper.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lramesh@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 1
#SBATCH --time=2-00:00:00
#SBATCH --qos=long
#SBATCH --mem=64G

#Load any modules that your program needs
module load deeplearning
cd ../fastell4py
python setup.py install --user
cd ../code

#Run your program
srun python train_dewarper.py