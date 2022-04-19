#!/usr/bin/bash

#SBATCH --job-name=multijob
#SBATCH -p swat_plus
#SBATCH -n 2
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH --output=batch_output_%J.out
#SBATCH --error=batch_error_%J.out
#SBATCH -t 29:00:00
#SBATCH --array=0-24

module load Python/3.9.5-GCCcore-10.3.0
source ~/workenv/bin/activate
python run_exp_with_jc.py -epochs 60 -patience 20 -lambda_regularization 0.01 -batch_size 500


