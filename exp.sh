#!/usr/bin/bash

#SBATCH --job-name=vanilaU
#SBATCH -p swat_plus
#SBATCH -n 2
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH --output=unet_output_%J.out
#SBATCH --error=unet_error_%J.out
#SBATCH -t 04:00:00

module load Python/3.9.5-GCCcore-10.3.0
source ~/testflow/bin/activate
python run_exp.py 


