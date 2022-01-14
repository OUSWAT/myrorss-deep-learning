#!/usr/bin/bash

#SBATCH --job-name=ExtractCropConv
#SBATCH -p swat_plus
#SBATCH -n 1
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH -t 01:00:00
module load Python/3.6.6-foss-2018b
source ~/testflow/bin/activate
python extract.py 


