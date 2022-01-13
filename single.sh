#!/usr/bin/bash

#SBATCH --job-name=ExtractCropConv
#SBATCH -p swat_plus
#SBATCH -n 1
#SBATCH --output=/condo/swatwork/mcmontalbano/MYRORRS/myrorss-deep-learning
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH -t 01:00:00
source tensorflow/bin/activate
python extract.py 


