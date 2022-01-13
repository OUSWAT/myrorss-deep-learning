#!/usr/bin/bash

#SBATCH --job-name=ExtractionTest
#SBATCH -p swat_plus
#SBATCH -n 1
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH -t 01:00:00
source ~fagg/pythonenv/tensorflow/bin/activate
python extract.py 


