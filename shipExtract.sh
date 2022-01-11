#!/usr/bin/bash

#SBATCH --job-name=ExtractionTest
#SBATCH -p swat_plus
#SBATCH -n 1
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/scripts
#SBATCH -t 01:00:00


echo "Running Extract: $STARTDATE"
echo "End-date: $ENDDATE"
python Extract.py $STARTDATE $ENDDATE

