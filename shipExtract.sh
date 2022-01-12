#!/usr/bin/bash

#SBATCH --job-name=ExtractionTest
#SBATCH -p swat_plus
#SBATCH -n 1
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/scripts
#SBATCH -t 01:00:00


echo "Start-date: $STARTDATE"
echo "End-date: $ENDDATE"
python extract.py $STARTDATE $ENDDATE

