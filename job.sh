#!/usr/bin/bash

#SBATCH --job-name=testjc
#SBATCH -p swat_plus
#SBATCH -n 2
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH --output=jobcontrol_output_%J.out
#SBATCH --error=jobcontrol_error_%J.out
#SBATCH -t 00:10:00

module load Python/3.9.5-GCCcore-10.3.0
source ~/testflow/bin/activate
python -u test_job_control.py 


