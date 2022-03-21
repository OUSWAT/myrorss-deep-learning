#! /usr/bin/bash
#SBATCH --job-name=strmMKR
#SBATCH -p swat_plus
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH --exclusive=user
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
#SBATCH -t 08:00:00
#SBATCH --output=strm_output_%J_stdout.out
#SBATCH --error=strm_error_%J_stderr.out
#SBATCH --mail-user=michael.montalbano@ou.edu

module load Python/3.9.5-GCCcore-10.3.0
source ~/testflow/bin/activate

echo "Running EXTRACTION: $DATE" 
python extract_shell.py $DATE 
 


