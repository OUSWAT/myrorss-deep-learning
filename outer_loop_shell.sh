#!/usr/bin/bash
#SBATCH -p swat_plus 
#SBATCH -t  08:00:00
#SBATCH --error=accumloop_error_%J.out 
#SBATCH --job-name=outerloopcronv
STARTDATE=20110326
ENDDATE=20110327

DATE=( $(seq $STARTDATE 1 $ENDDATE) )
echo ${DATE[@]}

for i in ${DATE[@]}
do

        while true; do
                squeue -u mcmontalbano
                PENDING=$(expr $(squeue -u mcmontalbano --start -h | wc -l) )
                echo "passed PENDING" 

                if [ $PENDING -lt 5 ]; then
                        echo $i 

                        cd /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning
                        echo "Running CLUE $i" 
                        sbatch --export DATE="$i" ext_shell.sh
                        date
                        sleep 60
                        break

                fi

                echo " TOO MANY PENDING JOBS RUNNING WAIT " 

                sleep 1800 

        done


done
