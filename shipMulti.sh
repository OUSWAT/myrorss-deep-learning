#!/usr/bin/bash
#SBATCH -p swat_plus
#SBATCH -J shipmulti
#SBATCH -t 24:00:00
#SBATCH --chdir /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning

STARTDATES=19980522
ENDDATES=(19980523)
one = 1
for i in ${ENDDATES[@]}
do 
    start=$(( $i - $one))
    echo "i = $i"
    echo "start = $start"

for i in ${ENDDATES[@]}
do
    DATE=( $(seq $STARTDATES 1 ${ENDDATES[i]}) )
    echo ${DATE[@]}
        while true; do
                squeue -u mcmontalbano
                PENDING=$(expr $(squeue -u mcmontalbano -n data -h | wc -l) )
                echo "passed PENDING"

                if [ $PENDING -lt 1 ]; then
                    echo $i

                    cd /condo/swatwork/mcmontalbano/MYRORSS/myrorss-deep-learning/scripts
                    echo "Running SegREDO $i"
                    sbatch --export=STARTDATE=$STARTDATES,ENDDATE=$i shipExtract.sh
                    date
                    echo "sbatch --export=STARTDATE=$STARTDATES,ENDDATE=$i shipExtract.sh"
                    sleep 60
                    break

                fi

                echo " TOO MANY PENDING JOBS RUNNING WAIT "

                sleep 1200

        done

    STARTDATES=$(( $STARTDATES + 100 ))
    echo $STARTDATES

done

