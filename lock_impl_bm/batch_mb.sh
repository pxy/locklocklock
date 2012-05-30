#!/bin/sh
#SBATCH -J mr_mbm
#SBATCH -t 00:20:00
#SBATCH -p node
#SBATCH -n 1
#SBATCH -A p2010018
## --qos=short 
#SBATCH --mail-type=all
#SBATCH --mail-user=jonatan.linden@it.uu.se

MODE=0
if [[ $# = 1 && "$1" = "-l" ]]; then
    MODE=1   
fi

SERVER=tintin
#LOCK=1 #clh
LOCK=0 #pth
TIME=120


for i in 1 2 4 6 8 10 12 14 16 ;do
    if [[ $MODE = 1 ]]; then
        scp -r ${SERVER}:locklocklock/lock_impl_bm/"$i" .
        continue
    fi

    if [ ! -d $i ]; then
	mkdir $i
    fi
    #./consprod -n $i -t 60 -c $LOCK -w10000,10000 -s1000,1000 && mv output* $i
    ./consprod -n $i -t ${TIME} -c $LOCK -w20000,20000 -s500,500 > $i/mbm_${i}_${LOCK}ee_${TIME}.dat
done
