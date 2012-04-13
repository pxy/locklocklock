#!/bin/sh
MODE=0
if [[ $# = 1 && "$1" = "-l" ]]; then
    MODE=1   
fi

SERVER=tintin
LOCK=0 #clh

for i in 1 2 4 8 16 ;do
    if [[ $MODE = 1 ]]; then
        scp -r ${SERVER}:locklocklock/lock_impl_bm/"$i" .
        continue
    fi
        
    if [ ! -d $i ]; then
	mkdir $i
    fi
    ./consprod -n $i -t 60 -c $LOCK -w10000,10000 -s1000,1000 && mv output* $i
done
