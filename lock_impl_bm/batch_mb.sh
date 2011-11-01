
MODE=0
if [[ $# = 1 && "$1" = "-l" ]]; then
    MODE=1   
fi


#for i in 500 750 1000 1100 1150 1200 1250 1300 1350 1400 1500 1750 2000 10000 50000 100000; do 
for i in 9500 40000 60000 80000 90000; do
    if [[ $MODE = 1 ]]; then
        scp -r kalkyl.uppmax.uu.se:locklocklock/lock_impl_bm/"$i" .
        continue
    fi
        
    if [ ! -d "$i" ]; then
	mkdir $i
    fi
    ./consprod -t 30 -c 0 -d -w$i,100000 -s100,100 && mv output* $i
done