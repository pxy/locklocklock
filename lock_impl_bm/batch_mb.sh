
for i in 500 750 1000 1250 1500 1750 2000; do 
    if [ ! -d "$i" ]; then
	mkdir $i
    fi
    ./consprod -t 30 -c 0 -d -w$i,100000 -s100,100 && mv output* $i
done