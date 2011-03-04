
cp "$1".net tmp.net
spn_update.py --name mu --val 1000000 tmp > tmp2.net
for i in {100000..1000000..100000}; do
    spn_update.py --name lambda --val $i tmp2 > "$1.net"
    spn_solve.sh "$1" &> /dev/null
    echo -n "$i "
    spn_calc.py -t T9 -p P2 P4 P5 P6 -- "$1"
done