#!/bin/bash
gnuplot << EOF
#set terminal pdf size 2cm,1.2cm #0 size 1024,768 enhanced dashed
set terminal aqua size 1024,768
set output "$1.pdf"
set title "$1"
set style fill solid 1.00 border -1
set style histogram #cluster gap 1
set style data histogram
set xtics nomirror rotate by -45
#set ytics 500 scale 0.1
#set format y ""
#set format x ""
set yrange [0:*]
set ylabel 'waiting time in cycles'
set xlabel '#threads per parallel stage'


plot "$1.dat" using 2:xtic(1) t "actual", '' using 3 t "estimate", '' using 4 t 'service time'
EOF
