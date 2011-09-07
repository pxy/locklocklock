#!/bin/bash
gnuplot << EOF
#set terminal pdf size 2cm,1.2cm #0 size 1024,768 enhanced dashed
set terminal aqua size 1024,768
set output "$1.pdf"
set nokey
set ylabel 'num of timings'
set xlabel 'waiting time'
set style fill solid #1.00 border -1
set style data histogram
set grid
set xrange[ 0: ]
#set xtics 5 scale 0.1 nomirror
#set ytics 500 scale 0.1
#set format y ""
#set format x ""

set yrange[0:]
plot "$1.dat" using 1 
EOF
