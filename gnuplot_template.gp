#!/opt/local/bin/gnuplot
set terminal aqua 0 title "my file title" size 1024,768 font "Helvetica,20" enhanced dashed
# set output

# move the legend out of the way
set key top left


# a bit bolder line styles
set style line 1 lt 5 lc rgb "orange" lw 2
set style line 2 lt 2 lc rgb "orange" lw 3
set style line 3 lt 5 lc rgb "blue" lw 2
set style line 4 lt 2 lc rgb "blue" lw 3
set style line 5 lt 5 lc rgb "red" lw 2



# using points?
#set pointsize 2

# tics
set tics font "Helvetica, 20" nomirror   # remove cluttering
set xtics font "Helvetica, 20"
set ytics font "Helvetica, 20"

set xtics 200000,200000,1000000          # sparse tics
set mxtics
set mytics 2

# finetuning of ranges
set yrange [0:]


# labels
set xlabel "exp. distr. lock attempt rate" font "Helvetica, 20"
set ylabel "avg. waiting time" font "Helvetica, 20"


plot '4threads.dat' index 2 u 1:2 t "4x0 t exp. meas." w l ls 1,\
	 '4threads.dat' index 2 u 1:4 t "2x2 t exp. meas." w l ls 3,\
     '4threads.dat' index 1 u 1:2 t "4x0 t model" w l ls 2,\
     '4threads.dat' index 0 u 1:2 t "2x2 t model" w l ls 4,\
     '4threads.dat' index 2 u 1:3 t "Q. model" w l ls 5,\
	 '4threads.dat' index 3 u 1:2 t "4x0 det. meas." w l ls 6



