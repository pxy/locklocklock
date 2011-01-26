source params.dat
#generate the script to draw graphs for different thread numbers
echo "generate the script to draw graphs for different thread numbers"
echo "set term post eps" >> $path_to_save/to_plot2.plt
for ((  i = $mu_start;  i <= $mu_end;  i *= $step ))
do
echo "set output '$path_to_save/comparison_d_$i.ps'" >> $path_to_save/to_plot2.plt
echo -n "plot '$path_to_save/comparison_mul_threads_d_$if_deterministic" >> $path_to_save/to_plot2.plt
echo -n "_$i'" >> $path_to_save/to_plot2.plt
echo -n "using 1:2 with lines, '$path_to_save/comparison_mul_threads_d_$if_deterministic" >> $path_to_save/to_plot2.plt
echo "_$i' using 1:3 with lines" >> $path_to_save/to_plot2.plt
done
gnuplot $path_to_save/to_plot2.plt
