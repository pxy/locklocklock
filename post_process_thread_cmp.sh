source params.dat
for ((  i = $mu_start;  i <= $mu_end;  i *= $step ))
do
for ((  n = $n_threads_start;  n <= $n_threads_end;  n += 2 ))
do
head -n 1 $path_to_save/d_$if_deterministic\_$n\_threads_$i | cut -d ' ' -f9 >> $path_to_save/tmp_$i
head -n 1 $path_to_save/model_$n\_threads_$i | cut -d ' ' -f1 >> $path_to_save/tmp_model_$i
done
paste $path_to_save/threads $path_to_save/tmp_$i $path_to_save/tmp_model_$i > $path_to_save/comparison_mul_threads_d_$if_deterministic\_$i
done
