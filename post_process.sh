source params.dat

echo "Post processing the data"
#post processing of the data
for ((  i = $mu_start;  i <= $mu_end;  i *= $step ))
do
for ((  n = $n_threads_start;  n <= $n_threads_end;  n += 2 ))
do
	cut -d ' ' -f6,9 $path_to_save/d_$if_deterministic\_$n\_threads_$i > $path_to_save/tmp1
	cut -d ' ' -f1 $path_to_save/model_$n\_threads_$i > $path_to_save/tmp2
	paste $path_to_save/tmp1 $path_to_save/tmp2 > $path_to_save/comparison_d_$if_deterministic\_$n\_threads_$i
	awk '{for (i=1; i<=NF; i++); print $1 "\t" $2 "\t" $3 "\t" $2-$3 "\t" ($2-$3)/$3 "\t" ($2-$3)/$2'} < $path_to_save/comparison_d_$if_deterministic\_$n\_threads_$i > $path_to_save/comparison_d_$if_deterministic\_$n\_threads_$i\_with_error
done
done
