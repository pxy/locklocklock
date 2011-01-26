source params.dat
#Run the experiment 
echo "Running the experiment"
for ((  i = $mu_start;  i <= $mu_end;  i *= $step ))
do      
for ((  j = i/$step;  j <= i;  j += i/$step ))
do           
for ((  n = $n_threads_start;  n <= $n_threads_end;  n += 2 ))
do
./test_t_v_$n $j $i $if_deterministic >> $path_to_save/d_$if_deterministic\_$n\_threads_$i
done
done
done

