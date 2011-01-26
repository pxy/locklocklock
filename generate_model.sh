source params.dat
export PYTHONPATH=/home/xiaoyue/lib/python2.4/site-packages/
#Generate data from the model
for ((  i = $mu_start;  i <= $mu_end;  i *= $step ))
do
for ((  n = $n_threads_start;  n <= $n_threads_end;  n += 2 ))
	do
lambda_start=$((i/$step))
	if [ $if_deterministic -ne 0 ];
	then
	./machine_repairman.py --steps $step -l $lambda_start $i -m $i $i -t $n $n --deterministic > $path_to_save/model_$n\_threads_$i
	else
	./machine_repairman.py --steps $step -l $lambda_start $i -m $i $i -t $n $n > $path_to_save/model_$n\_threads_$i
	fi
#./machine_repairman.py --steps $step -l $mu_start $i -m $i $i -t $n $n > $path_to_save/model_$n\_threads_$i
	done
	done


