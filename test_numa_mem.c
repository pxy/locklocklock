#define _GNU_SOURCE
#include <stdio.h>
#include <numa.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h> //for pthread_spinlock_t
#include <unistd.h>
#include "tsc.h"

#define N_THREADS 2
#define CPU_FREQ 2270000000
double time_in_cs[N_THREADS];
int num_access_each_thread[N_THREADS];
int flag = 0;
pthread_t threads[N_THREADS];
cpu_set_t cpuset[N_THREADS];

typedef struct
{
	int thread_id;
	struct random_data* rand_states;
	int arrival_lambda;
	pthread_spinlock_t *spinlock_ptr;
}thread_params;

void *work(void *thread_arg)
{
	while(!flag)
		;
	thread_params *thread_param = (thread_params*)thread_arg;
	int tid = thread_param->thread_id;
	int arrival_lambda = thread_param->arrival_lambda;
	double sevice_time = 0.00001;
	uint64_t start_time = read_tsc();
	int s; //for handling the return value of pthread_setaffinity
	s = pthread_setaffinity_np(threads[tid], sizeof(cpu_set_t), &cpuset[tid]);
	if (s != 0)
		perror("set affinity error\n");
	if(tid == 0) //wait for a while if i'm thread 0
	{
		while(read_tsc() - start_time < 2*sevice_time*CPU_FREQ)
			;
	}
	uint64_t getlock_time = read_tsc();
	pthread_spin_lock(thread_param->spinlock_ptr);	
	while(read_tsc() - getlock_time < sevice_time*CPU_FREQ)
	{
		;
	}
	pthread_spin_unlock(thread_param->spinlock_ptr);
	uint64_t end_cs_time = read_tsc();
	num_access_each_thread[tid]++;
	time_in_cs[tid] +=  (end_cs_time - getlock_time)/(double)CPU_FREQ;
	return ;
}

double abs_value(double a)
{
	if(a > 0)
		return a;
	else
		return -a;
}
int main(void)
{
	int node_id = 0;
	int arrival_lambda = 10;
	int thread_cpu_map[N_THREADS] = {1,6};
	int n = 10000;
	int i;
	int j;
	/*
	   pthread_spinlock_t *spinlock_ptr = malloc(sizeof(pthread_spinlock_t));
	   if(spinlock_ptr == NULL) //error handling of the malloc of the spinlock
	   {
	   printf("malloc of spinlock failed.\n");
	   }
	   else
	   {
	   printf("malloc of spinlock succeeded.\n");
	   }
	   free(spinlock_ptr);
	 */
	//pthread_t threads[N_THREADS];
	//cpu_set_t cpuset[N_THREADS]; //for setting the affinity of threads
	thread_params para[N_THREADS]; //The parameters to pass to the threads

	//printf("The return value of numa_get_run_node_mask(void) is %d\n",numa_get_run_node_mask());
	//printf("The return value of numa_max_node(void) is %d\n",numa_max_node());
	//numa_tonode_memory((void *)spinlock_ptr,sizeof(pthread_spinlock_t),node_id); //This doesn't work

	//initilize the spinlock pointer and put it on a specific node
	pthread_spinlock_t *spinlock_ptr = numa_alloc_onnode(sizeof(pthread_spinlock_t),node_id);
	if(spinlock_ptr == NULL) //error handling of the allocating of a spinlock pointer on a specific node
	{
		printf("alloc of spinlock on a node failed.\n");
	}
	else
	{
		printf("alloc of spinlock on a node succeeded.\n");
	}
	for(j = 0; j  < n; j++)
	{
		//initlize spinlock
		pthread_spin_init(spinlock_ptr,0);
		//create the threads
		for(i = 0; i < N_THREADS; i++)
		{
			int rc;
			int s;
			para[i].thread_id = i;
			para[i].arrival_lambda = arrival_lambda;
			para[i].spinlock_ptr = spinlock_ptr;
			CPU_ZERO(&cpuset[i]);
			CPU_SET(thread_cpu_map[i],&cpuset[i]);
			rc = pthread_create(&threads[i],NULL,work,(void*)&para[i]);
			if(rc)
			{
				printf("ERROR: return code from pthread_create() is %d for thread %d \n",rc,i);
				exit(-1);
			}
			flag = 1; 
		}
		for(i = 0; i < N_THREADS; i++)
		{
			pthread_join(threads[i],NULL);
		}
	}
	for(i = 0; i < N_THREADS; i++)
	{
		printf("The time to get one lock for thread %d is : %.9f\n",i,time_in_cs[i]/n);
	}
	double diff = abs_value(time_in_cs[0] - time_in_cs[1])/n;
	printf("The difference of the time to get one lock is : %.9f (%f) cycles\n",diff,diff*CPU_FREQ); //this is assuming there are only two processors (needs to be changed if there are more)
	pthread_spin_destroy(spinlock_ptr);
	numa_free(spinlock_ptr,sizeof(pthread_spinlock_t));
	pthread_exit(NULL);
	return 0;
}
