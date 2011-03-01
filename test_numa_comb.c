#define _GNU_SOURCE
#include <stdio.h>
#include <numa.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h> //for pthread_spinlock_t
#include <unistd.h>
#include "tsc.h"

#define N_THREADS 8
#define CPU_FREQ 2270000000
#define EXPERIMENT_TIME_IN_SEC 15
#define MAX_N_ACCESS 100000000
#define S_TO_N 1000000000

//#define DEBUG
double time_in_cs[N_THREADS];
int num_access_each_thread[N_THREADS];
int flag = 0;
pthread_t threads[N_THREADS];
cpu_set_t cpuset[N_THREADS];
int cs_order[MAX_N_ACCESS];
int access_count = 0;
uint64_t get_lock_time[MAX_N_ACCESS];
uint64_t release_lock_time[MAX_N_ACCESS];

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
	double sevice_time = 1.0;
	uint64_t start_time = read_tsc();
	int s; //for handling the return value of pthread_setaffinity
	int initial_loop_flag = 1;
	s = pthread_setaffinity_np(threads[tid], sizeof(cpu_set_t), &cpuset[tid]);
	if (s != 0)
		perror("set affinity error\n");
	while(read_tsc() - start_time < EXPERIMENT_TIME_IN_SEC*CPU_FREQ)
	{
		int if_print_cs_info = 1;
		uint64_t start = read_tsc();
		//if(!(tid == 1 && initial_loop_flag == 1)) 
	//	{
		while(read_tsc() - start < sevice_time*CPU_FREQ/2)
			;
	//	}
		uint64_t getlock_time = read_tsc();
		#ifdef DEBUG
		printf("Thread %d trying to get the lock at %lu \n",tid,getlock_time);
		#endif
		pthread_spin_lock(thread_param->spinlock_ptr);	
		#ifdef DEBUG
		printf("Thread %d got the lock at %lu \n",tid,read_tsc());
		#endif
		/************The critical section***************/
		uint64_t get_in_cs_time = read_tsc();
		get_lock_time[access_count] = get_in_cs_time;
		while(read_tsc() - get_in_cs_time < sevice_time*CPU_FREQ)
		{
		#ifdef DEBUG
			if(if_print_cs_info == 1) 
			{
				printf("Thread %d in cs.\n",tid);
				if_print_cs_info = 0;
			}
		#endif
			;
		}
		cs_order[access_count] = tid; //Record my id in the cs order array
		//printf("access count: %d\n",access_count);
		release_lock_time[access_count] = read_tsc();
		access_count++;
		/************End the critical section***************/
		pthread_spin_unlock(thread_param->spinlock_ptr);
		uint64_t end_cs_time = read_tsc();
#ifdef DEBUG
		printf("Thread %d released the lock at %lu \n",tid,end_cs_time);
#endif
		num_access_each_thread[tid]++;
		time_in_cs[tid] +=  (end_cs_time - getlock_time)/(double)CPU_FREQ;
	}
	return ;
}

double abs_value(double a)
{
	if(a > 0)
		return a;
	else
		return -a;
}
int main(int argc, char *argv[])
{
	int node_id = 0;
	int arrival_lambda = 10;
	int thread_cpu_map[N_THREADS] = {1,6}; 
	int i,j;
	int n_threads;
	int n_left;
	int n_right;
	int max_n_left = 3;
	int max_n_right = 7;
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
	/*****************make sure the number of arguments is correct and get the n_threads, n_left and n_right*/
	if(argc < 4)
	{
		printf("Usage: ./test_numa_comb number_of_threads number_of_threads_on_node0 number_of_threads_on_node1\n");
		exit(-1);
	}
	n_threads = atoi(argv[1]);
	n_left = atoi(argv[2]);
	n_right = atoi(argv[3]);
	/*******************Set the thread_cpu_map according to the n_left and n_right*/
	printf("n_threads: %d, n_left: %d, n_right: %d\n",n_threads,n_left,n_right);
	for(i = 0; i < n_left; i++)
	{
		thread_cpu_map[i] = max_n_left;
		max_n_left--;
	}
	for(i = n_left; i < n_threads; i++)
	{
		thread_cpu_map[i] = max_n_right;
		max_n_right--;
	}
	for(i = 0; i < n_threads; i++)
	{
		printf("Thread %d is on cpu %d\n",i,thread_cpu_map[i]);
	}

	thread_params para[n_threads]; //The parameters to pass to the threads

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
	//initlize spinlock
	pthread_spin_init(spinlock_ptr,0);
	//create the threads
	for(i = 0; i < n_threads; i++)
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
	for(i = 0; i < n_threads; i++)
	{
		pthread_join(threads[i],NULL);
	}
	for(i = 0; i < n_threads; i++)
	{
		printf("The time to get one lock for thread %d is : %.9f\n",i,time_in_cs[i]/num_access_each_thread[i]);
		printf("The number of lock accesses for thread %d is : %d\n",i,num_access_each_thread[i]);
	}
	//double diff = abs_value(time_in_cs[0]/num_access_each_thread[0] - time_in_cs[1]/num_access_each_thread[1]);
	//printf("The difference of the time to get one lock is : %.9f (%f cycles)\n",diff,diff*CPU_FREQ); //this is assuming there are only two processors (needs to be changed if there are more)

	for (i = 0; i < access_count; i++)
		printf("%d \n",cs_order[i]);
	printf("printing the get lock and release lock time");
	for (i = 0; i < access_count; i++)
		printf("%lu,%lu\n ",get_lock_time[i],release_lock_time[i]);
	for (i = 0; i < access_count - 1; i++)
	{
		double d = (double)(get_lock_time[i+1] - release_lock_time[i]);
		printf("%d release to %d require: %.9f seconds (%.9f nanoseconds,%d cycles)\n",cs_order[i],cs_order[i+1],(d/CPU_FREQ),(d/CPU_FREQ)*S_TO_N,d);
	}

	pthread_spin_destroy(spinlock_ptr);
	numa_free((void *)spinlock_ptr,sizeof(pthread_spinlock_t));
	pthread_exit(NULL);
	return 0;
}
