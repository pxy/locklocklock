#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include "tsc.h"
#include "cs_queue.c"

//#define DEBUG
#define FIX_TIME
#define NUM_THREADS 2
#define PRNG_BUFSZ 32
#define E 2.71828
//#define SERVICE_MIU 100
#define USE_SPINLOCK
#define EXPERIMENT_TIME_IN_SEC 60
#define LINUX // activate linux specific code
#define CPU_FREQ 2270000000

//For the padding of the thread parameters structure
#define CACHE_LINE_SIZE 64

int wait_flag ;
int SERVICE_MIU = 0;
int d_flag;

double time_in_cs[NUM_THREADS];
double experiment_time;
pthread_mutex_t sum_lock;
double service_time_sum[NUM_THREADS];
int num_lock_access = 0;
int num_access_each_thread[NUM_THREADS];

typedef struct
{
	int thread_id;
	struct random_data* rand_states;
	int arrival_lambda;
	//char padding[CACHE_LINE_SIZE-2*sizeof(int)-sizeof(struct random_data*)];
}thread_params;

//Given a randomly generated rand, convert a uniform distribution into an exponenetial distribution
double exp_rand(float miu,int rand)
{
	double ran = (-log((double)rand/(double)RAND_MAX)/miu);
	return ran;
}
double time_diff(struct timespec *start,struct timespec *end)
{
	return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec)*1e-9;
}

void *work(void *thread_arg)
{
#ifndef FIX_TIME
	int i;
#endif
	thread_params *thread_param = (thread_params *)thread_arg;
	double service_time, arrival_time;
	//get the parameters from thread_param
	int tid = thread_param->thread_id;
	int  arrival_lambda = thread_param->arrival_lambda;
	struct random_data* arg = thread_param->rand_states;
	uint64_t start_idle,start_lock_attempt,get_lock,end_cs;
	while(!wait_flag) //If we are using spinlock, we need to wait until all threads are created
	{
		;
	}
	uint64_t start_experiment_time = read_tsc(); 
	while(read_tsc() - start_experiment_time < EXPERIMENT_TIME_IN_SEC*CPU_FREQ) 
	{
		int r;
		//generate a random number according to the arrival lambda
		start_idle = read_tsc();
		random_r(arg, &r);
		//if(d_flag)
			//arrival_time = 1/(double)arrival_lambda;
		//else
			arrival_time = exp_rand(arrival_lambda,r);
		while(read_tsc() - start_idle < arrival_time*CPU_FREQ)
		{
			;
		}
		//printf("the arrival time for thread %d: %lu\n",tid,read_tsc()-start_idle);
		start_lock_attempt = read_tsc();

#ifdef DEBUG
		//printf("trylock %lf %d\n",(start_lock_attempt.tv_sec + start_lock_attempt.tv_nsec*1e-9),tid);
		//printf("Idling time: %lf\n",(time_diff(&start_idle,&start_lock_attempt)));
		if(time_diff(&start_idle,&start_lock_attempt) - arrival_time > 0.0001)
			printf("Actual idling time and generated arrival time difference: %lf\n",(time_diff(&start_idle,&start_lock_attempt) - arrival_time));
#endif

		//printf("Thread %d going to call spin_lock\n",tid);
		impl_enter_critical(tid);
			;
		//printf("Thread %d got the lock.\n",tid);
		get_lock = read_tsc();
		//printf("Thread %d in cs\n",tid);
#ifdef DEBUG
		printf("acquirelock %lf %d\n",(get_lock.tv_sec + get_lock.tv_nsec*1e-9),tid);
#endif
		random_r(arg, &r);
		if(d_flag)
			service_time = 1/(double)SERVICE_MIU;
		else
			service_time = exp_rand(SERVICE_MIU,r);
		//service_time_sum[tid] += service_time;
		while(read_tsc() - get_lock < service_time*CPU_FREQ)
		{
			;
		}
		//assert(time_diff(&start,&end_cs) >= service_time);
		impl_exit_critical(tid);
		end_cs = read_tsc();

		num_lock_access++;
		num_access_each_thread[tid]++;
		time_in_cs[tid] +=  (end_cs - start_lock_attempt)/(double)CPU_FREQ;
	}
	pthread_exit(NULL);
}

int main (int argc, char *argv[])
{
	pthread_t threads[NUM_THREADS];
	int rc;
	int t;
	double sum;
	thread_params para[NUM_THREADS];
	//double inter_arrival_time[NUM_THREADS];
	srand(time(NULL));
	int i= 0;
#ifdef LINUX
	cpu_set_t cpuset[NUM_THREADS]; //for setting affinity of the threads
#endif
	int count = 0;
	int s;
	int arrival_lambda;
	//Initialize random states variable
	struct random_data* rand_states;
	char* rand_statebufs;
	pthread_t* thread_ids;

	/*allocate memory*/
	rand_states = (struct random_data*)calloc(NUM_THREADS, sizeof(struct random_data));
	rand_statebufs = (char*)calloc(NUM_THREADS, PRNG_BUFSZ);
	thread_ids = (pthread_t*)calloc(NUM_THREADS, sizeof(pthread_t));
	for(i = 0; i < NUM_THREADS; i++)
	{
		//service_time_sum[i] = 0;
		num_access_each_thread[i] = 0;
	}
	if(argc < 4)
	{
		printf("Usage: ./test arrival_rate service_rate deterministic_flag\n");
		exit(-1);
	}
	arrival_lambda = atoi(argv[1]);
	SERVICE_MIU = atoi(argv[2]);
	d_flag = atoi(argv[3]);
//The following line is used to set the affinity of the threads to add architecture-specific feature into the measurement
	//int thread_cpu_map[NUM_THREADS] = {4,5,6,7,2,3}; //The 2-4 structure for 6 threads
	//int thread_cpu_map[NUM_THREADS] = {4,5,6,2,3}; //The 2-3 structure for 5 threads
	//int thread_cpu_map[NUM_THREADS] = {4,5,6,7,3}; //The 1-4 structure for 5 threads
	//int thread_cpu_map[NUM_THREADS] = {0,1,2,3,7}; //The 4-1 structure for 5 threads
	//int thread_cpu_map[NUM_THREADS] = {2,3,4}; //The 2-1 structure for 5 threads
	//int thread_cpu_map[NUM_THREADS] = {2,4,5}; //The 1-2 structure for 5 threads
	//int thread_cpu_map[NUM_THREADS] = {0,1,2,3,7}; //The 4-1 structure for 5 threads
	//int thread_cpu_map[NUM_THREADS] = {1,2,3,4,5,6}; //The 3-3 structure for 6 threads
	//int thread_cpu_map[NUM_THREADS] = {0,1,6,7};

	sum = 0;
	//initialize the lock
	impl_init(NUM_THREADS);
	//create the threads
	for(t = 0; t < NUM_THREADS; t++){
		para[t].thread_id = t;
		time_in_cs[t] = 0;
		//for the random variable
		initstate_r(random(), &rand_statebufs[t], PRNG_BUFSZ, &rand_states[t]);
		para[t].rand_states = &rand_states[t];
		para[t].arrival_lambda = arrival_lambda;
		rc = pthread_create(&threads[t], NULL, work, (void *)&para[t]);
		//printf("from main, create threads,test test\n");
		if (rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(-1);
			;
		}
	}
#ifdef LINUX
	//set the affinity of threads
	for(t = 0; t < NUM_THREADS; t++){
		CPU_ZERO(&cpuset[t]);	
		CPU_SET(t,&cpuset[t]);
		//CPU_SET(thread_cpu_map[t],&cpuset[t]);
		s = pthread_setaffinity_np(threads[t], sizeof(cpu_set_t), &cpuset[t]);
		if (s != 0)
			printf("pthread_setaffinity_np error of thread %d\n",t);
		count++;
		//printf("The value of count: %d\n",count);
	}
	if(count == NUM_THREADS)
		wait_flag = 1;
#endif
	/* Wait on the other threads */
	for(t=0; t<NUM_THREADS; t++)
	{
		pthread_join(threads[t], NULL);
	}
//finalize the lock
	impl_fini();
#ifdef FIX_TIME
	double sum_of_square = 0.0;
	assert(num_lock_access > 0);
	for(t = 0; t < NUM_THREADS; t++)
		assert(num_access_each_thread[t] > 0);
#endif

	for(t = 0; t < NUM_THREADS; t++)
	{
		sum += time_in_cs[t];
	}
#ifdef FIX_TIME
	for(t = 0; t < NUM_THREADS; t++)
	{
		sum_of_square += (time_in_cs[t]/(double)num_access_each_thread[t])* (time_in_cs[t]/(double)(num_access_each_thread[t]));
	}
	//for(t = 0; t < NUM_THREADS; t++)
	//{
		//printf("The number of lock accesses of thread %d: %d, total critical section time: %f, average cs time: %f.\n",t,num_access_each_thread[t],time_in_cs[t],time_in_cs[t]/num_access_each_thread[t]);
	//}
	//printf("There are %d lock accesses.\n",num_lock_access);
	printf("Average response time for lambda %d mu %d %.9f , variance: %.9f \n",arrival_lambda,SERVICE_MIU,sum/((double)num_lock_access),sum_of_square/(double)NUM_THREADS-(sum/(double)num_lock_access)*(sum/(double)num_lock_access)); 
	//printf("Sum of square: %f\n",sum_of_square);
	//printf("Square average: %f\n",sum_of_square/(double)NUM_THREADS);
	//printf("Average square:%f\n",(sum/(double)num_lock_access)*(sum/(double)num_lock_access));
	//printf("Variance: %f\n",sum_of_square/(double)NUM_THREADS-(sum/(double)num_lock_access)*(sum/(double)num_lock_access));
#else
	printf("Average response time for lambda %d: %f\n",arrival_lambda,sum/((double)NUM_THREADS*(NUM_LOOP - NUM_TAIL_LOOP)));
#endif
	/*
	   for(t = 0; t < NUM_THREADS; t++)
	   {
	   printf("The serivce time for thread: %d: %lf\n",t,service_time_sum[t]/(double)(NUM_LOOP-NUM_TAIL_LOOP)); 
	   }
	 */
	/*
	   int j,k;
	   for(t = 0; t < NUM_THREADS; t++)
	   {
	   for(i = 0; i < NUM_LOOP; i++)
	   {
	   for(k = 0; k < NUM_THREADS && k != t ; k++)
	   {
	   for(j = 0; j < NUM_LOOP; j++)
	   if(time_diff(&(timestamps[j][k].lock_time),&(timestamps[i][t].lock_time)) > 0)
	   assert (time_diff(&(timestamps[j][k].lock_time),&(timestamps[i][t].unlock_time)) > 0);
	   }
	   }
	   }*/
	pthread_exit(NULL);
}
