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

//#define DEBUG
#define FIX_TIME
#define NUM_THREADS 8
#define PRNG_BUFSZ 32
#define E 2.71828
#define SECOND_TO_USECOND 1000000
//#define SERVICE_MIU 100
#define NUM_LOOP 500
#define NUM_HEAD_LOOP 100
#define NUM_TAIL_LOOP 150 
#define USE_SPINLOCK
#define EXPERIMENT_TIME_IN_SEC 60 
#define LINUX // activate linux specific code
#define CPU_FREQ 2270000000

//For the padding of the thread parameters structure
#define CACHE_LINE_SIZE 64

#ifdef USE_SPINLOCK
pthread_spinlock_t spinlock;
#else
pthread_mutex_t sum_lock;
#endif 

int wait_flag ;
int SERVICE_MIU = 0;

double time_in_cs[NUM_THREADS];
double ex_time[NUM_THREADS]; 
double experiment_time;
pthread_mutex_t sum_lock;
double service_time_sum[NUM_THREADS];
int num_lock_access = 0;
int num_access_each_thread[NUM_THREADS];

struct lock_unlock_time
{
	struct timespec lock_time;
	struct timespec unlock_time;
};
struct lock_unlock_time timestamps[NUM_LOOP][NUM_THREADS];

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
#ifdef FIX_TIME
	while(read_tsc() - start_experiment_time < EXPERIMENT_TIME_IN_SEC*CPU_FREQ) 
#else	
	for(i = 0; i < NUM_LOOP; i++)
#endif
	{
		int r;
		//generate a random number according to the arrival lambda
		start_idle = read_tsc();
		random_r(arg, &r);
		//if(i == 0)
		arrival_time = exp_rand(arrival_lambda,r);
		//arrival_time = 1/(double)arrival_lambda;
		while(read_tsc() - start_idle < arrival_time*CPU_FREQ)
		{
			;
		}
		start_lock_attempt = read_tsc();

#ifdef DEBUG
		//printf("trylock %lf %d\n",(start_lock_attempt.tv_sec + start_lock_attempt.tv_nsec*1e-9),tid);
		//printf("Idling time: %lf\n",(time_diff(&start_idle,&start_lock_attempt)));
		if(time_diff(&start_idle,&start_lock_attempt) - arrival_time > 0.0001)
			printf("Actual idling time and generated arrival time difference: %lf\n",(time_diff(&start_idle,&start_lock_attempt) - arrival_time));
#endif

#ifdef USE_SPINLOCK
		pthread_spin_lock(&spinlock); 
#else
		//blocking mutex
		//pthread_mutex_lock (&sum_lock);
		//non blocking mutex
		while(pthread_mutex_trylock(&sum_lock)!= 0) 
			;
#endif
		get_lock = read_tsc();
#ifdef DEBUG
		printf("acquirelock %lf %d\n",(get_lock.tv_sec + get_lock.tv_nsec*1e-9),tid);
#endif
		random_r(arg, &r);
		service_time = exp_rand(SERVICE_MIU,r);
		//service_time = 1/(double)SERVICE_MIU;
		//service_time_sum[tid] += service_time;
		while(read_tsc() - get_lock < service_time*CPU_FREQ)
		{
			;
		}
		//assert(time_diff(&start,&end_cs) >= service_time);
#ifdef USE_SPINLOCK
		pthread_spin_unlock(&spinlock);
#else
		pthread_mutex_unlock (&sum_lock);
#endif
		end_cs = read_tsc();
#ifdef DEBUG
#endif
#ifdef FIX_TIME
		num_lock_access++;
		num_access_each_thread[tid]++;
		time_in_cs[tid] +=  (end_cs - start_lock_attempt)/(double)CPU_FREQ;
		//printf("end_cs: %u\n",end_cs);
		//printf("start_lock_attempt: %u\n",start_lock_attempt);
		//printf("difference: %u\n",end_cs - start_lock_attempt);
		//printf("Printing contention time: %f\n",(end_cs - start_lock_attempt)/(double)CPU_FREQ);
#else
		//Store the difference of the start and end critical section time
		if(i < NUM_LOOP - NUM_TAIL_LOOP) //don't include the first NUM_TAIL_LOOP critical section time
			time_in_cs[tid] +=  time_diff(&start_lock_attempt,&end_cs);
#endif
		//assert(time_diff(&start_lock_attempt,&end_cs) >= service_time);
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

#ifdef USE_SPINLOCK
	wait_flag = 0;
#else
	wait_flag = 1;
#endif	
	/*allocate memory*/
	rand_states = (struct random_data*)calloc(NUM_THREADS, sizeof(struct random_data));
	rand_statebufs = (char*)calloc(NUM_THREADS, PRNG_BUFSZ);
	thread_ids = (pthread_t*)calloc(NUM_THREADS, sizeof(pthread_t));
	for(i = 0; i < NUM_THREADS; i++)
	{
		//service_time_sum[i] = 0;
		num_access_each_thread[i] = 0;
	}
	if(argc < 3)
	{
		printf("Usage: ./test arrival_rate service_rate\n");
		exit(-1);
	}
	arrival_lambda = atoi(argv[1]);
	SERVICE_MIU = atoi(argv[2]);

	sum = 0;
#ifdef USE_SPINLOCK
		pthread_spin_init(&spinlock, 0);
#else
		pthread_mutex_init(&sum_lock, NULL);
#endif
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
			s = pthread_setaffinity_np(threads[t], sizeof(cpu_set_t), &cpuset[t]);
			if (s != 0)
				printf("pthread_setaffinity_np error of thread %d\n",t);
			count++;
			//printf("The value of count: %d\n",count);
		}
		if(count == NUM_THREADS);
		wait_flag = 1;
#endif
		/* Wait on the other threads */
		for(t=0; t<NUM_THREADS; t++)
		{
			pthread_join(threads[t], NULL);
		}
#ifdef USE_SPINLOCK
		pthread_spin_destroy(&spinlock);
#else
		pthread_mutex_destroy(&sum_lock);
#endif
		/*#ifdef DEBUG
		  printf("The experiment took %.6lf microseconds.\n", t2-t1);
		  for(t=0; t<NUM_THREADS; t++){
		  printf("Time in cs: for thread %d: %lf \n",t,time_in_cs[t]);
		  printf("Execution time for thread %d: %lf \n",t,ex_time[t]); 
		  }
#endif*/
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
		printf("Average response time for lambda %d, mu %d: %f , variance: %.9f \n",arrival_lambda,SERVICE_MIU,sum/((double)num_lock_access),sum_of_square/(double)NUM_THREADS-(sum/(double)num_lock_access)*(sum/(double)num_lock_access)); 
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
