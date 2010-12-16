#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

//#define DEBUG
#define NUM_THREADS 4
#define PRNG_BUFSZ 32
#define E 2.71828
#define SECOND_TO_USECOND 1000000
#define SERVICE_MIU 100
#define NUM_EXPERIMENTS 1
#define NUM_LOOP 100
#define NUM_HEAD_LOOP 0
#define USE_SPINLOCK
#define LINUX // activate linux specific code

//For the padding of the thread parameters structure
#define CACHE_LINE_SIZE 64

#ifdef USE_SPINLOCK
pthread_spinlock_t spinlock;
#else
pthread_mutex_t sum_lock;
#endif 
int wait_flag ;

double time_in_cs[NUM_THREADS];
double ex_time[NUM_THREADS]; 
double experiment_time;
pthread_mutex_t sum_lock;
double service_time_sum[NUM_THREADS];

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
	char padding[CACHE_LINE_SIZE-2*sizeof(int)-sizeof(struct random_data*)];
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
	int i;
	thread_params *thread_param = (thread_params *)thread_arg;
	double service_time, arrival_time;
	//get the parameters from thread_param
	int tid = thread_param->thread_id;
	int  arrival_lambda = thread_param->arrival_lambda;
	struct random_data* arg = thread_param->rand_states;
	
	//try timespec
	struct timespec start_idle,start_lock_attempt,start,end_cs,get_lock;
	
	//printf("The size of time_t: %lu \n",sizeof(time_t));
	while(!wait_flag) //If we are using spinlock, we need to wait until all threads are created
	{
		;
	}

	for(i = 0; i < NUM_LOOP; i++)
	{
		int r;
		//generate a random number according to the arrival lambda
		if(clock_gettime(CLOCK_MONOTONIC,&start_idle) !=0)
		{
			printf("get monotonic start_waiting time error\n");
			exit(-1);
		}
		random_r(arg, &r);
		//if(i == 0)
		arrival_time = exp_rand(arrival_lambda,r);
		//else
			//arrival_time = 1/(double)arrival_lambda;
		if(clock_gettime(CLOCK_MONOTONIC,&start_lock_attempt) != 0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		while(time_diff(&start_idle,&start_lock_attempt) < arrival_time)
		{
			if(clock_gettime(CLOCK_MONOTONIC,&start_lock_attempt)!=0)
			{
				printf("get monotonic time error\n");
				exit(-1);
			}
		}

		//assert(time_diff(&start_idle,&start_lock_attempt) - arrival_time > 0.0001)
#ifdef DEBUG
		printf("trylock %lf %d\n",(start_lock_attempt.tv_sec + start_lock_attempt.tv_nsec*1e-9),tid);
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
		if(clock_gettime(CLOCK_MONOTONIC,&get_lock)!=0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
#ifdef DEBUG
		printf("acquirelock %lf %d\n",(get_lock.tv_sec + get_lock.tv_nsec*1e-9),tid);
		timestamps[i][tid].lock_time.tv_sec = start_lock_attempt.tv_sec ;
		timestamps[i][tid].lock_time.tv_nsec = start_lock_attempt.tv_nsec*1e-9;
#endif

		random_r(arg, &r);
		service_time = exp_rand(SERVICE_MIU,r);
		service_time_sum[tid] += service_time;

		if(clock_gettime(CLOCK_MONOTONIC,&start) != 0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		if(clock_gettime(CLOCK_MONOTONIC,&end_cs) != 0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}

		while(time_diff(&start,&end_cs) < service_time)
		{
			if(clock_gettime(CLOCK_MONOTONIC,&end_cs)!=0)
			{
				printf("get monotonic time error\n");
				exit(-1);
			}
		}
		assert(time_diff(&start,&end_cs) >= service_time);
#ifdef USE_SPINLOCK
		pthread_spin_unlock(&spinlock);
#else
		pthread_mutex_unlock (&sum_lock);
#endif

#ifdef DEBUG
		if(clock_gettime(CLOCK_MONOTONIC,&end_cs)!=0)
		{
			printf("get end_cs monotonic time error\n");
			exit(-1);
		}
		//printf("%lf %d\n",(end_cs.tv_sec + end_cs.tv_nsec*1e-9),tid);
		if((time_diff(&end_cs,&get_lock)) - service_time > 0.0001)
			printf("Actual service section time and generated service time difference: %lf\n",(time_diff(&end_cs,&get_lock) - service_time));
		//printf("releaselock %lf %d\n",(end_cs.tv_sec + end_cs.tv_nsec*1e-9),tid);
		timestamps[i][tid].unlock_time.tv_sec = end_cs.tv_sec ;
		timestamps[i][tid].unlock_time.tv_nsec = end_cs.tv_nsec*1e-9;
#endif

		//Store the difference of the start and end critical section time
		if(i > NUM_HEAD_LOOP) //don't include the first NUM_HEAD_LOOP critical section time
			time_in_cs[tid] +=  time_diff(&start_lock_attempt,&end_cs);
		assert(time_diff(&start_lock_attempt,&end_cs) >= service_time);
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
		service_time_sum[i] = 0;
	}
	if(argc < 2)
	{
		printf("Usage: ./test arrival_lambda\n");
		exit(-1);
	}
	arrival_lambda = atoi(argv[1]);

	for(i = 0; i < NUM_EXPERIMENTS; i++)
	{
		sum = 0;
		struct timeval tim; double t1,t2; //for timing one experiment
		gettimeofday(&tim, NULL);
		t1 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
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
		gettimeofday(&tim, NULL);
		t2 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;

		for(t = 0; t < NUM_THREADS; t++)
		{
			printf("Response time for thread %d: %lf\n",t,time_in_cs[t]/(double)(NUM_LOOP-NUM_HEAD_LOOP));
			sum += time_in_cs[t];
		}
		printf("Average response time for lambda %d: %f\n",arrival_lambda,sum/((double)NUM_THREADS*(NUM_LOOP - NUM_HEAD_LOOP)));
		for(t = 0; t < NUM_THREADS; t++)
		{
			printf("The serivce time for thread: %d: %lf\n",t,service_time_sum[t]/(double)(NUM_LOOP-NUM_HEAD_LOOP)); 
		}
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
		}
	}
	pthread_exit(NULL);
}
