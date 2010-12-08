#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#define NUM_THREADS 8
#define PRNG_BUFSZ 32
#define E 2.71828
#define SECOND_TO_USECOND 1000000
#define SERVICE_MIU 1000000 //service rate = 1/average service time
#define NUM_EXPERIMENTS 1
#define NUM_LOOP 1000
//#define DEBUG
#define USE_SPINLOCK
#define LINUX // activate linux specific code

//For the padding
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
	struct timespec start,end;
	struct timespec start_cs, end_cs;
	/*
	if(clock_gettime(CLOCK_MONOTONIC,&start)!=0)
		printf("get monotonic time error\n");
	usleep(100);
	if(clock_gettime(CLOCK_MONOTONIC,&end)!=0)
		printf("get monotonic time error\n");
	printf("time diff: %lf\n",time_diff(&start,&end));
	*/

	
	//gettimeofday(&ex_tim, NULL);
	//t3 = ex_tim.tv_sec*SECOND_TO_USECOND + ex_tim.tv_usec;
	while(!wait_flag)
	{
		;
	}
	//printf("The value of wait_flag:%d\n",wait_flag);
	for(i = 0; i < NUM_LOOP; i++)
	{
		int r;
		//generate a random number according to the arrival lambda
		random_r(arg, &r);
		arrival_time = exp_rand(arrival_lambda,r);
#ifdef DEBUG
		printf("Thread %d arrival time: %lf\n",tid,arrival_time);
#endif
		//using usleep to change the arrival time
		//usleep((int)arrival_time*SECOND_TO_USECOND);
		if(clock_gettime(CLOCK_MONOTONIC,&start) !=0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		if(clock_gettime(CLOCK_MONOTONIC,&end) != 0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		while(time_diff(&start,&end) < arrival_time)
		{
			if(clock_gettime(CLOCK_MONOTONIC,&end)!=0)
			{
				printf("get monotonic time error\n");
				exit(-1);
			}
		}
		//printf("time diff: %lf\n",time_diff(&start,&end));


#ifdef DEBUG
		printf("Thread %d trying to get the lock\n",tid);
#endif
		if(clock_gettime(CLOCK_MONOTONIC,&start_cs) !=0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}

		//gettimeofday(&tim, NULL);
		//t1 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
		//start of the critical section
#ifdef USE_SPINLOCK
		//while(pthread_spin_trylock(&spinlock)!= 0)
		pthread_spin_lock(&spinlock); 
#else
		//blocking mutex
		//pthread_mutex_lock (&sum_lock);
		//non blocking mutex

		while(pthread_mutex_trylock(&sum_lock)!= 0) 
			;
#endif

#ifdef DEBUG
		printf("Thread %d got the lock\n",tid);
#endif
		random_r(arg, &r);
		//printf("random number r in service time is : %d.\n",r);
		service_time = exp_rand(SERVICE_MIU,r);
#ifdef DEBUG
		printf("Service time of Thread %d: %lf.\n",tid,service_time);
#endif

		//usleep(service_time*SECOND_TO_USECOND);
		if(clock_gettime(CLOCK_MONOTONIC,&start) != 0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		if(clock_gettime(CLOCK_MONOTONIC,&end) != 0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		while(time_diff(&start,&end) < service_time)
		{
			if(clock_gettime(CLOCK_MONOTONIC,&end)!=0)
			{
				printf("get monotonic time error\n");
				exit(-1);
			}
		}
		//printf("service time in seconds: %f.\n",service_time);
		assert(time_diff(&start,&end) >= service_time);
#ifdef USE_SPINLOCK
		pthread_spin_unlock(&spinlock);
#else
		pthread_mutex_unlock (&sum_lock);
#endif
		//End of the critical section
		if(clock_gettime(CLOCK_MONOTONIC,&end_cs) !=0)
		{
			printf("get monotonic time error\n");
			exit(-1);
		}
		//gettimeofday(&tim, NULL);

		//t2 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
#ifdef DEBUG
		printf("Thread %d released the lock\n",tid);
		//printf("Thread %d spent %.6lf microseconds from trying to get the lock to releasing the lock.\n", tid,t2-t1);
		//printf("service time in useconds: %f.\n",service_time*SECOND_TO_USECOND);
#endif
		time_in_cs[tid] +=  time_diff(&start_cs,&end_cs);
		//assert(time_diff(&start_cs,&end_cs) >= service_time);
		if(time_diff(&start_cs,&end_cs) < service_time)
		{
			printf("Difference of start_cs and end_cs: %f\n",time_diff(&start_cs,&end_cs));
			printf("start cs time: %ld\n",(&start_cs)->tv_sec);
			printf("start cs time in nsec: %ld\n",(&start_cs)->tv_nsec);
			printf("end cs time: %ld\n",(&end_cs)->tv_sec);
			printf("end cs time in nsec: %ld\n",end_cs.tv_nsec);
			exit(-1);
		}
		/*
		   if(time_in_cs[tid] < service_time)
		   {
		   printf("time_in_cs[%d]:%f\n",tid,time_in_cs[tid]);
		//printf("start cs time: %f",(&start_cs)->tv_sec);
		//printf("end cs time: %f",(&end_cs)->tv_sec);
		exit(-1);
		}*/
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
#ifdef DEBUG
		printf("The experiment took %.6lf microseconds.\n", t2-t1);
		for(t=0; t<NUM_THREADS; t++){
			printf("Time in cs: for thread %d: %lf \n",t,time_in_cs[t]);
			printf("Execution time for thread %d: %lf \n",t,ex_time[t]); 
		}
#endif

		for(t = 0; t < NUM_THREADS; t++)
		{
			//printf("Response time for thread %d: %lf\n",t,time_in_cs[t]/(double)NUM_LOOP);
			sum += time_in_cs[t];
		}
		printf("Average response time for lambda %d: %f\n",arrival_lambda,sum/((double)NUM_THREADS*NUM_LOOP));
		//contention_time  += sum/(double)NUM_THREADS;
		//printf("Contention percentage: %lf\n", (sum/(double)NUM_THREADS)/(t2-t1));
	}
	//printf("Average waiting time of all experimetns: %lf\n",contention_time/(double)NUM_EXPERIMENTS);
	pthread_exit(NULL);
}
