#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#define NUM_THREADS 4
#define E 2.71828
#define SECOND_TO_USECOND 1000000
#define ARRIVAL_LAMDA 20 //arrival rate = 1/average inter arrival time
#define SERVICE_MIU 100 //service rate = 1/average service time
#define NUM_EXPERIMENTS 1
#define NUM_LOOP 100
//#define DEBUG

#define LINUX // activate linux specific code

//declare a global mutex
//an array to store lock contention time
double time_in_cs[NUM_THREADS];
double ex_time[NUM_THREADS]; 
double experiment_time;
pthread_mutex_t sum_lock;
int count_contention;

typedef struct
{
	int thread_id;
//	double arrival_time;
}thread_params;

double exp_rand(float miu)
{
	//srand(time(NULL)); //generate random numbers
	double ran = (-log((double)rand()/(double)RAND_MAX)/miu);
	//printf("In exp_ran: %lf\n",ran);
	return ran;
}
double uniform()
{
	return (double)rand()/(double)RAND_MAX;
}
void *work(void *thread_arg)
{
	srand(time(NULL));
	thread_params *thread_param = (thread_params *)thread_arg;
	double service_time;
	//double next_arrival_time;
	double arrival_time;
	int tid = thread_param->thread_id;
	struct timeval tim;// = cs_time[tid];
	double t1,t2;
	struct timeval ex_tim;// = cs_time[tid];
	double t3,t4; //for measuring the execution time of a thread
	int i; 
	//int arrival_time = (int)thread_param->arrival_time;
	gettimeofday(&ex_tim, NULL);
	t3 = ex_tim.tv_sec*SECOND_TO_USECOND + ex_tim.tv_usec;
	for(i = 0; i < NUM_LOOP; i++)
	{
		//if(i == 0)
		arrival_time = exp_rand(ARRIVAL_LAMDA);
		//else
			//arrival_time = next_arrival_time;
#ifdef DEBUG
		printf("Thread %d arrival time: %lf\n",tid,arrival_time);
#endif
		usleep((int)arrival_time*SECOND_TO_USECOND);
#ifdef DEBUG
		printf("Thread %d trying to get the lock\n",tid);
#endif
		gettimeofday(&tim, NULL);
		t1 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
		//next_arrival_time = exp_rand(ARRIVAL_LAMDA); //generate the next arrival time before being served.
		pthread_mutex_lock (&sum_lock);
#ifdef DEBUG
		printf("Thread %d got the lock\n",tid);
#endif
		service_time = exp_rand(SERVICE_MIU);
		//service_time = 0.125;
#ifdef DEBUG
		printf("Service time of Thread %d: %lf.\n",tid,service_time);
#endif
		usleep(service_time*SECOND_TO_USECOND);
		pthread_mutex_unlock (&sum_lock);
		gettimeofday(&tim, NULL);

		t2 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
#ifdef DEBUG
		printf("Thread %d released the lock\n",tid);
		printf("Thread %d spent %.6lf microseconds from trying to get the lock to releasing the lock.\n", tid,t2-t1);
		printf("Thread %d spent %.6lf microseconds in the contention.\n", tid,t2-t1-service_time*SECOND_TO_USECOND);
#endif
		time_in_cs[tid] += t2 - t1; //- service_time*SECOND_TO_USECOND; 
		assert(t2-t1 > service_time*SECOND_TO_USECOND);
		//At the end of the service, check if the time spend in the critical section is already longer than the generated arrival time
		//if(t2 - t1 > next_arrival_time*SECOND_TO_USECOND)
			//next_arrival_time = 0;
	}
	gettimeofday(&ex_tim, NULL);
	t4 = ex_tim.tv_sec*SECOND_TO_USECOND + ex_tim.tv_usec;
	ex_time[tid] = t4 - t3;
	pthread_exit(NULL);
}
int main (int argc, char *argv[])
{
	pthread_t threads[NUM_THREADS];
	int rc;
	int t;
	double sum;
	count_contention = 0;
	thread_params para[NUM_THREADS];
	//double inter_arrival_time[NUM_THREADS];
	srand(time(NULL));
	int i= 0;
	double contention_time = 0;
	#ifdef LINUX
	cpu_set_t cpuset[NUM_THREADS]; //for setting affinity of the threads
	#endif
	int s;
	for(i = 0; i < NUM_EXPERIMENTS; i++)
	{
		sum = 0;
		struct timeval tim; double t1,t2; //for timing one experiment
		gettimeofday(&tim, NULL);
		t1 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
		/*
		   for(t=0; t<NUM_THREADS; t++)
		   {
		//inter_arrival_time[t] = exp_rand(ARRIVAL_LAMDA);
		inter_arrival_time[t] = uniform();
		}
		 */
		/*
#ifdef DEBUG
for(t=0; t<NUM_THREADS; t++)
printf("printing inter arrival time %d: %lf\n",t,inter_arrival_time[t]);
#endif
		 */
		//para[0].arrival_time = inter_arrival_time[0];
		/*for(t=1; t<NUM_THREADS; t++)
		  {
		  para[t].arrival_time = para[t-1].arrival_time + inter_arrival_time[t];
		  }*/
		pthread_mutex_init(&sum_lock, NULL);
		for(t = 0; t < NUM_THREADS; t++){
			para[t].thread_id = t;
			time_in_cs[t] = 0;
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
		}
#endif
		/* Wait on the other threads */
		for(t=0; t<NUM_THREADS; t++)
		{
			pthread_join(threads[t], NULL);
		}
		pthread_mutex_destroy(&sum_lock);
		gettimeofday(&tim, NULL);
		t2 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
		printf("The experiment took %.6lf microseconds.\n", t2-t1);
#ifdef DEBUG
		for(t=0; t<NUM_THREADS; t++){
			printf("Time in cs: for thread %d: %lf \n",t,time_in_cs[t]);
			printf("Execution time for thread %d: %lf \n",t,ex_time[t]); 
		}
#endif

		for(t = 0; t < NUM_THREADS; t++)
			sum += time_in_cs[t];
		printf("**********************************Average waiting time: %lf\n",sum/(double)(NUM_THREADS*NUM_LOOP));
		//contention_time  += sum/(double)NUM_THREADS;
		//printf("Contention percentage: %lf\n", (sum/(double)NUM_THREADS)/(t2-t1));
	}
	//printf("Average waiting time of all experimetns: %lf\n",contention_time/(double)NUM_EXPERIMENTS);
	pthread_exit(NULL);
}
