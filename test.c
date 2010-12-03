#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#define NUM_THREADS 4
#define PRNG_BUFSZ 32
#define E 2.71828
#define SECOND_TO_USECOND 1000000
#define ARRIVAL_LAMDA 20 //arrival rate = 1/average inter arrival time
#define SERVICE_MIU 100 //service rate = 1/average service time
#define NUM_EXPERIMENTS 1
#define NUM_LOOP 100
//#define DEBUG

#define LINUX // activate linux specific code

double time_in_cs[NUM_THREADS];
double ex_time[NUM_THREADS]; 
double experiment_time;
pthread_mutex_t sum_lock;

typedef struct
{
	int thread_id;
	struct random_data* rand_states;
	int arrival_lambda;
}thread_params;

//Given a randomly generated rand, convert a uniform distribution into an exponenetial distribution
double exp_rand(float miu,int rand)
{
	double ran = (-log((double)rand/(double)RAND_MAX)/miu);
	return ran;
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
	
	struct timeval tim,ex_tim;
	double t1,t2,t3,t4;

	gettimeofday(&ex_tim, NULL);
	t3 = ex_tim.tv_sec*SECOND_TO_USECOND + ex_tim.tv_usec;
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
		usleep((int)arrival_time*SECOND_TO_USECOND);
#ifdef DEBUG
		printf("Thread %d trying to get the lock\n",tid);
#endif

		gettimeofday(&tim, NULL);
		t1 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
		//start of the critical section
		pthread_mutex_lock (&sum_lock);
#ifdef DEBUG
		printf("Thread %d got the lock\n",tid);
#endif
 		random_r(arg, &r);
		//printf("random number r in service time is : %d.\n",r);
		service_time = exp_rand(SERVICE_MIU,r);
#ifdef DEBUG
		printf("Service time of Thread %d: %lf.\n",tid,service_time);
#endif
		usleep(service_time*SECOND_TO_USECOND);
		pthread_mutex_unlock (&sum_lock);
		//End of the critical section
		gettimeofday(&tim, NULL);

		t2 = tim.tv_sec*SECOND_TO_USECOND + tim.tv_usec;
#ifdef DEBUG
		printf("Thread %d released the lock\n",tid);
		printf("Thread %d spent %.6lf microseconds from trying to get the lock to releasing the lock.\n", tid,t2-t1);
		printf("Thread %d spent %.6lf microseconds in the contention.\n", tid,t2-t1-service_time*SECOND_TO_USECOND);
#endif
		time_in_cs[tid] += t2 - t1; //- service_time*SECOND_TO_USECOND; 
		assert(t2-t1 > service_time*SECOND_TO_USECOND);
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
	double contention_time = 0;
	#ifdef LINUX
	cpu_set_t cpuset[NUM_THREADS]; //for setting affinity of the threads
	#endif
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
		pthread_mutex_init(&sum_lock, NULL);
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
#ifdef DEBUG
		printf("The experiment took %.6lf microseconds.\n", t2-t1);
		for(t=0; t<NUM_THREADS; t++){
			printf("Time in cs: for thread %d: %lf \n",t,time_in_cs[t]);
			printf("Execution time for thread %d: %lf \n",t,ex_time[t]); 
		}
#endif

		for(t = 0; t < NUM_THREADS; t++)
			sum += time_in_cs[t];
		printf("Average response time for lambda = %d: %lf\n",arrival_lambda,sum/(double)(NUM_THREADS*NUM_LOOP));
		//contention_time  += sum/(double)NUM_THREADS;
		//printf("Contention percentage: %lf\n", (sum/(double)NUM_THREADS)/(t2-t1));
	}
	//printf("Average waiting time of all experimetns: %lf\n",contention_time/(double)NUM_EXPERIMENTS);
	pthread_exit(NULL);
}
