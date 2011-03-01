#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>

#define N 1000000
double time_diff(struct timespec *start,struct timespec *end)
{
	        return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec)*1e-9;
}

int main(void)
{
	struct timespec start,end;
	int i;
	if(clock_gettime(CLOCK_MONOTONIC,&start) !=0)
	{
		printf("get monotonic start_waiting time error\n");
		exit(-1);
	}
	for(i = 0; i < N; i++)
	{
		if(clock_gettime(CLOCK_MONOTONIC,&end) !=0)
		{
			printf("get monotonic start time error\n");
			exit(-1);
		}
	}
	if(clock_gettime(CLOCK_MONOTONIC,&end) !=0)
	{
		printf("get monotonic start_waiting time error\n");
		exit(-1);
	}
	printf("Time diff: %lf\n",time_diff(&start,&end));
	return 0;

}

