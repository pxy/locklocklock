#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "test_utils.h"

int 
cmp_timestamp(const void *ptr1, const void *ptr2)
{
	timestamp *a = (timestamp *)ptr1;
	timestamp *b = (timestamp *)ptr2;
	if(a->ts > b->ts)
		return 1;
	else if(a->ts == b->ts)
		return 0;
	else
		return -1;
}

void 
set_affinity(pthread_t thread, const cpu_set_t *cpuset)
{
	int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), cpuset);
	if (s != 0)
	{
		perror("set affinity error \n");
		exit(-1);
	}
}

int 
is_on_same_node(int i, int j, int n, int left, int right)
{
        if(right == 0)
		return 1;
	if((i < left && j >= left) || (i >= left && j < left))
		return 0;
	else
		return 1;
}
