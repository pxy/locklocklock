#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "test_utils.h"

int 
sqr (int x)
{
    return x*x;
}

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

void
print_mtx (int m, int n, float mtx[m][n], int l, int t, int r, int b, int type)
{
	float sum = 0.0;
	int cnt = 0;
	for(int i = t; i < b; i++)
	{
		for(int j = l; j < r; j++)
		{
			if (i == j) 
			{
				cnt++;
				continue;
			}
			sum += mtx[i][j];
			printf("%d -> %d: %.3f, ",i,j, mtx[i][j]);
		}
		printf("\n");
	}
	if (type)
	{
		printf("avg: %.2f\n", sum / ((float)(b - t)*(r - l) - (float)cnt));
	}
	else
	{
		printf("sum: %.2f\n", sum);
	}
}
