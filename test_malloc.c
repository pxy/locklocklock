#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#define N_THREADS 8 
#define N 100000000 
int main(int argc, char* argv[])
{
	uint64_t *b;
	int n = atoi(argv[1]);
	printf("Size of uint64_t is : %d\n",sizeof(uint64_t));
	uint64_t a[n*N_THREADS];
	printf("No error initilizing the array. The size of the array: %lu bytes\n",n*N_THREADS*sizeof(uint64_t));
	;
	b = (uint64_t *)malloc(sizeof(uint64_t)*n*N_THREADS);
	if(b == NULL)
		printf("malloc error.\n");
	else
		printf("malloc of %lu bytes successful\n",sizeof(uint64_t)*n*N_THREADS);
}
