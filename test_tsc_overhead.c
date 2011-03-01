#include <stdio.h>
#include "tsc.h"
#define CPU_FREQ 2270000000
#define N_LOOP 1000000
#define S_TO_N 1000000000
int main(void)
{
	int i;
	uint64_t start_time = read_tsc();
	for(i = 0; i < N_LOOP; i++)
	{
		read_tsc();
	}
	uint64_t end_time = read_tsc();
	printf("Time diff: %f nano second\n",(end_time - start_time)*S_TO_N/(float)(CPU_FREQ*N_LOOP));
	return 0;
}
