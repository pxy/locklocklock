#include <stdio.h>
#include "tsc.h"
#define CPU_FREQ 2270000000
#define N_LOOP 10000000
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
	printf("Read_tsc time avg: %f cycles\n",(end_time - start_time)/(float)N_LOOP);


	start_time = read_tsc_fenced();
	for (i = 0; i < N_LOOP; i++)
	{
	    read_tsc_fenced();
	}
	end_time = read_tsc_fenced();
	printf("Read_tsc_fenced time avg: %f cycles\n",(end_time - start_time)/(float)N_LOOP);


	start_time = read_tsc_fenced();
	for (i = 0; i < N_LOOP; i++)
	{
	    read_tsc_p();
	}
	end_time = read_tsc_fenced();
	printf("Read_tsc_fenced time avg: %f cycles\n",(end_time - start_time)/(float)N_LOOP);


	start_time = read_tsc_fenced();
	for (i = 0; i < N_LOOP; i++)
	{
	    read_tsc_cpuid();
	}
	end_time = read_tsc_fenced();
	printf("Read_tsc_cpuid time avg: %f cycles\n",(end_time - start_time)/(float)N_LOOP);


	return 0;
}
