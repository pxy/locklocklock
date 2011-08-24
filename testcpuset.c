#define _GNU_SOURCE
 #include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <errno.h>


// #define DEBUG

#ifdef DEBUG
#define dprintf(...) printf(__VA_ARGS__)
#else
#define dprintf(...) /* nothing */
#endif

#define E(c) do {                                       \
		int _c = (c);									\
		if (_c < 0) {									\
			fprintf(stderr, "Error: %s: %d: %s\n",		\
					__FILE__, __LINE__, #c);			\
			exit(EXIT_FAILURE);							\
		}												\
    } while (0)

int
main(int argc, char *argv[])
{
	cpu_set_t *cpusetp;
	size_t size;
	int num_cpus, cpu;
	
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <num-cpus>\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	num_cpus = atoi(argv[1]);
	
	cpusetp = CPU_ALLOC(num_cpus);
	if (cpusetp == NULL) {
		perror("CPU_ALLOC");
		exit(EXIT_FAILURE);
	}
	
	size = CPU_ALLOC_SIZE(num_cpus);
	
	CPU_ZERO_S(size, cpusetp);
	for (cpu = 0; cpu < num_cpus; cpu += 2)
		CPU_SET_S(cpu, size, cpusetp);

	printf("CPU_COUNT() of set:    %d\n", CPU_COUNT_S(size, cpusetp));

	CPU_FREE(cpusetp);
	exit(EXIT_SUCCESS);
}
