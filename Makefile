CC         = gcc
CFLAGS     = -Wall -std=c99 -pedantic
LDLIBS       = -lrt -lm -lnuma -pthread


all: test test_t_v test_queue test_numa_mem test_numa_cache test_numa_comb test_numa_comb_v2
#use implicit rules

#test: test.c test_timestamp_version.c test_queue_lock.c test_numa_mem.c test_numa_cache.c test_numa_comb.c test_numa_comb_v2.c
#	gcc -Wall -pthread -lrt -lm -o test test.c
#	gcc -Wall -pthread -lrt -lm -o test_t_v test_timestamp_version.c
#	gcc -std=gnu99 -Wall -pthread -lrt -lm -o test_queue test_queue_lock.c
#	gcc -pthread -lnuma -o test_numa_mem test_numa_mem.c
#	gcc -pthread -lnuma -o test_numa_cache test_numa_cache.c
#	gcc -pthread -lnuma -o test_numa_comb test_numa_comb.c
#	gcc -pthread -lnuma -o test_numa_comb_v2 test_numa_comb_v2.c
clean:
	rm -f *.o test
