CC         	= gcc
CFLAGS     	= -Wall -std=c99 -pedantic
LDLIBS     	= -lrt -lm -lnuma -pthread
RM		    = rm

all: 		test test_numa_mem test_numa_cache test_numa_comb_jl test_numa_comb_v2 test_tsc_overhead test_numa_mutex
#use implicit rules

#specific deps
test_numa_comb_jl: test_utils.o
test_numa_mutex: test_utils.o

clean:
	$(RM) -f *.o test
