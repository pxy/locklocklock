CC         	= gcc
CFLAGS     	= -Wall -std=c99 -pedantic
LDLIBS       	= -lrt -lm -lnuma -pthread
RM		= rm

all: 		test test_numa_mem test_numa_cache test_numa_comb_v2

#use implicit rules

clean:
	$(RM) -f *.o test
