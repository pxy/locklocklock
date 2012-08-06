CC         	= gcc
CFLAGS     	= -Wall -std=c99 -pedantic `pkg-config --cflags --libs glib-2.0 gsl` -I$(HOME)/local/include
LDLIBS     	= -pthread `pkg-config --libs glib-2.0 gsl` -L$(HOME)/local/lib -lclh


all: 		lock_mbm clh_test thread_placement
#use implicit rules

#specific deps
lock_mbm: j_util.o

clh_test: j_util.o

thread_placement: j_util.o

clean:
	$(RM) *.o lock_mbm clh_test thread_placement
