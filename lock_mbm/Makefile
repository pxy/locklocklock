CC         	= gcc
CFLAGS     	= -Wall -std=c99 -pedantic `pkg-config --cflags --libs glib-2.0 gsl`
LDLIBS     	= -pthread `pkg-config --libs glib-2.0 gsl`
RM		= rm

all: 		lock_mbm clh_test
#use implicit rules

#specific deps
lock_mbm: clh.o j_util.o

clh_test: j_util.o clh.o

clean:
	$(RM) -f *.o lock_mbm clh_test
