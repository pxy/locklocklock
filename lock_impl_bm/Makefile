CC         	= gcc
CFLAGS     	= -Wall -std=c99 -pedantic -D_XOPEN_SOURCE=700 `pkg-config --cflags --libs glib-2.0 gsl`
LDLIBS     	= -pthread `pkg-config --libs glib-2.0 gsl`
RM		    = rm

all: 		consprod
#use implicit rules

#specific deps
consprod: clh.o

clean:
	$(RM) -f *.o consprod
