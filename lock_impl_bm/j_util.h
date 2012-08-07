#ifndef _J_UTIL_H
#define _J_UTIL_H

#include <errno.h>
#include <stdio.h>

#define E(c)					\
    do {					\
	int _c = (c);				\
	if (_c < 0) {				\
	    fprintf(stderr, "E: %s: %d: %s\n",	\
		    __FILE__, __LINE__, #c);	\
	}					\
    } while (0)


#define E_en(c)					\
    do {					\
	int _c = (c);				\
	if (_c < 0) {				\
	    errno = _c; perror("E_en");		\
	}					\
    } while (0)
    
#define E_0(c)					\
    do {					\
	if ((c) == NULL) {			\
	    perror("E_0");			\
	}					\
    } while (0)

extern pid_t gettid (void);

extern void pin (pid_t t, int cpu);



#endif
