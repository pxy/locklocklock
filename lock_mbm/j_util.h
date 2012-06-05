#ifndef _J_UTIL_H
#define _J_UTIL_H

#include <errno.h>
#include <stdio.h>
#include <stdint.h>


#define dprintf(...)                            \
    if (DEBUG)					\
	printf(__VA_ARGS__)                        


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


#if defined(__x86_64__)

static inline uint64_t __attribute__((always_inline))
read_tsc_p()
{
   uint64_t tsc;
   __asm__ __volatile__ ("rdtscp\n"
	 "shl $32, %%rdx\n"
	 "or %%rdx, %%rax"
	 : "=a"(tsc)
	 :
	 : "%rcx", "%rdx");
   return tsc;
}

static inline void __attribute__((always_inline))
asm_atomic_inc_int32(int32_t *var)
{
    __asm__ ("lock incl %0;"
	     : "+m" (*var));
}


#else
#error Unsupported architecture
#endif


extern pid_t gettid (void);

extern void pin (pid_t t, int cpu);



#endif
