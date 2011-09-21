#ifndef _TSC_H
#define _TSC_H

#include <stdint.h>

#if defined(__x86_64__)

static inline uint64_t __attribute__((always_inline))
read_tsc()
{
   uint64_t tsc;
   __asm__ ("rdtsc\n"
	 "shl $32, %%rdx\n"
	 "or %%rdx, %%rax"
	 : "=a"(tsc)
	 :
	 : "rdx");
   return tsc;
}

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


static inline uint64_t  __attribute__((always_inline))
read_tsc_fenced()
{
   uint64_t tsc;
   __asm__ ("lfence\n"
	 "rdtsc\n"
	 "shl $32, %%rdx\n"
	 "or %%rdx, %%rax"
	 : "=a"(tsc)
	 :
	 : "rdx");
   return tsc;
}

__inline__ uint64_t 
read_tsc_cpuid() 
{
    uint32_t lo, hi;
    __asm__ __volatile__ (      // serialize
    "xorl %%eax,%%eax \n        cpuid"
    ::: "%rax", "%rbx", "%rcx", "%rdx");
    /* We cannot use "=A", since this would use %rax on x86_64 */
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return (uint64_t)hi << 32 | lo;
}


#else

#error Unsupported architecture

#endif

#endif
