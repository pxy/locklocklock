#ifndef _TSC_H
#define _TSC_H

#include <stdint.h>

#if defined(__x86_64__)

static inline uint64_t
read_tsc()
{
   uint64_t tsc;
   asm ("rdtsc\n"
	 "shl $32, %%rdx\n"
	 "or %%rdx, %%rax"
	 : "=a"(tsc)
	 :
	 : "rdx");
   return tsc;
}

static inline uint64_t
read_tsc_fenced()
{
   uint64_t tsc;
   asm ("lfence\n"
	 "rdtsc\n"
	 "shl $32, %%rdx\n"
	 "or %%rdx, %%rax"
	 : "=a"(tsc)
	 :
	 : "rdx");
   return tsc;
}

#elif defined(__i386__)

static inline uint64_t
read_tsc()
{
   uint32_t eax, edx;
   asm ("rdtsc\n"
	 : "=a"(eax), "=d"(edx));
   return ((uint64_t)edx << 32) | eax;
}

static inline uint64_t
read_tsc_fenced()
{
   uint32_t eax, edx;
   asm ("lfence\n"
	 "rdtsc\n"
	 : "=a"(eax), "=d"(edx));
   return ((uint64_t)edx << 32) | eax;
}

#else

#error Unsupported architecture

#endif

#endif
