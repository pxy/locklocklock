#define _GNU_SOURCE
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "j_util.h"

pid_t 
gettid(void) {
    return (pid_t) syscall(SYS_gettid);
}


void
pin(pid_t t, int cpu) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(cpu, &cpuset);
  E_en(sched_setaffinity(t, sizeof(cpu_set_t), &cpuset));
  fprintf(stderr, "%d pinned to %d\n", (int)t, cpu);
  
}
