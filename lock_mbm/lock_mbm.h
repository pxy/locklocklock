#ifndef _LOCK_MBM_H
#define _LOCK_MBM_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "clh.h"

typedef struct {
    int *rout_m;
    int *serv_m;
} class_t;

typedef struct {
    uint64_t lc_t;
    uint64_t cs_t;
    uint64_t s_t;
    int c;
} l_ts_t;


typedef struct {
    //lock_t l[];
    //pthread_mutex_t *l;
    clh_lock_t *l;
    int serv_t;
} lockgroup_t;

void print_proc_cx(void);
void *run(void *);
static void lock(int lg_idx, int tid);
static void unlock(int lg_idx, int tid);
static int  get_next_d (int, int);
static int  get_next_e (int, int);
static void p_unlock(void *lock, int tid);
static void p_lock(void *lock, int tid);
void save_arr(void);

typedef struct {
    void *lock;
    unsigned int id;
} lock_t;


typedef struct {
    pthread_t thread;
    int tid;
    class_t class_info;
    int pos_cnt;
    gsl_rng *rng;
    l_ts_t *l_ts;
} thread_args_t;

typedef struct {
    uint64_t try;
    uint64_t acq;
    uint64_t rel;
    int lock;
} time_info_t;

#define FOR_I(arr, iter, type)			\
    for (int i = 0;                             \
         i < arr->len && iter = g_array_index(arr, type, i);    \
         i++)

static inline void __attribute__((always_inline))
atomic_inc_int32(int32_t *var)
{
    __asm__ ("lock incl %0;"
	     : "+m" (*var));
}

static inline void __attribute__((always_inline))
atomic_dec_int32(int32_t *var)
{
    __asm__ ("lock decl %0;"
	     : "+m" (*var));
}

#endif
