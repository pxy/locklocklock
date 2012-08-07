
#ifndef _CONSPROD_H
#define _CONSPROD_H

typedef void (* lock_fun_ptr_t)(int thread);
typedef int  (* rng_fun_ptr_t)(int rate);

int set_lock_impl(int);
static void lock(int);
static void unlock(int);
static int  get_next_d (int);
static int  get_next_e (int);
static void p_unlock(int thread);
static void p_lock(int thread);


typedef struct {
    int think_t;
    int service_t;
} class_t;

typedef struct {
    lock_fun_ptr_t init;
    lock_fun_ptr_t lock;
    lock_fun_ptr_t unlock;
} lock_impl_t;

typedef struct {
    pthread_t thread;
    int tid;
    int t_inc_per_cnt;
    class_t class_info;
    rng_fun_ptr_t next_f;
} thread_args_t;

typedef struct {
    uint64_t try;
    uint64_t acq;
    uint64_t rel;
} time_info_t;


#define FOR(arr, iter, type)                    \
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
