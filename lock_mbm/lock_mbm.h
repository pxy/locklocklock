#ifndef _LOCK_MBM_H
#define _LOCK_MBM_H

#define SHA1_LEN 20

#define handle_error_en(en, msg) \
               do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

typedef struct {
    int think_t;
    int service_t;
    int lock_group;
} class_t;


int set_lock_impl(int);
void *run(void *);
void *pick_lock(class_t *class_info);
static void lock(void *, int);
static void unlock(void *, int);
static int  get_next_d (int);
static int  get_next_e (int);
static void p_unlock(void *lock, int thread);
static void p_lock(void *lock, int thread);
void save_arr(void);


typedef void (* lock_fun_ptr_t)(void *lock, int thread);
typedef int  (* rng_fun_ptr_t)(int rate);


typedef struct {
    lock_fun_ptr_t init;
    lock_fun_ptr_t lock;
    lock_fun_ptr_t unlock;
} lock_impl_t;

typedef struct {
    pthread_t thread;
    int tid;
    class_t class_info;
    rng_fun_ptr_t next_f;
} thread_args_t;

typedef struct {
    uint64_t try;
    uint64_t acq;
    uint64_t rel;
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
