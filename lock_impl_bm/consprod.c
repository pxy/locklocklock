#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <glib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <unistd.h>
#include <string.h>

#include "clh.h"
#include "../tsc.h"
#include "consprod.h"

#if !defined (__linux__) || !defined(__GLIBC__)
#error "This stuff only works on Linux!"
#endif

#define handle_error_en(en, msg) \
               do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)



// kalkyl freq
#define MHZ 2270
#define NTHREADS 4
//#define USE_THREADPINNING




void (* lock_impl)(int thread);
void (* unlock_impl)(int thread);
int (* next_f)(int mu);

int set_lock_impl(int);
void *run(void *);
static void lock(int);
static void unlock(int);
static int get_next_d (int);
static int get_next_e (int);
static void p_unlock(int thread);
static void p_lock(int thread);
void save_arr(void);

GArray *t_times[NTHREADS];

char type;
char rng_type;
gsl_rng *rng;
uint64_t end;
time_info_t current_ts[NTHREADS];
thread_args_t threads[NTHREADS];
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

#ifdef USE_THREADPINNING
int s;	  
cpu_set_t cpuset[NTHREADS];
int thread_cpu_map[NTHREADS] = {0,2,4,6};
#endif

int runtime;


int main(int argc, char *argv[]){

    class_t classes[2];
    int opt;
    int errexit = 0;
    extern char *optarg;
    extern int optind, optopt;

    /* default time gen: exponential */
    next_f = &get_next_e;
    rng_type = 'e';


    /* Read options/settings */
    
    while ((opt = getopt(argc, argv, "c:t:d")) != -1) {
        switch (opt) {
        case 'c':
            if (set_lock_impl(atoi(optarg)))
                errexit = 1;
            break;
        case 't':
            runtime = atoi(optarg);
            if(runtime <= 0){
                fprintf(stderr,
                        "Invalid argument '%s'. Duration of experiment "
                        "must be positive.\n",
                        optarg);
                errexit = 1;
            }
            break;
        case 'd':
            next_f = &get_next_d;
            rng_type = 'd';
            break;
            //case 'h':
            //usage(stdout, argv[0]);
            //exit(EXIT_SUCCESS);
            
        case ':':
        case '?':
            errexit = 1;
            break;

        default:
            puts("aborting");
            abort();
        }
    } // while

    if (errexit) {
        exit(EXIT_FAILURE);
    }

    /* INIT */
    clh_init(NTHREADS);
    if (NULL == (rng = gsl_rng_alloc(gsl_rng_mt19937))) {
        perror("gsl_rng_alloc");
    }
    gsl_rng_set(rng, read_tsc_p());

    for (int i = 0; i < NTHREADS; i++)
        t_times[i] = g_array_sized_new(FALSE, TRUE, sizeof (time_info_t), runtime*5000);


    /***** SETUP *****/
    // stop experiment when time end (in cycles) is reached
    end = runtime * MHZ * 1000000L + read_tsc_p();
    // two different classes
    // unfair values class1: 6, 120, class2: 800, 180;
    classes[0].think_t = MHZ*6;
    classes[0].service_t = MHZ*120;

    classes[1].think_t = MHZ*800;
    classes[1].service_t = MHZ*180;


    /* SPAWN */
    for (int i = 0; i < NTHREADS; i++) {
        thread_args_t *t = &threads[i];
        t->tid = i;
        t->class_info = classes[i % 2];
        if (pthread_create(&t->thread, NULL, &run, t))
            return (EXIT_FAILURE);

#ifdef USE_THREADPINNING	  
        CPU_ZERO(&cpuset[i]);
        CPU_SET(thread_cpu_map[i], &cpuset[i]);
        s += pthread_setaffinity_np(t->thread, sizeof(cpu_set_t), &cpuset[i]);
        if (s != 0)
            handle_error_en(s, "set affinity error\n")`;
#endif

    }

    /* JOIN */
    for (int i = 0; i < NTHREADS; i++)
        pthread_join(threads[i].thread, NULL);

    save_arr();

    /* CLEANUP */
    for (int i = 0; i < NTHREADS; i++)
        g_array_free(t_times[i], FALSE);

    gsl_rng_free(rng);
    clh_fini();
    
    return(EXIT_SUCCESS);
}


/*
 * Each thread runs this.
 */
void* run(void *_args){
    thread_args_t * args = (thread_args_t*)_args;
    uint64_t start;
    int pause;
    int cnt = 0;
    do {
        // think locally
        start = read_tsc_p();
        pause = next_f(args->class_info.think_t);
        while(read_tsc_p() - start < pause)
            ;
        
        /* critical section */
        lock (args->tid);
        start = read_tsc_p();
        pause = next_f(args->class_info.service_t);
        while(read_tsc_p() - start < pause)
            ;
        unlock (args->tid);
        cnt++;

    } while (read_tsc_p() < end);
    
    printf("tid: %d, cnt: %d\n", args->tid, cnt);
    return NULL;
}

/* lock/unlock functions, including timing */
static void 
lock(int tid)
{
    current_ts[tid].try = read_tsc_p();
    lock_impl(tid);
    current_ts[tid].acq = read_tsc_p();
}

static void
unlock(int tid)
{
    unlock_impl(tid);
    current_ts[tid].rel = read_tsc_p();
    g_array_append_val (t_times[tid], current_ts[tid]);
}

static int
get_next_e(int mu) {
    return (int) gsl_ran_exponential (rng, (double) mu);
}

static int
get_next_d(int mu) {
    return (int) mu;
}

/* pthread mutex wrappers */
static void
p_lock(int thread)
{
    if (pthread_mutex_lock(&mutex) != 0) {
        perror("pthread_mutex_lock failed");
    }
}

static void
p_unlock(int thread)
{
    if (pthread_mutex_unlock(&mutex) != 0) {
        perror("pthread_mutex_lock failed");
    }
}
/* pthread mutex wrappers */

/* Selects which lock implementation is to be used, (p)threads (0) or
 * (q)ueue lock (1)
 */
int
set_lock_impl (int i)
{
    if (0 == i) {
        lock_impl = &p_lock;
        unlock_impl = &p_unlock;
        type = 'p';
    } else if (1 == i) {
        lock_impl = &clh_lock;
        unlock_impl = &clh_unlock;
        type = 'q';
    } else {
        return -1;
    }
    return 0;
}



/* print collected data to files */
void
save_arr()
{
    time_info_t ti;
    FILE *fp_try, *fp_acq, *fp_rel;    
    char prefix[20];
    char stry[30], sacq[30], srel[30];

    snprintf (prefix, sizeof prefix, "%s_%d_%c%c_%d", "output", NTHREADS, type, rng_type, runtime);
    
    strcpy (stry, prefix);
    strcat (stry, "_try.dat");
    if (NULL == (fp_try = fopen (stry, "w"))) {
        perror("failed open file");
        return;
    }

    strcpy (sacq, prefix);
    strcat (sacq, "_acq.dat");
    if (NULL == (fp_acq = fopen (sacq, "w"))) {
        perror("failed open file");
        return;
    }

    strcpy (srel, prefix);
    strcat (srel, "_rel.dat");
    if (NULL == (fp_rel = fopen (srel, "w"))) {
        perror("failed open file");
        return;
    }

    for (int i = 0; i < NTHREADS; i++) {
        for (int j = 0; j < t_times[i]->len; j++) {
            ti = g_array_index(t_times[i], time_info_t, j);
            fprintf (fp_try, "%ld %d 0\n", ti.try, i);
            fprintf (fp_acq, "%ld %d 0\n", ti.acq, i);
            fprintf (fp_rel, "%ld %d 0\n", ti.rel, i);
        }
    }
    fclose(fp_try);
    fclose(fp_acq);
    fclose(fp_rel);
}
