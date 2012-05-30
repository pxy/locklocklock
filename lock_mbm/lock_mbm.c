#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <glib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include "clh.h"
#include "lock_mbm.h"
#include "j_util.h"
#include "stream_cluster.h"


#if !defined (__linux__) || !defined(__GLIBC__)
#error "OS must be linux."
#endif

#define DEBUG 0
#define SAVE_TS 1
#define PIN_THREADS

// kalkyl freq
#define MHZ 2270
#define GHZ 2.270

// tintin freq
//#define MHZ 3000
//#define GHZ 3.000

void (* lock_impl)  (void *l);
void (* unlock_impl)(void *l);
int  (* next_cs_f)  (int mu);
int  (* next_l_f)   (int mu);

void *run(void *);
void save_arr(void);
void print_proc_cx(void);

char type;
char *rng_type;
gsl_rng *rng;
uint64_t end;

/* one array of timestamp data for each thread */
#ifdef SAVE_TS
GArray **t_times;
#endif
time_info_t *current_ts;
thread_args_t *threads;

#ifdef PIN_THREADS
int *cpu_map;
#endif

#ifdef COND_THRU
volatile int in_cs = 0;
#endif

/* runtime of experiment in s */
int runtime;
int loop_limit = 0;

#if ! (defined(NCLASS) && defined(NTHREADS) && defined(NLGS))
#error define!
#endif 

extern int class_threads[];
extern double routs[][NLGS][NLGS];
extern int servs[][2*NLGS][2*NLGS];

#define M_IDX(mx, row, col) ((mx)[(col) + NLGS * (row)])
//HACK
#define M_IDX_2(mx, row, col) ((mx)[(col) + 2*NLGS * (row)])

/* current state */
int pos[NTHREADS];
lockgroup_t lockgroups [NLGS];
class_t classes [NCLASS];
int nthreads = NTHREADS;

void
init_lg(lockgroup_t *lg) {
    //E_0(lg->l = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t)));
    //E(pthread_mutex_init(lg->l, NULL));
}

void
init_mb() {
    init_lg(&lockgroups[0]);
    //init_lg(&lockgroups[1]);

    for (int i = 0; i < NCLASS; i++) {
	classes[i].rout_m = (double *) &routs[i];
	classes[i].serv_m = (int *) &servs[i];
    }
}

void
fini_mb() {
}

int
local_t (thread_args_t *ta, int lidx)
{
    int p = pos[ta->tid];
    return next_l_f(M_IDX_2(ta->class_info.serv_m,2*p+1,lidx));
}

int
cs_t (thread_args_t *ta, int lidx)
{
    int p = pos[ta->tid];
    return next_l_f(M_IDX(ta->class_info.serv_m,2*p,0));
}

int
select_lock(thread_args_t *ta)
{
    int p = pos[ta->tid];
    // FIX
    // generate int with prob from rout
    //return ta->class_info.rout_m[p][0];
    pos[ta->tid] = (p + 1) % NLGS;
    return pos[ta->tid];
}

/***************************************************************** 
 * The actual microbenchmark.
 * Each thread runs this function.
 */
void * 
run(void *_args)
{
    thread_args_t * args = (thread_args_t*)_args;
    uint64_t start;
    int pause;
    /* number of lock accesses */
    int cnt = 0;
    int lidx;

#ifdef PIN_THREADS
    pin(gettid(), cpu_map[args->tid]);
#endif
    do {
        // think locally
        start = read_tsc_p();
	lidx = select_lock(args); //where are we going?
        pause = local_t(args, lidx);
	while(read_tsc_p() - start < pause)
            ;
        /* BEGIN critical section */
	lock (lidx, args->tid);
        start = read_tsc_p();
        pause = cs_t(args, lidx);
	cnt++;
	while(read_tsc_p() - start < pause)
            ;

	unlock (lidx, args->tid);
	/* END critical section */
    } while (read_tsc_p() < end);
    /* end of measured execution */
    return NULL;
}


/* parse options and setups experiment */
int 
main(int argc, char *argv[]){

    int opt;
    int errexit = 0;
    extern char *optarg;
    extern int optind, optopt;

    int d_type;
    /* default time gen: exponential */
    next_cs_f = &get_next_e;
    next_l_f  = &get_next_e;
    rng_type  = "ee";
    /* Read options/settings */
    while ((opt = getopt(argc, argv, "c:t:d:l:")) != -1) {
        switch (opt) {
        case 'c':
	    if (set_lock_impl(atoi(optarg)) < 0)
                errexit = 1;
            break;
	case 'l':
	    if ((loop_limit = atoi(optarg)) <= 0)
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

	  d_type = atoi(optarg);
	  if (d_type < 0 || d_type > 1){
	    fprintf(stderr,
		    "Invalid argument '%s'.",
		    optarg);
	    errexit = 1;

	  }
	  else if (d_type == 0) {
            next_cs_f = &get_next_d;
	    next_l_f  = &get_next_d;
            rng_type  = "dd";
	  } else {
	    next_cs_f = &get_next_d;
	    next_l_f  = &get_next_e;
	    rng_type  = "de";
	  }
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
	fprintf(stderr, "Aborting...\n");
        exit(EXIT_FAILURE);
    }

    /* INIT */
    current_ts = (time_info_t *) malloc (sizeof(time_info_t)*nthreads);
    threads    = (thread_args_t *) malloc (sizeof(thread_args_t)*nthreads);

#ifdef PIN_THREADS
    cpu_map    = (int*) malloc (sizeof(int)*nthreads);
#endif

    init_mb();
    clh_init(nthreads);

#ifdef SAVE_TS
    t_times    = (GArray **) malloc (sizeof (GArray *)*nthreads);
    for (int i = 0; i < nthreads; i++)
        t_times[i] = g_array_sized_new(FALSE, 
				       TRUE, 
				       sizeof (time_info_t), 
				       runtime*5000);
#endif


    if (NULL == (rng = gsl_rng_alloc(gsl_rng_mt19937))) {
        perror("gsl_rng_alloc");
    }
    gsl_rng_set(rng, read_tsc_p());

    /***** SETUP *****/
    // stop experiment when time end (in cycles) is reached
    end = runtime * MHZ * 1000000L + read_tsc_p();

    
    dprintf("end: %"PRId64"\n", end);
    

#ifdef PIN_THREADS
    // define how the threads should be pinned to cores
    for (int i = 0; i<nthreads; i++) 
	cpu_map[i] = i;
#endif

    int64_t start = read_tsc_p();
    /* SPAWN */
    for (int i = 0; i < nthreads; i++) {
        thread_args_t *t = &threads[i];
        t->tid = i;

	/* which class does this thread belong to? */
        t->class_info = classes[class_threads[i]];

        if (pthread_create(&t->thread, NULL, &run, t))
            return (EXIT_FAILURE);
    }

    /* JOIN */
    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i].thread, NULL);

    int64_t end = read_tsc_p();

    fprintf(stderr, "# total time: %ld\n", end - start);
    dprintf("end time: %"PRId64"\n", end);

    save_arr();
    
    

/* free */

    /* CLEANUP */
#ifdef SAVE_TS
    for (int i = 0; i < nthreads; i++)
        g_array_free(t_times[i], FALSE);
    free(t_times);
#endif

    free(current_ts);
    gsl_rng_free(rng);

    free(threads);


#ifdef PIN_THREADS
    free(cpu_map);
#endif 
    clh_fini();
    fini_mb();
    return(EXIT_SUCCESS);
}


void
lock(int lg_idx, int tid) 
{
    current_ts[tid].try = read_tsc_p();
    //lock_impl(lockgroups[lg_idx].l);
    lock_impl((void *)&tid);
    current_ts[tid].acq = read_tsc_p();
}

void
unlock(int lg_idx, int tid)
{
    //unlock_impl(lockgroups[lg_idx].l);
    unlock_impl((void *) &tid);
    current_ts[tid].rel = read_tsc_p();
#ifdef SAVE_TS
    g_array_append_val (t_times[tid], current_ts[tid]);
#endif

}

static int
get_next_e(int mu) {
    return (int) gsl_ran_exponential (rng, (double) mu);
}

int
get_next_d(int mu) {
    return (int) mu;
}

/* pthread mutex wrappers */
static void
p_lock(void *l)
{
    E_en(pthread_mutex_lock((pthread_mutex_t *)l) != 0);
}

static void
p_unlock(void *l)
{
    E_en(pthread_mutex_unlock((pthread_mutex_t *)l) != 0);
}
/* END pthread mutex wrappers */

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

    snprintf (prefix, sizeof prefix, 
	      "%s_%d_%c%s", "output", 
	      nthreads, type, rng_type);
    
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
#ifdef SAVE_TS
    for (int i = 0; i < nthreads; i++) {
        for (int j = 0; j < t_times[i]->len; j++) {
            ti = g_array_index(t_times[i], time_info_t, j);
            fprintf (fp_try, "%ld %d 0\n", ti.try, i);
            fprintf (fp_acq, "%ld %d 0\n", ti.acq, i);
            fprintf (fp_rel, "%ld %d 0\n", ti.rel, i);
        }
    }
#endif


    fclose(fp_try);
    fclose(fp_acq);
    fclose(fp_rel);
}


void
print_proc_cx ()
{
    FILE *filePtr;
    char sBuf[65536];
    char *path = "/proc/self/status";
    E_0(filePtr = fopen(path, "r"));

    int n;
    while ((n = fread(sBuf, 1, sizeof(sBuf), filePtr))) {
	fwrite(sBuf, 1, n, stderr);
    }

    fclose (filePtr);
}



