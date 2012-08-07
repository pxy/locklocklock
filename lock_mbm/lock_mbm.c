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
#include <assert.h>
#include <malloc.h>
#include <libclh.h>

#include "lock_mbm.h"
#include "j_util.h"

/*
 * Select microbenchmark settings
 */

//#include "streamcluster.h"
#include "bodytrack.h"
//#include "bodytrack_eq_serv.h"
//#include "fluidanimate.h"

#if !defined (__linux__) || !defined(__GLIBC__)
#error "OS must be linux."
#endif

#define DEBUG 0
#define NDEBUG  // no asserts
#define PIN_THREADS
//#define DETERM_ROUT 1

// kalkyl freq
#define MHZ 2270
#define GHZ 2.270

// tintin freq
//#define MHZ 3000
//#define GHZ 3.000

/* global vars */
uint64_t end;
/* one array of timestamp data for each thread */
time_info_t *current_ts;
thread_args_t *threads;
#ifdef PIN_THREADS
int *cpu_map;
#endif

/* runtime of experiment in s */
int runtime;
int loop_limit = 0;

/* set up settings for microbenchmark, service times etc. */

#if ! (defined(NCLASS) && defined(NLGS))
#error parameters undefined!
#endif 

int class_nths[NCLASS];
extern int nlocks_lg[NLGS];
extern int routs[][NLGS][2];
extern double rout2[][NLGS][NLGS];
extern int servs[][2*NLGS];

#define ROUT_IDX(mx, row, col) ((mx)[(col) + 2 * (row)])

#define NEXT_CS_T(mu, tid) get_next_e(mu, tid)
#define NEXT_L_T(mu, tid) get_next_e(mu, tid)

#define LOCK_IMPL(lockp, tid) clh_lock((clh_lock_t *) lockp, tid)
#define UNLOCK_IMPL(lockp, tid) clh_unlock((clh_lock_t *) lockp, tid)
//#define LOCK_IMPL(lockp, tid) p_lock(lockp, tid)
//#define UNLOCK_IMPL(lockp, tid) p_unlock(lockp, tid)

/* current state */
extern int init_pos[NCLASS];
lockgroup_t lockgroups [NLGS];
class_t classes [NCLASS];
int nthreads;
int multiplier;
gsl_ran_discrete_t *routs2[NCLASS][NLGS];

void
init_lg(lockgroup_t *lg, int idx) {
    lg->nlocks = nlocks_lg[idx];
    E_0(lg->threads = (int *)malloc(sizeof(int)*nthreads));
    lg->l = (clh_lock_t *)memalign(64, sizeof(clh_lock_t) * nlocks_lg[idx]);
    for (int i = 0; i < nlocks_lg[idx]; i++) {
	clh_init(&lg->l[i], nthreads);
    }
    printf("lg %d initialised with %d locks.\n", idx, lg->nlocks);
}

void
fini_lg(lockgroup_t *lg) {
    free(lg->threads);
    for (int i = 0; i < lg->nlocks; i++) {
	clh_destroy(&lg->l[i]);
    }
    free(lg->l);
}

void
init_mb() {
    nthreads = 0;
    for (int i = 0; i < NCLASS; i++) {
	nthreads += class_nths[i];
	for (int j = 0; j < NLGS; j++) {
	    routs2[i][j] = gsl_ran_discrete_preproc(NLGS, rout2[i][j]);
	}
	classes[i].rout = routs2[i];
    }

    printf("running with %d threads.\n", nthreads);
    
    for (int i = 0; i < NLGS; i++) {
	init_lg(&lockgroups[i], i);
    }

    for (int i = 0; i < NCLASS; i++) {
	classes[i].rout_m = (int *) routs[i];
	classes[i].serv_m = servs[i];
	// inflate lock holding time
	for (int j = 0; j < NLGS; j++) {
	    classes[i].serv_m[2*j] *= multiplier;
	}
    }
}

void
fini_mb() {

    for (int i = 0; i < NCLASS; i++) {
	for (int j = 0; j < NLGS; j++) {
	    gsl_ran_discrete_free (routs2[i][j]);
	}
    }
    for (int i = 0; i < NLGS; i++) {
	fini_lg(&lockgroups[i]);
    }
}

int
local_t (thread_args_t *ta, int lidx)
{
    int p = ta->pos;
    assert(p < NLGS);
    assert(ta->class_info.serv_m[2*p] > 0);
    return NEXT_L_T(ta->class_info.serv_m[2*p+1], ta->tid);
}

int
cs_t (thread_args_t *ta, int lidx)
{
    int p = ta->pos;
    assert(ta->class_info.serv_m[2*p] > 0);
    return NEXT_CS_T(ta->class_info.serv_m[2*p], ta->tid);
}

int
select_lock (thread_args_t *ta)
{
    int p = ta->pos;
    // if it's time to switch lock

#ifndef DETERM_ROUT
    // jump randomly according to probs
    ta->pos = gsl_ran_discrete (ta->rng, ta->class_info.rout[ta->pos]);
    dprintf("pos cnt : %d\n", ta->pos_cnt);
    
#else
    (ta->pos_cnt)--;
    if (0 >= ta->pos_cnt) {
	ta->pos = ROUT_IDX(ta->class_info.rout_m,p,0);
	ta->pos_cnt  = ROUT_IDX(ta->class_info.rout_m,ta->pos,1);
    }
    // otherwise, stay
#endif
    dprintf("new pos : %d\n", ta->pos);
    return ta->pos;
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
    int pause_lc, pause_cs;
    /* number of lock accesses */
    int cnt = 0;
    int lidx;


#ifdef PIN_THREADS
    pin(gettid(), cpu_map[args->tid]);
#endif
    // calibrate for the first local computation time
    current_ts[args->tid].rel = read_tsc_p();

    do {
        // think locally
        start = read_tsc_p();
	lidx = select_lock(args); //where are we going?
	/* compute times here, prob. of local computation
	 * being long is higher */
        pause_lc = local_t(args, lidx);
        pause_cs = cs_t(args, lidx);
	while(read_tsc_p() - start < pause_lc)
            ;
        /* BEGIN critical section */
	lock (lidx, args);
        start = read_tsc_p();
	cnt++;
	while(read_tsc_p() - start < pause_cs)
            ;
	unlock (lidx, args->tid);
	/* END critical section */
	dprintf("Tid %d finished lock unlock seq.\n", args->tid);
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
    int * cp;
    /* Read options/settings */
    while ((opt = getopt(argc, argv, "c:t:l:m:")) != -1) {
        switch (opt) {
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
	case 'c':
	    for (cp = class_nths; cp < &class_nths[NCLASS] &&
		     (*cp = atoi(strsep(&optarg, ","))) != -1;) {
		if (*cp != -1)
		    cp++;
	    }
	    break;
	case 'm':
	    multiplier = atoi(optarg);
	    if(multiplier <= 0){
                fprintf(stderr, "invalid argument.\n");
		errexit = 1;
	    }
	    break;
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

    init_mb();

    current_ts = (time_info_t *) malloc (sizeof(time_info_t)*nthreads);
    threads    = (thread_args_t *) malloc (sizeof(thread_args_t)*nthreads);

#ifdef PIN_THREADS
    cpu_map    = (int*) malloc (sizeof(int)*nthreads);
#endif

    /***** SETUP *****/
    // stop experiment when time end (in cycles) is reached
    end = runtime * MHZ * 1000000L + read_tsc_p();

    
    dprintf("end: %"PRId64"\n", end);
    

#ifdef PIN_THREADS
    // define how the threads should be pinned to cores
    for (int i = 0; i<nthreads; i++) 
	cpu_map[i] = i;
#endif

    int cur_class = 0;
    int class_tid = 0;
    int64_t start = read_tsc_p();
    /* SPAWN */
    for (int i = 0; i < nthreads; i++) {
	if (class_nths[cur_class] <= class_tid) {
	    cur_class++;
	    class_tid = 0;
	}
	class_tid++;

        thread_args_t *t = &threads[i];
        t->tid = i;
	if (NULL == (t->rng = gsl_rng_alloc(gsl_rng_mt19937))) {
	    perror("gsl_rng_alloc");
	}
	gsl_rng_set(t->rng, read_tsc_p()+i); // salt it!!?

	t->l_ts = (l_ts_t *)malloc(sizeof(l_ts_t) * NLGS);
	
	for (int j = 0; j < NLGS; j++) {
	    t->l_ts[j].lc_t = 0;
	    t->l_ts[j].s_t = 0;
	    t->l_ts[j].cs_t = 0;
	    t->l_ts[j].c = 0;
	}

	/* which class does this thread belong to? */
        t->class_info = classes[cur_class];
	t->pos = init_pos[cur_class];

        if (pthread_create(&t->thread, NULL, &run, t))
            return (EXIT_FAILURE);
    }

    /* JOIN */
    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i].thread, NULL);

    int64_t end = read_tsc_p();

    fprintf(stderr, "# total time: %ld\n", end - start);

    int old_i = 0;
    int class = 0;
    printf("queue_%d = [[", nthreads);
    for (int i = 0; i < nthreads; i++) {
	for (int j = 0; j < NLGS; j++) {
	    if (i + 1 - old_i >= class_nths[class]) {

		int cnt = threads[old_i].l_ts[j].c;
		if (cnt > 0) {
		    printf("%"PRId64", ", threads[old_i].l_ts[j].cs_t/cnt);
		} else {
		    printf("0, ");
		}
		if (j == NLGS - 1) {
		    printf("],\n");
		    old_i = i + 1;
		    class ++;
		    if (i + 1 < nthreads) printf("[");
		}

	    }
	    else {
		threads[old_i].l_ts[j].c += threads[i].l_ts[j].c;
		threads[old_i].l_ts[j].cs_t += threads[i].l_ts[j].cs_t;
	    }
	}
    }
    printf("]\n");

    old_i = 0;
    class = 0;
    printf("serv_%d = [[", nthreads);
    for (int i = 0; i < nthreads; i++) {
	for (int j = 0; j < NLGS; j++) {
	    if (i + 1 - old_i >= class_nths[class]) {

		int cnt = threads[old_i].l_ts[j].c;
		if (cnt > 0) {
		    printf("%"PRId64", ", threads[old_i].l_ts[j].lc_t/cnt);
		    printf("%"PRId64", ", threads[old_i].l_ts[j].s_t/cnt);
		}
		else {
		    printf("0, 0, ");
		}
		if (j == NLGS - 1) {
		    printf("],\n");
		    old_i = i + 1;
		    class ++;
		    if (i + 1 < nthreads) printf("[");
		}

	    }
	    else {
		threads[old_i].l_ts[j].lc_t += threads[i].l_ts[j].lc_t;
		threads[old_i].l_ts[j].s_t += threads[i].l_ts[j].s_t;
	    }
	}

    }
    printf("]\n");


/* free */
    

    /* CLEANUP */
    free(current_ts);
    
    for (int i = 0; i < nthreads; i++) {
	gsl_rng_free(threads[i].rng);
	free(threads[i].l_ts);

    }
    free(threads);
#ifdef PIN_THREADS
    free(cpu_map);
#endif 
    fini_mb();
    return(EXIT_SUCCESS);
}


void *
pick_lock_lg(lockgroup_t *lg, thread_args_t *ta)
{
    //FIX select idx
    int idx = 0;
    if (lg->nlocks > 1) {
	idx = gsl_rng_uniform_int (ta->rng, lg->nlocks);
	//idx = gsl_ran_gaussian_ziggurat (ta->rng, 5);
	//idx = fmin(fmax(0, idx+lg->nlocks/2), lg->nlocks - 1);
	dprintf("idx inside lockgroup: %d\n", idx);
	
    }
    lg->threads[ta->tid] = idx;
    return (void *) &lg->l[idx];
}

void *
pick_unlock_lg(lockgroup_t *lg, int tid)
{
    return (void *) &lg->l[lg->threads[tid]];
}

void
lock(int lg_idx, thread_args_t *ta) 
{
    uint64_t start, end;
    // LG
    dprintf("lock lg_idx: %d\n", lg_idx);
    void *l = pick_lock_lg(&lockgroups[lg_idx], ta);
    start = read_tsc_p();
    LOCK_IMPL(l, ta->tid);
    end = read_tsc_p();
    ta->l_ts[lg_idx].cs_t += end - start;
    ta->l_ts[lg_idx].c++;
    ta->l_ts[lg_idx].lc_t += start - current_ts[ta->tid].rel;

// FIX for serv time
    current_ts[ta->tid].acq = read_tsc_p();
}

void
unlock(int lg_idx, int tid)
{
    uint64_t end;
    // LG
    dprintf("unlock lg_idx: %d\n", lg_idx);
    void *l = pick_unlock_lg(&lockgroups[lg_idx], tid);
    UNLOCK_IMPL(l, tid);
    end = read_tsc_p();
    threads[tid].l_ts[lg_idx].s_t += end - current_ts[tid].acq;

    current_ts[tid].rel = read_tsc_p();

}

static int
get_next_e(int mu,int tid) {
    return (int) gsl_ran_exponential (threads[tid].rng, (double) mu);
}

static int
get_next_u(int mu, int tid) {
    return (int) 0;
}

int
get_next_d(int mu, int tid) {
    return (int) mu;
}

/* pthread mutex wrappers */
static void
p_lock(void *l, int tid)
{
    E_en(pthread_mutex_lock((pthread_mutex_t *)l) != 0);
}

static void
p_unlock(void *l, int tid)
{
    E_en(pthread_mutex_unlock((pthread_mutex_t *)l) != 0);
}
/* END pthread mutex wrappers */





