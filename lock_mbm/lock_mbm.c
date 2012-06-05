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
#include "clh.h"
#include "lock_mbm.h"
#include "j_util.h"
//#include "streamcluster.h"
#include "bodytrack.h"


#if !defined (__linux__) || !defined(__GLIBC__)
#error "OS must be linux."
#endif

#define DEBUG 0
#define NDEBUG  // no asserts
//#define SAVE_TS 0
#define PIN_THREADS

// kalkyl freq
#define MHZ 2270
#define GHZ 2.270

// tintin freq
//#define MHZ 3000
//#define GHZ 3.000

/* global vars */
char type;
char *rng_type;
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

/* runtime of experiment in s */
int runtime;
int loop_limit = 0;

/* set up settings for microbenchmark, service times etc. */

#if ! (defined(NCLASS) && defined(NTHREADS) && defined(NLGS))
#error parameters undefined!
#endif 

extern int class_threads[];
extern int routs[][NLGS][2];
extern int servs[][2*NLGS];

#define ROUT_IDX(mx, row, col) ((mx)[(col) + 2 * (row)])

#define NEXT_CS_T(mu, tid) get_next_e(mu, tid)
#define NEXT_L_T(mu, tid) get_next_e(mu, tid)
#define LOCK_IMPL(lockp, tid) clh_lock(lockp, tid)
#define UNLOCK_IMPL(lockp, tid) clh_unlock(lockp, tid)

//#define LOCK_IMPL(lockp, tid) p_lock(lockp, tid)
//#define UNLOCK_IMPL(lockp, tid) p_unlock(lockp, tid)


/* current state */
extern int pos[NTHREADS];
lockgroup_t lockgroups [NLGS];
class_t classes [NCLASS];
int nthreads = NTHREADS;

void
init_lg(lockgroup_t *lg) {
    //E_0(lg->l = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t)));
    //E(pthread_mutex_init(lg->l, NULL));
    lg->l = (clh_lock_t *)memalign(64, sizeof(clh_lock_t));
    clh_init(lg->l, nthreads);
    dprintf("lg initialised.\n");
}

void
fini_lg(lockgroup_t *lg) {
    //E(pthread_mutex_destroy(lg->l));
    clh_destroy(lg->l);
    free(lg->l);
}

void
init_mb() {
    for (int i = 0; i < NLGS; i++) {
	init_lg(&lockgroups[i]);
    }
    for (int i = 0; i < NCLASS; i++) {
	classes[i].rout_m = (int *) routs[i];
	classes[i].serv_m = servs[i];
	// inflate lock holding time
	for (int j = 0; j < NLGS; j++) {
	    classes[i].serv_m[2*j] *= 3;
	}
    }
}

void
fini_mb() {
    for (int i = 0; i < NLGS; i++) {
	fini_lg(&lockgroups[i]);
    }
}

int
local_t (thread_args_t *ta, int lidx)
{
    int p = pos[ta->tid];
    assert(ta->class_info.serv_m[2*p] > 0);
    return NEXT_L_T(ta->class_info.serv_m[2*p+1], ta->tid);
}

int
cs_t (thread_args_t *ta, int lidx)
{
    int p = pos[ta->tid];
    assert(ta->class_info.serv_m[2*p] > 0);
    return NEXT_CS_T(ta->class_info.serv_m[2*p], ta->tid);
}

int
select_lock (thread_args_t *ta)
{
    int p = pos[ta->tid];
    dprintf("pos cnt : %d\n", ta->pos_cnt);
    (ta->pos_cnt)--;

    // if it's time to switch lock
    if (0 >= ta->pos_cnt) {
	pos[ta->tid] = ROUT_IDX(ta->class_info.rout_m,p,0);
	ta->pos_cnt  = ROUT_IDX(ta->class_info.rout_m,pos[ta->tid],1);
    }
    // otherwise, stay
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

    // calibrate for the first local computation time
    current_ts[args->tid].rel = read_tsc_p();

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

    /* Read options/settings */
    while ((opt = getopt(argc, argv, "t:l:")) != -1) {
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

#ifdef SAVE_TS
    t_times    = (GArray **) malloc (sizeof (GArray *)*nthreads);
    for (int i = 0; i < nthreads; i++)
        t_times[i] = g_array_sized_new(FALSE, 
				       TRUE, 
				       sizeof (time_info_t), 
				       runtime*5000);
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

    int64_t start = read_tsc_p();
    /* SPAWN */
    for (int i = 0; i < nthreads; i++) {
        thread_args_t *t = &threads[i];
        t->tid = i;
	if (NULL == (threads[i].rng = gsl_rng_alloc(gsl_rng_mt19937))) {
	    perror("gsl_rng_alloc");
	}
	gsl_rng_set(threads[i].rng, read_tsc_p()+i); // salt it!!?

	threads[i].l_ts = (l_ts_t *)malloc(sizeof(l_ts_t) * NLGS);
	
	for (int j = 0; j < NLGS; j++) {
	    threads[i].l_ts[j].lc_t = 0;
	    threads[i].l_ts[j].s_t = 0;
	    threads[i].l_ts[j].cs_t = 0;
	}
	
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

#ifdef SAVE_TS
    save_arr();
#endif
    printf("queue time : \n");
    for (int i = 0; i < nthreads; i++) {
	for (int j = 0; j < NLGS; j++) {
	    int cnt = threads[i].l_ts[j].c;
	    if (cnt > 0)
		printf("%"PRId64" ", threads[i].l_ts[j].cs_t/cnt);
	    else
		printf("0 ");
	}
	printf("\n");
    } 
    printf("service time : \n");
    for (int i = 0; i < nthreads; i++) {
	for (int j = 0; j < NLGS; j++) {
	    int cnt = threads[i].l_ts[j].c;
	    if (cnt > 0) {
		printf("%"PRId64" ", threads[i].l_ts[j].lc_t/cnt);
		printf("%"PRId64" ", threads[i].l_ts[j].s_t/cnt);
	    }
	    else
		printf("0 0 ");
	}
	printf("\n");
    } 


    printf("total contention : ");
    uint64_t tot = 0;
    for (int i = 0; i < nthreads; i++) {
	for (int j = 0; j < NLGS; j++) {
	    tot += threads[i].l_ts[j].cs_t;
	}
    } 
    printf("%"PRId64" per thread\n", tot/nthreads);


    printf("Count : \n");
    for (int i = 0; i < nthreads; i++) {
	for (int j = 0; j < NLGS; j++) {
	    printf("%"PRId64" ", threads[i].l_ts[j].c);
	}
	printf("\n");
    }
    

/* free */

    /* CLEANUP */
    free(current_ts);

    for (int i = 0; i < nthreads; i++) {
	gsl_rng_free(threads[i].rng);
	free(threads[i].l_ts);
#ifdef SAVE_TS
	g_array_free(t_times[i], FALSE);
	free(t_times);
#endif
    }

    free(threads);


#ifdef PIN_THREADS
    free(cpu_map);
#endif 
    fini_mb();
    return(EXIT_SUCCESS);
}


void
lock(int lg_idx, int tid) 
{
    uint64_t start, end;
    start = read_tsc_p();
    LOCK_IMPL(lockgroups[lg_idx].l, tid);
    end = read_tsc_p();
    threads[tid].l_ts[lg_idx].cs_t += end - start;
    threads[tid].l_ts[lg_idx].c++;
    threads[tid].l_ts[lg_idx].lc_t += start - current_ts[tid].rel;

#ifdef SAVE_TS    
    current_ts[tid].try = start;
    current_ts[tid].acq = end;
#endif
// FIX for serv time
    current_ts[tid].acq = read_tsc_p();

}

void
unlock(int lg_idx, int tid)
{
    uint64_t end;
    
    UNLOCK_IMPL(lockgroups[lg_idx].l, tid);
    end = read_tsc_p();
    threads[tid].l_ts[lg_idx].s_t += end - current_ts[tid].acq;

    current_ts[tid].rel = read_tsc_p();
#ifdef SAVE_TS
    current_ts[tid].lock = lg_idx;
    g_array_append_val (t_times[tid], current_ts[tid]);
#endif


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
            fprintf (fp_try, "%ld %d %d\n", ti.try, i, ti.lock);
            fprintf (fp_acq, "%ld %d %d\n", ti.acq, i, ti.lock);
            fprintf (fp_rel, "%ld %d %d\n", ti.rel, i, ti.lock);
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



