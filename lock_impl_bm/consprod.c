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

#include "clh.h"
#include "../tsc.h"
#include "consprod.h"
#include "j_util.h"

#if !defined (__linux__) || !defined(__GLIBC__)
#error "OS must be linux."
#endif

// kalkyl freq
#define MHZ 2270
#define GHZ 2.270

// tintin freq
//#define MHZ 3000
//#define GHZ 3.000

//#define SLOPE 1 // constant service time
#define SLOPE 1000 // service time slope increase by 1 microsec

#define PIN_THREADS
//#define COND_THRU


void (* lock_impl)  (int thread);
void (* unlock_impl)(int thread);
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
time_info_t *total_ts;
thread_args_t *threads;
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

#ifdef PIN_THREADS
int *cpu_map;
//int thread_cpu_map[NTHREADS] = {0,2,4,6};
#endif

#ifdef COND_THRU
volatile int in_cs = 0;
#endif

/* Number of threads to be used in experiment.
 * Has to be an exponent of 2, if using classes */
int nthreads = 2;

/* runtime of experiment in s */
int runtime;

/* used as a basic checksum of correctness */
static volatile int ctrl;


/* local think time, given in microsec */
int think_c0 = 6000, think_c1 = 800;
/* critical section time, given in microsec */
int serv_c0 = 120000, serv_c1 = 180;
int loop_limit = 0;



/***************************************************************** 
 * The actual microbenchmark.
 * Each thread runs this function.
 */
void* 
run(void *_args)
{
    thread_args_t * args = (thread_args_t*)_args;
    uint64_t start;
    int pause;
    /* number of lock accesses */
    int cnt = 0;

#ifdef PIN_THREADS
    pin(gettid(), cpu_map[args->tid]);
#endif
    do {
        // think locally
        start = read_tsc_p();
        pause = next_l_f(args->class_info.think_t);
        while(read_tsc_p() - start < pause)
            ;

        /* BEGIN critical section */
        lock (args->tid);
	ctrl++; // already protected
        start = read_tsc_p();
        pause = next_cs_f(args->class_info.service_t);
	//pause += ctrl*args->t_inc_per_cnt;
	while(read_tsc_p() - start < pause)
            ;
	cnt++;
        unlock (args->tid);
	/* END critical section */
	
    } while (read_tsc_p() < end);
    /* end of measured execution */

    int l_t = total_ts[args->tid].try / cnt;
    int q_t = total_ts[args->tid].acq / cnt;
    int s_t = total_ts[args->tid].rel / cnt;

/*
 * tid
 * lock count
 * interarrival time
 * queue time
 * service time
 */
    printf("%d %d %d %d %d\n", args->tid, cnt, l_t, q_t, s_t);

//    print_proc_cx();

    return NULL;
}




/* parse options and setups experiment */
int main(int argc, char *argv[]){

    class_t classes[2];
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
    
    while ((opt = getopt(argc, argv, "n:s:c:t:w:d:l:")) != -1) {
        switch (opt) {
	case 'n':
	    if((nthreads = atoi(optarg)) <= 0)
                errexit = 1;
            break;
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
        case 'w':
            if (2 != sscanf(optarg, "%d,%d", &think_c0, &think_c1))
                errexit = 1;
            break;
	case 's':
	    // comma separated service time for 2 classes
	    if (2 != sscanf(optarg, "%d,%d", &serv_c0, &serv_c1))
	        errexit = 1;
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
    
    /* allocate */
#ifdef SAVE_TS
    t_times    = (GArray **) malloc (sizeof (GArray *)*nthreads);
#endif
    current_ts = (time_info_t *) malloc (sizeof(time_info_t)*nthreads);
    total_ts   = (time_info_t *) malloc (sizeof(time_info_t)*nthreads);
    threads    = (thread_args_t *) malloc (sizeof(thread_args_t)*nthreads);
    //memset(&total_ts, 0, (sizeof(time_info_t)*nthreads));
#ifdef PIN_THREADS
    cpu_map    = (int*) malloc (sizeof(int)*nthreads);
#endif 
    for (int i = 0; i < nthreads; i++) {
	total_ts[i].try = 0;
	total_ts[i].acq = 0;
	total_ts[i].rel = 0;
    }


    clh_init(nthreads);
    if (NULL == (rng = gsl_rng_alloc(gsl_rng_mt19937))) {
        perror("gsl_rng_alloc");
    }
    gsl_rng_set(rng, read_tsc_p());

#ifdef SAVE_TS
    for (int i = 0; i < nthreads; i++)
        t_times[i] = g_array_sized_new(FALSE, 
				       TRUE, 
				       sizeof (time_info_t), 
				       runtime*5000);
#endif

    /***** SETUP *****/
    // stop experiment when time end (in cycles) is reached
    end = runtime * MHZ * 1000000L + read_tsc_p();
    // two different classes
    classes[0].think_t = think_c0;
    classes[0].service_t = serv_c0;

    classes[1].think_t = think_c1;
    classes[1].service_t = serv_c1;

    
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
        /* how much should the service time increase
	 * per lock access */
	t->t_inc_per_cnt = SLOPE;
	/* which of the two classes does this thread
	 * belong to? */
        t->class_info = classes[i % 2];

        if (pthread_create(&t->thread, NULL, &run, t))
            return (EXIT_FAILURE);
    }

    /* JOIN */
    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i].thread, NULL);

    int64_t end = read_tsc_p();

    fprintf (stderr, "checksum value: %d\n", ctrl);
    fprintf(stderr, "# total time: %ld\n", end - start);

#ifdef SAVE_TS
    save_arr();
#endif

    /* CLEANUP */
#ifdef SAVE_TS
    for (int i = 0; i < nthreads; i++)
        g_array_free(t_times[i], FALSE);
#endif
    gsl_rng_free(rng);

/* free */
#ifdef SAVE_TS
    free(t_times);
#endif
    free(current_ts);
    free(total_ts);
    free(threads);
#ifdef PIN_THREADS
    free(cpu_map);
#endif 
    clh_fini();
    
    return(EXIT_SUCCESS);
}



/* lock/unlock functions, including timing */
void 
lock(int tid)
{
    current_ts[tid].try = read_tsc_p();
    lock_impl(tid);
    current_ts[tid].acq = read_tsc_p();
    // local comp time
    if (current_ts[tid].rel > 0) 
	total_ts[tid].try += current_ts[tid].try - current_ts[tid].rel;
}

void
unlock(int tid)
{
    unlock_impl(tid);
    current_ts[tid].rel = read_tsc_p();
#ifdef SAVE_TS
    g_array_append_val (t_times[tid], current_ts[tid]);
#endif
    // queue time
    total_ts[tid].acq += current_ts[tid].acq - current_ts[tid].try;
    // service time
    total_ts[tid].rel += current_ts[tid].rel - current_ts[tid].acq;

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
p_lock(int thread)
{
    E_en(pthread_mutex_lock(&mutex) != 0);
}

static void
p_unlock(int thread)
{
    E_en(pthread_mutex_unlock(&mutex) != 0);
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
	      "%s_%d_%c%s_%d", "output", 
	      nthreads, type, rng_type, runtime);
    
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



