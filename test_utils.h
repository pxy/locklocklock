
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

typedef struct
{
	int thread_id;
	struct random_data* rand_states;
	int arrival_lambda;
	pthread_spinlock_t *spinlock_ptr;
} thread_params;

typedef struct
{
	uint64_t ts;
	int id;
} timestamp;

extern int sqr (int x);
extern int is_on_same_node(int i, int j, int n, int left, int right);
extern void set_affinity(pthread_t thread, const cpu_set_t *cpuset);
extern int cmp_timestamp(const void *ptr1, const void *ptr2);
extern void print_mtx (int m, int n, float mtx[m][n], int l, int t, int r, int b, int type);

#endif
