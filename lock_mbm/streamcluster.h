/* must be defined here */
#define NCLASS 2
#define NLGS   1
#define NTHREADS 4

int class_threads[] = {0,1,1,1};

int pos[NTHREADS];



/* only lock routing */
int routs[NCLASS][NLGS][2] = {
    {
	{0, 1}
    },{
	{0, 1}
    }
};

/* service times for locks AND local computation times */
int servs[NCLASS][2*NLGS] = {
    {
	60000,
	250000,
    },
    {
	60000,
	350000,
    }
};
