/* must be defined here */
#define NCLASS 2
#define NLGS   1
#define NTHREADS 8

int init_pos[] = {0,0};

int nlocks_lg[] = {1};


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
	15000,
	250000,
    },
    {
	15000,
	350000,
    }
};

double rout2[NCLASS][NLGS][NLGS] = {
    {
	{1},
    },{
	{1},
    }
};


