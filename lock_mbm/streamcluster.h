

/* must be defined here */
#define NCLASS 2
#define NLGS   1
#define NTHREADS 5

int class_threads[] = { 0, 1,1,1,1};

/* only lock routing */
double routs[NCLASS][NLGS][NLGS] = {
    {
	{1.}
    },{    
	{1.}
    }
};

/* service times for locks AND local computation times */
int servs[NCLASS][2*NLGS][2*NLGS] = {
    {
	{60000, 60000},
	{250000, 250000}  // local
    },
    {
	{60000, 60000},
	{350000, 350000}  // local
    }
};
