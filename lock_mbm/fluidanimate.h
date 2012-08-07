/* must be defined here */
#define NCLASS 1
#define NLGS   1

int nlocks_lg[] = {100};
    
int init_pos[] = {0};

/* only lock routing */
int routs[NCLASS][NLGS][2] = {
    {
	{0, 1},
    }
};

/* service times for locks AND local computation times */
int servs[NCLASS][2*NLGS] = {
    {
//	13000, 7700000
	5000, 100000
    },
};

double rout2[NCLASS][NLGS][NLGS] = {
    {
	{ 0.}
    }
    
};
