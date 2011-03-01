#include "tsc.h"
int main(void)
{
	int i;
	uint64_t begin,end;
	begin =  read_tsc();
	for(i = 0; i < 1000000; i++)
	{
		read_tsc();
	}
	end =  read_tsc();
	printf("timestamp counter difference %u\n",end - begin);
	printf("Each read_tsc takes: %f ticks\n",(end-begin)/(double)1000000);
}
