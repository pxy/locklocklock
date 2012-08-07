#include "tsc.h"
#include <stdio.h>
int main(void)
{
	int i;
	uint64_t begin,end;
	begin =  read_tsc_p();
	for(i = 0; i < 100000000; i++)
	{
		read_tsc_p();
	}
	end =  read_tsc_p();
	printf("timestamp counter difference %u\n",end - begin);
	printf("Each read_tsc takes: %f ticks\n",(end-begin)/(double)100000000);
}
