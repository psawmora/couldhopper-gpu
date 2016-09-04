
#include <stdlib.h>
#include <stdio.h>
extern "C" {
 #include "smpp_pdu_struct.h"
}

__global__ void simpleCudaTest() {
    


}

int test()
{
	simpleCudaTest<<<1,1>>>();
	printf("Cuda Simple Test Completed");
	return 0;
}
