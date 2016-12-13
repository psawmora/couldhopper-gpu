#include <stdlib.h>
#include <stdio.h>

#include "cuda_decoder_device.h"
extern "C" {
#include "smpp_pdu_struct_cuda.h"
}

__global__ void launchDecode() {
  printf("RUNNING\n");
  testPrint();
}

void cudaTest() {

    launchDecode <<< 1, 512 >>> ();
    cudaDeviceSynchronize();
    printf("Cuda SMPP decoding completed.");
    return;
}
