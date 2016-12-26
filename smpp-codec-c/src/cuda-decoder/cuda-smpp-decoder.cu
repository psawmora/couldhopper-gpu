#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cuda_decoder_device.h"

//nvprof --analysis-metrics -o  smpp-server-analysis.nvprof%p --profile-child-processes bin/smpp-server console

extern "C" {
#include "smpp_pdu_struct_cuda.h"
}

static uint8_t *pduBuffer_d;
static CudaPduContext *pduContexts_d;
static CudaDecodedContext *decodedPduStructList_d;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
true

) {
if (code != cudaSuccess) {
fprintf(stderr,
"\nGPUassert: %s %s %d\n",
cudaGetErrorString(code),
        file, line
);
if (abort)
exit(code);
}
}
__shared__ CudaPduContext directPduContext[1024];

__global__ void launchDecode(int nPduContext, CudaPduContext *pduContexts,
                             CudaDecodedContext *decodedPduStructList, uint8_t *pduBuffer) {
    int threadIndex = ((threadIdx.x + threadIdx.y * blockDim.x) + (blockDim.x * blockDim.y) * threadIdx.z) +
                      (blockDim.x * blockDim.y * blockDim.z) *
                      ((gridDim.x * blockIdx.y) + (blockIdx.x) + (gridDim.x * gridDim.y) * blockIdx.z);


//    printf("ThreadIndex - x - %d | y - %d\n", threadIdx.x, threadIdx.y);
//    printf("decoding launched  - Correlation Id %s | Length - %d\n", pduContexts[0].correlationId, pduContexts[0].start);

    int initBatchSize = nPduContext / ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));
    int remainder = nPduContext % ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));

//    int additionalElements = threadIndex < remainder ? 1 : 0;
    int additionalElements = threadIndex < remainder ? 1 : 0;
    int batchSize = initBatchSize + additionalElements;
    if (batchSize > 0) {
        int i;
        for (i = 0; i < batchSize; i++) {
            int index = threadIndex + i * ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));
//            printf("ThreadIndex - 1 - %d | Index - %d\n", threadIndex, index);
            directPduContext[(threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) +
                             (i)] = pduContexts[index];
        }
        __syncthreads();

        for (i = 0; i < batchSize; i++) {
            int index = threadIndex + i * ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));
            CudaPduContext *cudaPduContext = &directPduContext[
                    (threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) + i];
            decodeSinglePdu(
                    &directPduContext[(threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) + i],
                    &decodedPduStructList[index], pduBuffer);
        }
    }
}

void cudaTest() {
    printf("Cuda SMPP decoding completed\n");

}

CudaPduContext *allocatePinnedPduContext(int length) {
    CudaPduContext *pduContexts;
    gpuErrchk(cudaMallocHost((void **) &pduContexts, sizeof(CudaPduContext) * length));
    return pduContexts;
}

void initCudaParameters(uint32_t pduContextSize, uint64_t pduBufferLength) {
    gpuErrchk(cudaMalloc((void **) &pduBuffer_d, sizeof(uint8_t) * pduBufferLength));
    gpuErrchk(cudaMalloc((void **) &pduContexts_d, sizeof(CudaPduContext) * pduContextSize));
    gpuErrchk(cudaMalloc((void **) &decodedPduStructList_d, sizeof(CudaDecodedContext) * pduContextSize));
}

void freePinndedPduContext(int length, CudaPduContext *pduContexts) {
    cudaFreeHost(pduContexts);
}

void freePinndedDecodedContext(int length, CudaDecodedContext *decodedPduStructList) {
    cudaFreeHost(decodedPduStructList);
}

void decodeCuda(CudaMetadata cudaMetadata) {

    CudaPduContext *pduContexts = cudaMetadata.cudaPduContexts;
    CudaDecodedContext *decodedPduStructList = cudaMetadata.decodedPduStructList;
    uint8_t *pduBuffer = cudaMetadata.pduBuffer;
    int nPduContext = cudaMetadata.length;
    int pduContextSize = nPduContext * sizeof(CudaPduContext);
    int decodedContextSize = nPduContext * sizeof(CudaDecodedContext);

    cudaMemcpy(pduBuffer_d, pduBuffer, sizeof(uint8_t) * cudaMetadata.pduBufferLength, cudaMemcpyHostToDevice);
    cudaMemcpy(pduContexts_d, pduContexts, sizeof(CudaPduContext) * nPduContext, cudaMemcpyHostToDevice);

    CudaDim block = cudaMetadata.blockDim;
//    printf("Cuda Block Dim | x - %d\n", block.x);
    CudaDim grid = cudaMetadata.gridDim;
    dim3 gridDim(grid.x, grid.y, grid.z);
    dim3 blockDim(block.x, block.y, block.z);

    gpuErrchk(cudaPeekAtLastError());
    launchDecode << < gridDim, blockDim >> > (nPduContext, pduContexts_d, decodedPduStructList_d, pduBuffer_d);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(
            cudaMemcpy(decodedPduStructList, decodedPduStructList_d, sizeof(CudaDecodedContext) * nPduContext,
                       cudaMemcpyDeviceToHost));
    gpuErrchk(cudaGetLastError());
    cudaDeviceSynchronize();
//    cudaFree(decodedPduStructList_d);
//    cudaFree(pduContexts_d);
//    cudaFree(pduBuffer_d);

//    printf("Cuda SMPP decoding completed\n");
}
