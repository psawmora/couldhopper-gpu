#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "cuda_decoder_device.h"
#include "smpp_codec.h"

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
__shared__ uint32_t nPduCount;
__shared__ int minPduIndex;


__global__ void launchDecodeLocal(int nPduContext, int startIndex, CudaPduContext *originalPduContexts,
                                  CudaDecodedContext *originalDecodedPduStructList, uint8_t *pduBuffer) {
    int threadIndex = ((threadIdx.x + threadIdx.y * blockDim.x) + (blockDim.x * blockDim.y) * threadIdx.z) +
                      (blockDim.x * blockDim.y * blockDim.z) *
                      ((gridDim.x * blockIdx.y) + (blockIdx.x) + (gridDim.x * gridDim.y) * blockIdx.z);

    CudaPduContext *pduContexts = originalPduContexts + startIndex;
    CudaDecodedContext *decodedPduStructList = originalDecodedPduStructList + startIndex;
    int initBatchSize = nPduContext / ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));
    int remainder = nPduContext % ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));

    int additionalElements = threadIndex < remainder ? 1 : 0;
    int batchSize = initBatchSize + additionalElements;
    if (batchSize > 0) {
/*
        if ((threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) == 0) {
            printf("pdu count 1 - 0 - %d\n", nPduContext);
        }
*/
//        printf("batch size - %d\n", batchSize);
        int i;
/*
        for (i = 0; i < batchSize; i++) {
            int index = threadIndex + i * ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));
            int localIndex = (threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) + i;
            CudaPduContext aStruct1 = pduContexts[index];
            CudaPduContext aStruct2 = directPduContext[localIndex];
            aStruct2 = aStruct1;
        }
*/
//        __syncthreads();
        for (i = 0; i < batchSize; i++) {
            int index = threadIndex + i * ((blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z));
/*
            CudaPduContext *cudaPduContext = &directPduContext[
                    (threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) + i];
*/
            CudaPduContext *cudaPduContext = &pduContexts[index];
            CudaDecodedContext *decodedPduStruct = &decodedPduStructList[index];
            decodeSinglePdu(cudaPduContext, decodedPduStruct, pduBuffer);
        }
    }
}

__global__ void findPacketBoundry(uint8_t *pduBuffer,
                                  uint32_t bufferLength,
                                  CudaDecodedContext *decodedPduStructList,
                                  CudaPduContext *globalDirectPduContext,
                                  int correlationIdLength) {
    uint8_t pattern[8] = {127, 127, 127, 127, 127, 127, 127, 127};
    int threadIndex = ((threadIdx.x + threadIdx.y * blockDim.x) + (blockDim.x * blockDim.y) * threadIdx.z) +
                      (blockDim.x * blockDim.y * blockDim.z) *
                      ((gridDim.x * blockIdx.y) + (blockIdx.x) + (gridDim.x * gridDim.y) * blockIdx.z);

    uint32_t totalThreadCount = (blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * gridDim.z);
    int init20BytesBatchSize = bufferLength / 20;
    int remainder20Bytes = bufferLength % 20;

    int initBatchSize = init20BytesBatchSize / totalThreadCount;
    int remainder = init20BytesBatchSize % totalThreadCount;
    int additionalElements = threadIndex < remainder ? 1 : 0;
    int batchSize = (initBatchSize + additionalElements) * 20 + (threadIndex == 0) * remainder20Bytes;
    int threadBlockIndex = threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z;
    if (threadBlockIndex == 0) {
        nPduCount = 0;
        minPduIndex = 100000000;
    }
    __syncthreads();

    if (batchSize > 0) {
        int startIndex = threadIndex;
        if (threadIndex < remainder && threadIndex > 0) {
            startIndex = threadIndex * (initBatchSize + additionalElements) * 20 + remainder20Bytes;
        }
        if (threadIndex >= remainder && threadIndex > 0) {
            startIndex = ((remainder) * (initBatchSize + 1) * 20 + remainder20Bytes) +
                         (threadIndex - remainder) * initBatchSize * 20;
        }
        int stop = 0;
        int count = 0;
        int patternPosition = 0;
        int numberOfPdu = 0;

        while (!stop) {
            uint8_t valueAtIndex = *(pduBuffer + startIndex + count);
            if (valueAtIndex == pattern[patternPosition]) {
                patternPosition++;
            } else {
                patternPosition = 0;
            }
            if (patternPosition == 8) {
                ++numberOfPdu;
                patternPosition = 0;

                uint64_t breakPoint = (uint64_t) (startIndex + count + 1);
                int index = readUint32WithoutContext(pduBuffer, (uint64_t) breakPoint);
                CudaPduContext *pduContext = &globalDirectPduContext[index];
                breakPoint += 4;

                uint64_t nextReadIndex =
                        readStringByLength(
                                (uint64_t) (breakPoint), pduBuffer, correlationIdLength, pduContext->correlationId);
                breakPoint = nextReadIndex;

                uint32_t pduLength = readUint32WithoutContext(pduBuffer, (uint64_t) breakPoint);
                pduContext->start = (uint32_t) breakPoint;
                pduContext->length = pduLength;
                breakPoint += pduLength - 1;
                atomicMin(&minPduIndex, index);
                /**
                 * 1. Create CudaPduContext with start and length values.
                 * 2. Store it in block shared memory.
                 * 3. After the block is done processing, store them in the global memory.
                 * 4. Call the child kernel to do the actual decoding.
                 */
            }
            count++;
            if ((patternPosition == 0 && count >= batchSize) || (startIndex + count) >= bufferLength) {
                stop = 1;
            }
        }
//        printf("Pdu Count - %d\n", numberOfPdu);
        atomicAdd(&nPduCount, numberOfPdu);
    }

    __syncthreads();
    if (threadBlockIndex == 0) {
        if (nPduCount > 0) {
            int blockSize = nPduCount >= 256 ? 256 / 3 : nPduCount / 3;
            int gridSize = (nPduCount / blockSize) >= 1 ? (nPduCount / blockSize) : 1;
//            printf("Pdu Count - 0 - %d - min - %d\n", nPduCount, minPduIndex);

            dim3 gridDim(gridSize, gridSize, gridSize);
            dim3 blockDim(blockSize, blockSize, blockSize);
            launchDecodeLocal << < gridDim, blockDim >> >
                                            (nPduCount, minPduIndex, globalDirectPduContext, decodedPduStructList, pduBuffer);
        }
        cudaDeviceSynchronize();
    }
}

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
            directPduContext[(threadIdx.x + threadIdx.y * blockDim.x + (blockDim.x * blockDim.y) * threadIdx.z) + i]
                    = pduContexts[index];
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


/**
 * findPacketBoundry(uint8_t *pduBuffer,
                                   uint32_t bufferLength,
                                   CudaDecodedContext *decodedPduStructList,
                                   CudaPduContext *globalDirectPduContext,
                                   int correlationIdLength)
 * @param cudaMetadata
 */

void decodeCudaDynamic(CudaMetadata cudaMetadata) {
    CudaDim block = cudaMetadata.blockDim;
    CudaDim grid = cudaMetadata.gridDim;
    dim3 gridDim(grid.x, grid.y, grid.z);
    dim3 blockDim(block.x, block.y, block.z);

    uint8_t *pduBuffer = cudaMetadata.pduBuffer;
    cudaMemcpy(pduBuffer_d, pduBuffer, sizeof(uint8_t) * cudaMetadata.pduBufferLength, cudaMemcpyHostToDevice);

    gpuErrchk(cudaPeekAtLastError());
    findPacketBoundry << < gridDim, blockDim >> > (
            pduBuffer_d,
                    cudaMetadata.pduBufferLength,
                    decodedPduStructList_d,
                    pduContexts_d,
                    15);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    CudaDecodedContext *decodedPduStructList = cudaMetadata.decodedPduStructList;
    gpuErrchk(
            cudaMemcpy(decodedPduStructList, decodedPduStructList_d, sizeof(CudaDecodedContext) * cudaMetadata.length,
                       cudaMemcpyDeviceToHost));
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    fflush(stdout);
}

void decodeCuda(CudaMetadata cudaMetadata) {

    CudaPduContext *pduContexts = cudaMetadata.cudaPduContexts;
    CudaDecodedContext *decodedPduStructList = cudaMetadata.decodedPduStructList;
    uint8_t *pduBuffer = cudaMetadata.pduBuffer;
    int nPduContext = cudaMetadata.length;
    int pduContextSize = nPduContext * sizeof(CudaPduContext);
    int decodedContextSize = nPduContext * sizeof(CudaDecodedContext);

    // Copies the PduBuffer
    // Devide the PduContexts and decodePduContexts by the number streams.
    // Parallely offload those to the GPU.
    // Do process them parallely and wait for the response.

    cudaMemcpy(pduBuffer_d, pduBuffer, sizeof(uint8_t) * cudaMetadata.pduBufferLength, cudaMemcpyHostToDevice);

    cudaStream_t streams[cudaStreamCount];
    cudaError_t results[cudaStreamCount];

    int streamCount = 0;
    int batchSize = nPduContext > cudaStreamCount ? nPduContext / cudaStreamCount : nPduContext;
    int index = 0;
    while (index < nPduContext) {
        results[streamCount] = cudaStreamCreate(&(streams[streamCount]));
        int startIndex = index;
        int length = (nPduContext - index) <= batchSize ? (nPduContext - index) : batchSize;
        index += length;
        results[streamCount] =
                cudaMemcpyAsync(
                        pduContexts_d + startIndex, pduContexts + startIndex, sizeof(CudaPduContext) * length,
                        cudaMemcpyHostToDevice,
                        streams[streamCount]);
        streamCount++;

    }


    CudaDim block = cudaMetadata.blockDim;
    CudaDim grid = cudaMetadata.gridDim;
    int gridx = grid.x < streamCount ? 1 : grid.x / streamCount;
    int gridy = grid.y < streamCount ? 1 : grid.y / streamCount;
    int gridz = grid.z < streamCount ? 1 : grid.z / streamCount;
    dim3 gridDim(gridx, gridy, gridz);
    dim3 blockDim(block.x, block.y, block.z);
    index = 0;
    int i = 0;
    while (index < nPduContext) {
//        printf("Cuda Stream Number - %d\n", i);
        int startIndex = index;
        int length = (nPduContext - index) <= batchSize ? (nPduContext - index) : batchSize;
        index += length;
        gpuErrchk(cudaPeekAtLastError());
        launchDecode << < gridDim, blockDim, 0, streams[i] >> >
                                                (length, pduContexts_d + startIndex, decodedPduStructList_d + startIndex,
                                                        pduBuffer_d);
        gpuErrchk(cudaPeekAtLastError());
        i++;
    }

    index = 0;
    i = 0;
    while (index < nPduContext) {
        int startIndex = index;
        int length = (nPduContext - index) <= batchSize ? (nPduContext - index) : batchSize;
        index += length;
        gpuErrchk(
                cudaMemcpyAsync(decodedPduStructList + startIndex, decodedPduStructList_d + startIndex,
                                sizeof(CudaDecodedContext) * length,
                                cudaMemcpyDeviceToHost, streams[i]));
        gpuErrchk(cudaGetLastError());
        i++;
    }


    cudaDeviceSynchronize();
    for (i = 0; i < streamCount; i++) {
        results[i] = cudaStreamDestroy(streams[i]);
    }
//    cudaFree(decodedPduStructList_d);
//    cudaFree(pduContexts_d);
//    cudaFree(pduBuffer_d);

//    printf("Cuda SMPP decoding completed\n");
}
