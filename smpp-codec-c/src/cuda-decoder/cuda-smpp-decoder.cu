#include <stdlib.h>
#include <stdio.h>

#include "cuda_decoder_device.h"
extern "C" {
#include "smpp_pdu_struct_cuda.h"
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
		true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString(code),
				file, line);
		if (abort)
			exit(code);
	}
}
__shared__ CudaPduContext directPduContext[512];

__global__ void launchDecode(int nPduContext, CudaPduContext *pduContexts,
		CudaDecodedContext *decodedPduStructList, uint8_t *pduBuffer) {
	int threadIndex = threadIdx.x + blockDim.x * blockIdx.x;
//	printf("ThreadIndex - %d\n", threadIndex);
//    printf("decoding launched  - Correlation Id %s | Length - %d\n",pduContexts[0].correlationId, pduContexts[0].start);
	int initBatchSize = nPduContext / (blockDim.x * gridDim.x);
	int remainder = nPduContext % (blockDim.x * gridDim.x);
	int additionalElements = threadIndex < remainder ? 1 : 0;
	int batchSize = initBatchSize + additionalElements;
	if (batchSize > 0) {
		int start =
				threadIndex < remainder ?
						batchSize * threadIndex :
						remainder * (batchSize + 1)
								+ (threadIndex - remainder) * batchSize;
		int i;
//        printf("\n");
		for (i = start; i < start + batchSize; i++) {
//			printf(" Batch Size %d | real index %d ", batchSize, i);
			directPduContext[threadIdx.x + (i - start)] = pduContexts[i];
		}
//		printf("\n");
		__syncthreads();

		for (i = start; i < start + batchSize; i++) {
//        	printf("Cuda - index %d\n",threadIdx.x + (i - start));
			CudaPduContext* cudaPduContext = &directPduContext[threadIdx.x
					+ (i - start)];
//        	printf("CUDA - Correlation Id %s | Length - %d\n",pduContexts[0].correlationId,pduContexts[0].start);
			decodeSinglePdu(&directPduContext[threadIdx.x + (i - start)], &decodedPduStructList[i],
					pduBuffer);
		}
	}
}

void cudaTest() {
	printf("Cuda SMPP decoding completed\n");

}
void decodeCuda(CudaMetadata cudaMetadata) {

	CudaPduContext *pduContexts = cudaMetadata.cudaPduContexts;
	CudaDecodedContext *decodedPduStructList = cudaMetadata.decodedPduStructList;
	uint8_t *pduBuffer = cudaMetadata.pduBuffer;
	int nPduContext = cudaMetadata.length;

	CudaPduContext *pduContexts_d;
	CudaDecodedContext *decodedPduStructList_d;
	uint8_t *pduBuffer_d;

	int pduContextSize = nPduContext * sizeof(CudaPduContext);
	int decodedContextSize = nPduContext * sizeof(CudaDecodedContext);

	cudaMalloc((void **) &pduBuffer_d,
			sizeof(uint8_t) * cudaMetadata.pduBufferLength);
	cudaMemcpy(pduBuffer_d, pduBuffer,
			sizeof(uint8_t) * cudaMetadata.pduBufferLength,
			cudaMemcpyHostToDevice);

	cudaMalloc((void **) &pduContexts_d, sizeof(CudaPduContext) * nPduContext);
	cudaMemcpy(pduContexts_d, pduContexts, sizeof(CudaPduContext) * nPduContext,
			cudaMemcpyHostToDevice);

	gpuErrchk(
			cudaMalloc((void ** ) &decodedPduStructList_d,
					sizeof(CudaDecodedContext) * nPduContext));
	gpuErrchk(
			cudaMemcpy(decodedPduStructList_d, decodedPduStructList,
					sizeof(CudaDecodedContext) * nPduContext,
					cudaMemcpyHostToDevice));

	CudaPduContext *testContext = (CudaPduContext *) malloc(
			sizeof(CudaPduContext));
	memcpy(testContext, &pduContexts[0], sizeof(CudaPduContext));
	/*printf(
			"Before launch Correlation Id %s | pdu count %d | buffer length %d\n",
			testContext->correlationId, nPduContext,
			cudaMetadata.pduBufferLength);*/

	launchDecode<<<512, 512>>>(nPduContext, pduContexts_d, decodedPduStructList_d,
			pduBuffer_d);
	gpuErrchk(cudaPeekAtLastError());
	cudaDeviceSynchronize();
	gpuErrchk(
			cudaMemcpy(decodedPduStructList, decodedPduStructList_d,
					sizeof(CudaDecodedContext) * nPduContext,
					cudaMemcpyDeviceToHost));
	gpuErrchk(cudaGetLastError());
	free(testContext);
	cudaDeviceSynchronize();
	cudaFree(decodedPduStructList_d);
	cudaFree(pduContexts_d);
	cudaFree(pduBuffer_d);

	printf("Cuda SMPP decoding completed\n");
}
