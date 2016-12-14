#include "smpp_codec.h"

static CudaPduContext *cudaPduContext; // Pre-allocate Cuda-Context for decoding;

static uint32_t currentCudaContextSize;

void init(CodecConfiguration *configuration) {
    // Need to initialize loggers.
    // Then need to pre-allocate pinned memory and Cuda Global Memory.
    currentCudaContextSize = configuration->pduContextInitSize;
    cudaPduContext = allocatePinnedPduContext(currentCudaContextSize);
    initCudaParameters(configuration->pduContextInitSize, configuration->pduBufferInitSize);
}

CudaDecodedContext *decodeGpu(DecoderMetadata decoderMetadata) {
    uint8_t *pduBuffers = decoderMetadata.pduBuffers;
    uint32_t size = decoderMetadata.size;
    uint64_t bufferCapacity = decoderMetadata.bufferCapacity;
    uint32_t correlationIdLength = decoderMetadata.correlationIdLength;
    ByteBufferContext byteBufferContext = {pduBuffers, 0, (uint64_t) bufferCapacity};
    CudaPduContext *pduContexts = cudaPduContext;
    int i;
    int startPosition = 0;
    for (i = 0; i < size; i++) {
        char *correlationId = readStringByLength(&byteBufferContext, correlationIdLength);
        uint32_t pduLength = readUint32(&byteBufferContext);
        int startIndex = startPosition + correlationIdLength;
        strncpy(pduContexts[i].correlationId, correlationId, sizeof(char) * (correlationIdLength + 1));
        pduContexts[i].start = (uint32_t) startIndex;
        pduContexts[i].length = pduLength;
        /*
                printf("CorrelationId 1 - %s | CorrelationId 2 - %s | start-position - %d| pdu length -  %d | Read Index - %ld \n",
                       correlationId, pduContexts[i].correlationId, startPosition, pduLength, byteBufferContext.readIndex);
        */
        startPosition += (correlationIdLength + pduLength);
        byteBufferContext.readIndex = (uint64_t) startPosition;
    }

    CudaDecodedContext *decodedPduStructList = malloc(sizeof(CudaDecodedContext) * size);
    CudaMetadata metadata = {size, pduBuffers, pduContexts, decodedPduStructList, (uint64_t) bufferCapacity};
    decodeCuda(metadata);
    return decodedPduStructList;
}

DecodedContext *decodePthread(DecoderMetadata decoderMetadata) {

    uint8_t *pduBuffers = decoderMetadata.pduBuffers;
    uint32_t size = decoderMetadata.size;
    uint64_t bufferCapacity = decoderMetadata.bufferCapacity;
    uint32_t correlationIdLength = decoderMetadata.correlationIdLength;

    ByteBufferContext byteBufferContext = {pduBuffers, 0, bufferCapacity};
    DirectPduContext *pduContexts = malloc(sizeof(DirectPduContext) * size);
    int i;
    int startPosition = 0;
    for (i = 0; i < size; i++) {
        char *correlationId = readStringByLength(&byteBufferContext, correlationIdLength);
        uint32_t pduLength = readUint32(&byteBufferContext);
        int startIndex = startPosition + correlationIdLength;
        pduContexts[i].correlationId = correlationId;
        pduContexts[i].pduBuffer = pduBuffers;
        pduContexts[i].start = startIndex;
        pduContexts[i].length = pduLength;
        startPosition += (correlationIdLength + pduLength);
        /*
                printf("CorrelationId - %s | Read Index - %d |  pdu length -  %d | start-position - %d\n",
                       correlationId, byteBufferContext.readIndex, pduLength, startPosition);
        */
        byteBufferContext.readIndex = startPosition;
    }

    int nThread = N_THREAD;
    int index = 0;
    int batchSize = size > nThread ? size / nThread : size;
    DecodedContext *decodedPduStructList = malloc(sizeof(DecodedContext) * size);
    ThreadParam *threadParams[nThread];

    pthread_t threads[nThread];
    int threadIndex = 0;
    while (index < size) {
        int startIndex = index;
        int length = (size - index) <= batchSize ? (size - index) : batchSize;
        index += length;
        ThreadParam *threadParam = malloc(sizeof(ThreadParam));
        threadParam->startIndex = startIndex;
        threadParam->length = length;
        threadParam->pduContexts = pduContexts;
        threadParam->decodedPduStructList = decodedPduStructList;
        threadParams[threadIndex] = threadParam;
        int state = pthread_create(&threads[threadIndex], NULL, decode, (void *) threadParam);
        threadIndex++;
    }

    for (i = 0; i < threadIndex; i++) {
        pthread_join(threads[i], NULL);
    }
    free(pduContexts);
    return decodedPduStructList;
}
