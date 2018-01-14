#include "libconfig.h"
#include "smpp_codec.h"

int useGpu = 0;
int useDynamicParallelism = 0;
CudaDim blockDimProdction;
CudaDim gridDimProduction;
int cpuDecodeThreshold = 4000;
int nCpuCores = 2;
int cudaStreamCount = 1;
float cudaEventRunningTime = 0;
int printAccuracyLog=0;
static CudaPduContext *cudaPduContext; // Pre-allocate Cuda-Context for decoding;

static uint32_t currentCudaContextSize;

static CodecMode currentMode;

static log4c_category_t *gpuTunerCategory;
static log4c_category_t *cpuTunerCategory;

static int maxPacketSize = 100;
static int maxBatchSize = 10000;
typedef struct cuda_performance_metric_struct {
    CudaDim *blockDimList;
    CudaDim *gridDimList;
    uint32_t length;

} CudaPerformanceMetric;

static CudaPerformanceMetric performanceMetric;

static int tunerLoopCount = 1;
static int performanceMetricCount = 0;

void configureTuner(config_t *propConfig);

CudaDecodedContext *decodeGpuDynamic(DecoderMetadata decoderMetadata);

void init(CodecConfiguration *configuration) {
    // Need to initialize loggers.
    // Then need to pre-allocate pinned memory and Cuda Global Memory.
    gpuTunerCategory = log4c_category_get("performance_tuning_gpu_logger");
    cpuTunerCategory = log4c_category_get("performance_tuning_cpu_logger");
    log4c_init();

    currentMode = PRODUCTION;
    config_t propConfig;
    config_init(&propConfig);
    if (!config_read_file(&propConfig, configuration->propertyFilePath)) {
        fprintf(stderr, "%s:%d - %s\n",
                config_error_file(&propConfig), config_error_line(&propConfig), config_error_text(&propConfig));
        config_destroy(&propConfig);
        return;
    }

    config_lookup_bool(&propConfig, "isUseGpu", &useGpu);
    config_lookup_bool(&propConfig, "isUseDynamicParallelism", &useDynamicParallelism);
    config_lookup_int(&propConfig, "max_packet_size", &maxPacketSize);
    config_lookup_int(&propConfig, "max_batch_size", &maxBatchSize);
    config_lookup_int(&propConfig, "tuner_loop_count", &tunerLoopCount);
    config_lookup_int(&propConfig, "number_of_cpu_cores", &nCpuCores);
    config_lookup_int(&propConfig, "cuda_stream_count", &cudaStreamCount);
    config_lookup_bool(&propConfig, "print_accuracy_log", &printAccuracyLog);

    currentCudaContextSize = (uint32_t) maxBatchSize;
    cudaPduContext = allocatePinnedPduContext(maxBatchSize);
    initCudaParameters((uint32_t) maxBatchSize, ((uint64_t) maxPacketSize * (uint64_t) maxBatchSize));

    int isTunerMode = 0;
    config_lookup_bool(&propConfig, "isTunerMode", &isTunerMode);
    currentMode = isTunerMode ? TUNER : PRODUCTION;
    printf("Is Use GPU %d | Max Packet Size %d | Max Batch Size %d | Tuner Mode %d | Accuracy Log %d\n",
           useGpu, maxPacketSize, maxBatchSize, currentMode, printAccuracyLog);
    fflush(stdout);
    switch (currentMode) {
        case TUNER: {
            configureTuner(&propConfig);
            break;
        }
        default: {
            int x, y, z;
            config_setting_t *block = config_lookup(&propConfig, "production.block");
            config_setting_t *grid = config_lookup(&propConfig, "production.grid");

            config_setting_lookup_int(block, "x", &x);
            config_setting_lookup_int(block, "y", &y);
            config_setting_lookup_int(block, "z", &z);

            blockDimProdction.x = (uint32_t) x;
            blockDimProdction.y = (uint32_t) y;
            blockDimProdction.z = (uint32_t) z;

            config_setting_lookup_int(grid, "x", &x);
            config_setting_lookup_int(grid, "y", &y);
            config_setting_lookup_int(grid, "z", &z);

            gridDimProduction.x = (uint32_t) x;
            gridDimProduction.y = (uint32_t) y;
            gridDimProduction.z = (uint32_t) z;

            config_lookup_int(&propConfig, "production.cpu_decode_threshold", &cpuDecodeThreshold);
        }
    }
}

void configureTuner(config_t *propConfig) {
    config_setting_t *blockSetting = config_lookup(propConfig, "tuner.block");
    config_setting_t *gridSetting = config_lookup(propConfig, "tuner.grid");

    if (blockSetting != NULL && gridSetting != NULL) {
        int length = config_setting_length(blockSetting); // Only need the length of one metric
        performanceMetricCount = length;
        performanceMetric.blockDimList = malloc(sizeof(CudaDim) * length);
        performanceMetric.gridDimList = malloc(sizeof(CudaDim) * length);
        int i;
        for (i = 0; i < length; ++i) {
            int x, y, z;
            config_setting_t *block = config_setting_get_elem(blockSetting, i);
            config_setting_t *grid = config_setting_get_elem(gridSetting, i);

            config_setting_lookup_int(block, "x", &x);
            config_setting_lookup_int(block, "y", &y);
            config_setting_lookup_int(block, "z", &z);
            performanceMetric.blockDimList[i].x = (uint32_t) x;
            performanceMetric.blockDimList[i].y = (uint32_t) y;
            performanceMetric.blockDimList[i].z = (uint32_t) z;

            config_setting_lookup_int(grid, "x", &x);
            config_setting_lookup_int(grid, "y", &y);
            config_setting_lookup_int(grid, "z", &z);
            performanceMetric.gridDimList[i].x = (uint32_t) x;
            performanceMetric.gridDimList[i].y = (uint32_t) y;
            performanceMetric.gridDimList[i].z = (uint32_t) z;
        }
    }
}

void startPerfTunerPthread(DecoderMetadata decoderMetadata) {
    printf("Printing accuracy - %d", printAccuracyLog);
    fflush(stdout);
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "\nStarting Performance Tuning For CPU\n");
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "===================================\n\n");
    int i, j;
    time_t start_t, end_t;
    double diff_t;
    struct timespec tstart = {0, 0}, tend = {0, 0};
    time(&start_t);
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO,
                       "Number of packets being decoded - %d\n\n", decoderMetadata.size);
    for (i = 0; i < tunerLoopCount; i++) {
        DecodedContext *pStruct = decodePthread(decoderMetadata);
        if(printAccuracyLog) {
         printf("Printing accuracy logs - %d", decoderMetadata.size);
         fflush(stdout);
         int j;
         for(j = 0; j < decoderMetadata.size; j++){
           DecodedContext decodedContext = pStruct[j];
           BaseSmReq baseSmReq = *(BaseSmReq *) decodedContext.pduStruct;
           log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "source-address-value - %s|", baseSmReq.sourceAddress->addressValue);
           log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "destination-address-value - %s|", baseSmReq.destinationAddress->addressValue);
           log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "message - %s\n", baseSmReq.shortMessage);
         }
        }
        free(pStruct);
    }
    time(&end_t);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    diff_t = (((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) - ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec)) /
             tunerLoopCount;
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "TimeTaken - %.5f \n", diff_t);
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "Throughput - %.5f \n", (decoderMetadata.size) / diff_t);
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, "Time for a single packet (Micro-Seconds) - %.5f \n\n",
                       (diff_t * 1000000) / (decoderMetadata.size));
    log4c_category_log(cpuTunerCategory, LOG4C_PRIORITY_INFO, " =====================\n\n");
}

void startPerfTuner(DecoderMetadata decoderMetadata) {
    int i, j;
    time_t start_t, end_t;
    double diff_t;
    struct timespec tstart = {0, 0}, tend = {0, 0};

    log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "\nStarting Performance Tuning For GPU\n");
    log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "===================================\n\n");
    printf("SIZE - %ld", sizeof(CudaPduContext) * 1024);
    for (i = 0; i < performanceMetricCount; i++) {
        CudaDim gridDim = performanceMetric.gridDimList[i];
        CudaDim blockDim = performanceMetric.blockDimList[i];
        decoderMetadata.gridDim = gridDim;
        decoderMetadata.blockDim = blockDim;
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO,
                           "Block Size {x,y,z} - {%d, %d, %d} | Grid Size {x,y,z} - {%d, %d, %d} \n",
                           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO,
                           "Number of packets being decoded - %d\n", decoderMetadata.size);

        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO,
                           "Is Using Dynamic Parallelism - %d\n\n", useDynamicParallelism);
        time(&start_t);
        clock_gettime(CLOCK_MONOTONIC, &tstart);
        float gpuEventTime = 0;
        for (j = 0; j < tunerLoopCount; j++) {
            CudaDecodedContext *pStruct;
            if (useDynamicParallelism) {
                 pStruct = decodeGpuDynamic(decoderMetadata);
                gpuEventTime += cudaEventRunningTime;
//                printf("Smpp Pdu type %d\n", pStruct[decoderMetadata.size - 1].commandId);
//                printf("Smpp CorrelationId %s\n", pStruct[decoderMetadata.size - 1].correlationId);
//                printf("Smpp Message %s\n", pStruct[decoderMetadata.size - 1].pduStruct.shortMessage);
            } else {
                pStruct = decodeGpu(decoderMetadata);
                gpuEventTime += cudaEventRunningTime;
            }

            if(printAccuracyLog) {
               printf("Printing accuracy logs - %d", decoderMetadata.size);
               fflush(stdout);
               int j;
               printf("Decoded Context count - %d", decoderMetadata.size);
               for(j = 0; j < decoderMetadata.size; j++){
                   CudaDecodedContext decodedContext = pStruct[j];
                   CudaBaseSmReq baseSmReq = (CudaBaseSmReq) decodedContext.pduStruct;
                   log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "source-address-value - %s|", baseSmReq.sourceAddress.addressValue);
                   log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "destination-address-value - %s|", baseSmReq.destinationAddress.addressValue);
                   log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "message - %s\n", baseSmReq.shortMessage);
               }
            }
            free(pStruct);
        }

        time(&end_t);
        clock_gettime(CLOCK_MONOTONIC, &tend);
        diff_t = (((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) - ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec)) /
                 tunerLoopCount;
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "TimeTaken - %.5f \n", (diff_t * 1000000));
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "Throughput - %.5f \n", (decoderMetadata.size) / diff_t);
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "Time for a single packet (Micro-Seconds) - %.5f \n\n",
                           (diff_t * 1000000) / (decoderMetadata.size));
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, "GPU Event Time - %.5f \n", gpuEventTime);
        log4c_category_log(gpuTunerCategory, LOG4C_PRIORITY_INFO, " =====================\n");
    }
    freePinndedMemory();
    freePinndedPduContext(maxBatchSize, cudaPduContext);
}

CudaDecodedContext *decodeGpuDynamic(DecoderMetadata decoderMetadata) {
    CudaDecodedContext *decodedPduStructList = malloc(sizeof(CudaDecodedContext) * decoderMetadata.size);
    CudaMetadata metadata = {decoderMetadata.size, decoderMetadata.pduBuffers, -1, decodedPduStructList,
                             (uint64_t) decoderMetadata.bufferCapacity,
                             decoderMetadata.blockDim, decoderMetadata.gridDim};
    decodeCudaDynamic(metadata);
    return decodedPduStructList;
}

CudaDecodedContext *decodeGpu(DecoderMetadata decoderMetadata) {
    if (useDynamicParallelism) {
        return decodeGpuDynamic(decoderMetadata);
    }
    uint8_t *pduBuffers = decoderMetadata.pduBuffers;
    uint32_t size = decoderMetadata.size;
    uint64_t bufferCapacity = decoderMetadata.bufferCapacity;
    uint32_t correlationIdLength = decoderMetadata.correlationIdLength;
    ByteBufferContext byteBufferContext = {pduBuffers, 0, (uint64_t) bufferCapacity};
    CudaPduContext *pduContexts = cudaPduContext;
    int i;
    int startPosition = 0;
    for (i = 0; i < size; i++) {
        byteBufferContext.readIndex += 12;
        char *correlationId = readStringByLength(&byteBufferContext, correlationIdLength);
        uint32_t pduLength = readUint32(&byteBufferContext);
        int startIndex = startPosition + correlationIdLength + 12;
        strncpy(pduContexts[i].correlationId, correlationId, sizeof(char) * (correlationIdLength + 1));
        pduContexts[i].start = (uint32_t) startIndex;
        pduContexts[i].length = pduLength;
        startPosition += (correlationIdLength + pduLength + 12);
        byteBufferContext.readIndex = (uint64_t) startPosition;
    }

    CudaDecodedContext *decodedPduStructList = malloc(sizeof(CudaDecodedContext) * size);
    CudaMetadata metadata = {size, pduBuffers, pduContexts, decodedPduStructList, (uint64_t) bufferCapacity,
                             decoderMetadata.blockDim, decoderMetadata.gridDim};
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
        byteBufferContext.readIndex += 12;
        char *correlationId = readStringByLength(&byteBufferContext, correlationIdLength);
        uint32_t pduLength = readUint32(&byteBufferContext);
        int startIndex = startPosition + correlationIdLength + 12;
        pduContexts[i].correlationId = correlationId;
        pduContexts[i].pduBuffer = pduBuffers;
        pduContexts[i].start = startIndex;
        pduContexts[i].length = pduLength;
        startPosition += (correlationIdLength + pduLength + 12);
        byteBufferContext.readIndex = startPosition;
    }

    int nThread = nCpuCores;
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
