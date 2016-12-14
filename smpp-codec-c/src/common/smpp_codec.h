/**
 * The base header file of the Smpp-Code
 */

#ifndef SMPP_CODEC_C_SMPP_CODEC_H
#define SMPP_CODEC_C_SMPP_CODEC_H

#include "log4c.h"
#include "smpp_util.h"
#include "smpp_pdu_struct.h"
#include "smpp_pdu_struct_cuda.h"

typedef struct decoder_metadata_struct {
    uint8_t *pduBuffers;
    uint32_t size;
    uint64_t bufferCapacity;
    uint32_t correlationIdLength;
} DecoderMetadata;

typedef enum codec_mode {
    PERFORMANCE_TUNE, PRODUCTION
} CodecMode;

typedef struct codec_configuration_struct {
//    char *propertyFileName;
//    char *propertyFilePath;
//    CodecMode mode;
    uint32_t pduContextInitSize;
    uint64_t pduBufferInitSize;
} CodecConfiguration;

void init(CodecConfiguration *configuration);

CudaDecodedContext *decodeGpu(DecoderMetadata decoderMetadata);

DecodedContext *decodePthread(DecoderMetadata decoderMetadata);

#ifdef PTHREAD_DECODER
#define decodePduDirect(env, pduContainerBuffer, size, correlationIdLength) decodeWithPthread(env, pduContainerBuffer, size, correlationIdLength)
#elif CUDA_DECODER
#define decodePduDirect(env, pduContainerBuffer, size, correlationIdLength) decodeWithCuda(env, pduContainerBuffer, size, correlationIdLength)
#else
#define decodePduDirect(env, pduContainerBuffer, size, correlationIdLength) decodeWithPthread(env, pduContainerBuffer, size, correlationIdLength)
#endif
#endif
