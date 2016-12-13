/**
 * The base header file of the Smpp-Code
 */

#ifndef SMPP_CODEC_C_SMPP_CODEC_H
#define SMPP_CODEC_C_SMPP_CODEC_H

typedef enum codec_mode {
    PERFORMANCE_TUNE, PRODUCTION
} CodecMode;

typedef struct codec_configuration_struct {
    char *propertyFileName;
    char *propertyFilePath;
    CodecMode mode;

} CodecConfiguration;

void init(CodecConfiguration *configuration);

#ifdef PTHREAD_DECODER
#define decodePduDirect(env, pduContainerBuffer, size, correlationIdLength) decodeWithPthread(env, pduContainerBuffer, size, correlationIdLength)
#elif CUDA_DECODER
#define decodePduDirect(env, pduContainerBuffer, size, correlationIdLength) decodeWithCuda(env, pduContainerBuffer, size, correlationIdLength)
#else
#define decodePduDirect(env, pduContainerBuffer, size, correlationIdLength) decodeWithPthread(env, pduContainerBuffer, size, correlationIdLength)
#endif
#endif
