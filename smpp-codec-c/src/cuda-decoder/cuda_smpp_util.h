
extern "C" {
#include "smpp_pdu_struct_cuda.h"
}

#ifndef CUDA_SMPP_UTIL
#define CUDA_SMPP_UTIL

__device__ uint8_t readUint8(ByteBufferContext *pduBufferContext);

__device__ uint32_t readUint32(ByteBufferContext *pduBufferContext);

__device__ void readNBytes(ByteBufferContext *pduBufferContext, int length, uint8_t *byteBuffer);

__device__ void readNullTerminatedString(ByteBufferContext *pduBufferContext, int maxLength, char *stringValue);

#endif