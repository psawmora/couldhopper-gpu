/*
 Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
 Created by prabath on 6/26/16.
 */

#include <stdio.h>
#include <stdint.h>
#include <strings.h>
#include "cuda_smpp_util.h"

__device__ int str_len(const char *str) {
//	printf("char - %c | ",*str);
    return (*str) ? str_len(++str) + 1 : 0;
}

uint8_t readUint8(ByteBufferContext *pduBufferContext) {
    /*printf("readUint8 - Read Index - %ld | Limit - %ld\n",
            pduBufferContext->readIndex, pduBufferContext->limit);*/
    uint8_t value = (pduBufferContext->buffer)[pduBufferContext->readIndex];
    pduBufferContext->readIndex += 1;
    return value;
}

uint32_t readUint32(ByteBufferContext *pduBufferContext) {
    /*printf("readUint32 - Read Index - %ld | Limit - %ld\n",
            pduBufferContext->readIndex, pduBufferContext->limit);*/
    uint8_t * buffer = pduBufferContext->buffer + pduBufferContext->readIndex;
    uint32_t value = (uint32_t) ((buffer[0] & 255) << 24 | (buffer[1] & 255) << 16 | (buffer[2] & 255) << 8 | (buffer[3] & 255));
    pduBufferContext->readIndex += 4;
    return value;
}

uint32_t readUint32WithoutContext(uint8_t *currentBuffer, uint64_t readIndex) {
    uint8_t * buffer = currentBuffer + readIndex;
    uint32_t value = (uint32_t) ((buffer[0] & 255) << 24 | (buffer[1] & 255) << 16 | (buffer[2] & 255) << 8 | (buffer[3] & 255));
    return value;
}

void readNBytes(ByteBufferContext *pduBufferContext, int length,
                uint8_t *byteBuffer) {
    /*
     printf("readNBytes - Read Index - %d | Limit - %d\n",
     pduBufferContext->readIndex, pduBufferContext->limit);
     */

    uint64_t readIndex = pduBufferContext->readIndex;
    if (readIndex + length > readIndex + pduBufferContext->limit) {
        printf("Read length - |%ld| overflows the limit\n", length);
        return;
    }

    uint8_t * startPoint = (pduBufferContext->buffer) + readIndex;
    // Need to free memory as soon as the the JNI call returns.
    int i;
    for (i = 0; i < length; i++) {
        byteBuffer[i] = startPoint[i];
    }
    pduBufferContext->readIndex += length;
}

void readNullTerminatedString(ByteBufferContext *pduBufferContext,
                              int maxLength, char *stringValue) {

    /*printf("readNullTerminatedString - Read Index - %d | Limit - %d\n",
            pduBufferContext->readIndex, pduBufferContext->limit);*/

    uint64_t readIndex = pduBufferContext->readIndex;
    uint8_t * startPoint = (pduBufferContext->buffer) + readIndex;
    int stringLength = str_len((const char *) startPoint);
//	int stringLength = 200;
//	printf("\nString length - %d\n", stringLength);
    if (stringLength + 1 > maxLength) {
        printf("String length is greater than the maximum length - %d\n",
               stringLength);
        return;
    }
//    printf("String length - %d | ReadIndex - %d \n", stringLength, readIndex);
    // Need to free memory as soon as the the JNI call returns.
    int i;
    for (i = 0; i < stringLength; i++) {
        stringValue[i] = (char) startPoint[i];
        /*if (startPoint[i] == '\0') {
            break;
        }*/
    }
    stringValue[stringLength] = '\0';
    pduBufferContext->readIndex += (stringLength + 1);
//	pduBufferContext->readIndex += (i);
}

uint64_t readStringByLength(uint64_t readIndex, uint8_t *buffer, int length, char *stringValue) {
    const uint8_t *startPoint = buffer + readIndex;
    int i;
    for (i = 0; i < length; i++) {
        stringValue[i] = startPoint[i];
    }
    stringValue[length] = '\0';
    readIndex += length;
    return readIndex;
}

