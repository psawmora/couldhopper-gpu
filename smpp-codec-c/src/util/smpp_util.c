/*
Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
Created by prabath on 6/26/16.
*/

#include <stdint.h>
#include <string.h>

#include "smpp_util.h"

uint8_t readUint8(ByteBufferContext *pduBufferContext) {
//    printf("readUint8 - Read Index - %d | Limit - %d\n", pduBufferContext->readIndex, pduBufferContext->limit);
    uint8_t value = (pduBufferContext->buffer)[pduBufferContext->readIndex];
    pduBufferContext->readIndex += 1;
    return value;
}

uint32_t readUint32(ByteBufferContext *pduBufferContext) {
//    printf("readUint32 - Read Index - %d | Limit - %d\n", pduBufferContext->readIndex, pduBufferContext->limit);
    uint8_t *buffer = pduBufferContext->buffer + pduBufferContext->readIndex;
    uint32_t value = (buffer[0] & 255) << 24 | (buffer[1] & 255) << 16 | (buffer[2] & 255) << 8 | (buffer[3] & 255);
    pduBufferContext->readIndex += 4;
    return value;
}

char *readString(ByteBufferContext *pduBufferContext, int startIndex, int length) {
    uint8_t readIndex = pduBufferContext->readIndex;
    if (readIndex + length < startIndex + length) {
        char value[length + 1];
        int i;
        for (i = 0; i < length; i++) {
            value[i] = (pduBufferContext->buffer)[i + startIndex];
        }
        value[length] = '\0';
        return value;
    }
    return NULL;
}

Tlv *decodeTLV(ByteBufferContext *pduBufferContext, int maxTlvCount) {
    uint8_t *byteBuffer = pduBufferContext->buffer;
    while (pduBufferContext->readIndex < pduBufferContext->limit) {
        Tlv tlv;

    }
}

uint8_t *readNBytes(ByteBufferContext *pduBufferContext, int length) {
//    printf("readNBytes - Read Index - %d | Limit - %d\n",
//    pduBufferContext->readIndex, pduBufferContext->limit);

    uint64_t readIndex = pduBufferContext->readIndex;
    if (readIndex + length > pduBufferContext->limit) {
//        printf("Read length - |%d| overflows the limit\n", length);
        return -1;
    }

    uint8_t *startPoint = (pduBufferContext->buffer) + readIndex;
    // Need to free memory as soon as the the JNI call returns.
    uint8_t *byteBuffer = (uint8_t *) malloc(sizeof(uint8_t) * length);
    int i;
    for (i = 0; i < length; i++) {
        byteBuffer[i] = startPoint[i];
    }
    pduBufferContext->readIndex += length;
    return byteBuffer;
}

char *readNullTerminatedString(ByteBufferContext *pduBufferContext, int maxLength) {
/*    printf("readNullTerminatedString - Read Index - %d | Limit - %d\n",
           pduBufferContext->readIndex, pduBufferContext->limit)*/;

    uint64_t readIndex = pduBufferContext->readIndex;
    const uint8_t *startPoint = (pduBufferContext->buffer) + readIndex;
    int stringLength = strlen(startPoint);
    if (stringLength + 1 > maxLength) {
//        printf("String length is greater than the maximum length - %d\n", stringLength);
        return -1;
    }
//    printf("String length - %d\n", stringLength);
    // Need to free memory as soon as the the JNI call returns.
    char *stringValue = (char *) malloc(sizeof(char) * (stringLength + 1));
    int i;
    for (i = 0; i < stringLength; i++) {
        stringValue[i] = startPoint[i];
    }
    stringValue[stringLength] = '\0';
    pduBufferContext->readIndex += (stringLength + 1);
    return stringValue;
}







