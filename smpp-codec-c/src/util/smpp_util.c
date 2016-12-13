/*
Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
Created by prabath on 6/26/16.
*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "smpp_util.h"

int str_len(const char *str) {
//    printf("char - %c | ", *str);
    return (*str) ? str_len(++str) + 1 : 0;
}

uint8_t readUint8(ByteBufferContext *pduBufferContext) {
//    printf("readUint8 - Read Index - %d | Limit - %d\n", pduBufferContext->readIndex, pduBufferContext->limit);
    uint8_t value = (pduBufferContext->buffer)[pduBufferContext->readIndex];
    pduBufferContext->readIndex += 1;
    return value;
}

uint16_t readUint16(ByteBufferContext *pduBufferContext) {
    uint8_t *buffer = pduBufferContext->buffer + pduBufferContext->readIndex;
    uint16_t value = (buffer[0] & 255) << 8 | (buffer[1] & 255);
/*
    printf("readUint16 - Read Index - %d | Limit - %d | Int Value %d\n", pduBufferContext->readIndex, pduBufferContext->limit,
           value);
*/
    pduBufferContext->readIndex += 2;
    return value;
}

uint32_t readUint32(ByteBufferContext *pduBufferContext) {
    uint8_t *buffer = pduBufferContext->buffer + pduBufferContext->readIndex;
    uint32_t value = (buffer[0] & 255) << 24 | (buffer[1] & 255) << 16 | (buffer[2] & 255) << 8 | (buffer[3] & 255);
/*
    printf("readUint32 - Read Index - %d | Limit - %d | Int Value %d\n", pduBufferContext->readIndex, pduBufferContext->limit,
           value);
*/
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

uint8_t *readNBytes(ByteBufferContext *pduBufferContext, int length) {
/*
    printf("readNBytes - Read Index - %d | Limit - %d\n",
           pduBufferContext->readIndex, pduBufferContext->limit);
*/

    uint64_t readIndex = pduBufferContext->readIndex;
    if (readIndex + length > readIndex + pduBufferContext->limit) {
        printf("Read length - |%d| overflows the limit\n", length);
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
/*
    printf("readNullTerminatedString - Read Index - %d | Limit - %d\n",
           pduBufferContext->readIndex, pduBufferContext->limit);
*/

    uint64_t readIndex = pduBufferContext->readIndex;
    const uint8_t *startPoint = (pduBufferContext->buffer) + readIndex;
    int stringLength = str_len((const char *) startPoint);
    if (stringLength + 1 > maxLength) {
        printf("String length is greater than the maximum length - %d\n", stringLength);
        return -1;
    }
//    printf("String length - %d | ReadIndex - %d \n", stringLength, readIndex);
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

char *readStringByLength(ByteBufferContext *pduBufferContext, int length) {
/*
    printf("readNullTerminatedString - Read Index - %d | Limit - %d\n",
           pduBufferContext->readIndex, pduBufferContext->limit);
*/

    uint64_t readIndex = pduBufferContext->readIndex;
    const uint8_t *startPoint = (pduBufferContext->buffer) + readIndex;
//    printf("By Length - String length - %d | ReadIndex - %d \n", length, readIndex);
    // Need to free memory as soon as the the JNI call returns.
    char *stringValue = (char *) malloc(sizeof(char) * (length + 1));
    int i;
    for (i = 0; i < length; i++) {
        stringValue[i] = startPoint[i];
    }
    stringValue[length] = '\0';
    pduBufferContext->readIndex += length;
    return stringValue;
}








