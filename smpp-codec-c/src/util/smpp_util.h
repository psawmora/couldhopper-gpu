/*
Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
Created by prabath on 6/26/16.
*/

#include "smpp_pdu_struct.h"

#ifndef SMPP_CODEC_C_SMPP_UTIL_H
#define SMPP_CODEC_C_SMPP_UTIL_H

typedef struct byte_buffer_context_struct {
    uint8_t *buffer;
    uint64_t readIndex;
    uint64_t limit;
} ByteBufferContext;

uint8_t readUint8(ByteBufferContext *pduBufferContext);

uint32_t readUint32(ByteBufferContext *pduBufferContext);

char *readString(ByteBufferContext *pduBufferContext, int startIndex, int endIndex);

char *readNullTerminatedString(ByteBufferContext *pduBufferContext, int maxLength);

uint8_t *readNBytes(ByteBufferContext *pduBufferContext, int length);

#endif //SMPP_CODEC_C_SMPP_UTIL_H
