/*
Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
Created by prabath on 6/26/16.
*/

#include "pdu_common_struct.h"

#ifndef SMPP_CODEC_C_SMPP_UTIL_H
#define SMPP_CODEC_C_SMPP_UTIL_H

uint8_t readUint8(ByteBufferContext *pduBufferContext);

uint32_t readUint32(ByteBufferContext *pduBufferContext);

char *readString(ByteBufferContext *pduBufferContext, int startIndex, int endIndex);

char *readNullTerminatedString(ByteBufferContext *pduBufferContext, int maxLength);

char *readStringByLength(ByteBufferContext *pduBufferContext, int length);

uint8_t *readNBytes(ByteBufferContext *pduBufferContext, int length);

#endif //SMPP_CODEC_C_SMPP_UTIL_H
