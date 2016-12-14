/*
Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
Created by prabath on 6/26/16.
*/

#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/queue.h>
#include <unistd.h>

#include "smpp_pdu_struct.h"
#include "smpp_util.h"

void printAddress(Address *address);

SubmitSmReq *decodeSubmitSm(PduContext *pduContext, ByteBufferContext *context);

typedef struct DecodedTlvContextStruct {
    Tlv *tlvList;
    uint16_t nTlv;
} DecodedTlvContext;

DecodedTlvContext decodeTlv(ByteBufferContext *context);

void *decode(void *threadParam) {
    ThreadParam *context = (ThreadParam *) threadParam;
    pid_t x = syscall(__NR_gettid);
    DirectPduContext *pduContexts = context->pduContexts;
    DecodedContext *decodedPduStructList = context->decodedPduStructList;
    int length = context->length;
    int startIndex = context->startIndex;
    int i = startIndex;
//    printf("Id %d - Start - %d | End - %d\n", x, i, startIndex + length);
    for (i = startIndex; i < startIndex + length; i++) {
        void *decodedPduStruct = decodeSingle(&pduContexts[i]);
        if (decodedPduStruct != 0) {
            DecodedContext *decodedContext = (DecodedContext *) decodedPduStruct;
//            printf("Command-Id %d \n", decodedContext->commandId);
            decodedPduStructList[i] = *decodedContext;
        } else {
            printf("DecodedPduContext is NULL. PDU Id - %d  |  %s |\n", i, pduContexts[i].correlationId);
        }
    }
    printf("\n");
}

void *decodeSingle(DirectPduContext *pduContext) {
    char *correlationId = pduContext->correlationId;
    int length = pduContext->length;
//    printf("Started decoding - %d\n", length);
    int startIndex = pduContext->start;
    uint8_t *pduBuffer = pduContext->pduBuffer;
//    printf("Correlation Id - %s\n", correlationId);
    ByteBufferContext *bufferContext = malloc(sizeof(ByteBufferContext));
    bufferContext->buffer = pduBuffer;
    bufferContext->readIndex = startIndex;
    bufferContext->limit = length;
    bufferContext->initialReadIndex = startIndex;

    SmppHeader *smppHeader = malloc(sizeof(SmppHeader));

    smppHeader->commandLength = readUint32(bufferContext);
    smppHeader->commandId = readUint32(bufferContext);
    smppHeader->commandStatus = readUint32(bufferContext);
    smppHeader->sequenceNumber = readUint32(bufferContext);

    printf("Limit - %ld\n", bufferContext->limit);
    printf("CommandLength - %d\n", smppHeader->commandLength);
    printf("CommandId - %d\n", smppHeader->commandId);
    printf("CommandStatus - %d\n", smppHeader->commandStatus);
    printf("SequenceNumber - %d\n", smppHeader->sequenceNumber);
    if (smppHeader->commandId == 4) {
        SubmitSmReq *submitSmReq = decodeSubmitSm(pduContext, bufferContext);
        submitSmReq->header = smppHeader;
        DecodedContext *decodedContext = malloc(sizeof(DecodedContext));
        decodedContext->commandId = 4;
        decodedContext->pduStruct = submitSmReq;
        decodedContext->correlationId = pduContext->correlationId;
        return decodedContext;
    }
    return 0;
}

SubmitSmReq *decodeSubmitSm(PduContext *pduContext, ByteBufferContext *context) {
    ByteBufferContext bufferContext = *context;
//    printf("Decoding SubmitSm\n");
    SubmitSmReq *submitSm = malloc(sizeof(SubmitSmReq));
    submitSm->serviceType = readNullTerminatedString(&bufferContext, 6);
/*
    if (submitSm->serviceType != -1) {
        printf("Service Type - %s\n", submitSm->serviceType);
    }
*/
    Address *sourceAddress = malloc(sizeof(Address));;
    sourceAddress->ton = readUint8(&bufferContext);
    sourceAddress->npi = readUint8(&bufferContext);
    sourceAddress->addressValue = readNullTerminatedString(&bufferContext, 21);
    printAddress(sourceAddress);
    submitSm->sourceAddress = sourceAddress;

    Address *destinationAddress = malloc(sizeof(Address));
    destinationAddress->ton = readUint8(&bufferContext);
    destinationAddress->npi = readUint8(&bufferContext);
    destinationAddress->addressValue = readNullTerminatedString(&bufferContext, 21);
    printAddress(destinationAddress);
    submitSm->destinationAddress = destinationAddress;

    submitSm->esmClass = readUint8(&bufferContext);
    submitSm->protocolId = readUint8(&bufferContext);
    submitSm->priority = readUint8(&bufferContext);
    submitSm->scheduleDeliveryTime = readNullTerminatedString(&bufferContext, 17);
    submitSm->validityPeriod = readNullTerminatedString(&bufferContext, 17);
    submitSm->registeredDelivery = readUint8(&bufferContext);
    submitSm->replaceIfPresent = readUint8(&bufferContext);
    submitSm->dataCoding = readUint8(&bufferContext);
    submitSm->defaultMsgId = readUint8(&bufferContext);
    submitSm->smLength = readUint8(&bufferContext);
    if (submitSm->smLength > 0) {
        submitSm->shortMessage = readNBytes(&bufferContext, submitSm->smLength);
    }
    DecodedTlvContext tlvContext = decodeTlv(&bufferContext);
    submitSm->tlvList = tlvContext.tlvList;
    submitSm->tlvCount = tlvContext.nTlv;
    return submitSm;
}

DecodedTlvContext decodeTlv(ByteBufferContext *context) {
    DecodedTlvContext decodedTlvContext;
    uint16_t tlvCount = 0;
    Tlv *tlvList = malloc(sizeof(Tlv) * 10); // This need to be a dynamic list.
    printf("Tlv read-index - %d | limit - %d\n", context->readIndex, context->limit);
    while ((context->readIndex - context->initialReadIndex) < context->limit - 4 && tlvCount < 10) {
        uint16_t tag = readUint16(context);
        printf("tlv count  - %d\n", tlvCount);
        printf("tag - %d\n", tag);
        uint32_t length = readUint16(context);
        printf("length - %d\n", length);
        if ((context->readIndex - context->initialReadIndex) <= context->limit - length) {
            uint8_t *value = readNBytes(context, length);
            printf("VALUEEEEEEEEEEEEEEEEEEEEEEEEEE - %s\n", value);
            Tlv tlv = {tag, length, value};
            tlvList[tlvCount] = tlv;
            tlvCount++;
        } else {
            printf("Tlv remaining - %d - %d\n", (context->readIndex - context->initialReadIndex), context->limit - length);
            break;
        }
    }
    decodedTlvContext.nTlv = tlvCount;
    decodedTlvContext.tlvList = tlvList;
    return decodedTlvContext;
}

void printAddress(Address *address) {
    printf("ton - %d\n", address->ton);
    printf("npi - %d\n", address->npi);
    printf("address - %s\n\n\n", (char *) address->addressValue);
}
