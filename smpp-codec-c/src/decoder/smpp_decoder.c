/*
Copyright (c) 2016 Prabath Weerasinghe - MSC Project - SMPP Codec. All rights reserved.
Created by prabath on 6/26/16.
*/

#include <sys/types.h>
#include <sys/syscall.h>

#include "smpp_pdu_struct.h"
#include "smpp_util.h"

void printAddress(Address *address);

SubmitSmReq *decodeSubmitSm(PduContext *pduContext, ByteBufferContext *context);

void *decode(void *threadParam) {
    ThreadParam *context = (ThreadParam *) threadParam;

    pid_t x = syscall(__NR_gettid);

    PduContext *pduContexts = context->pduContexts;
    DecodedContext *decodedPduStructList = context->decodedPduStructList;
    int length = context->length;
    int startIndex = context->startIndex;
    int i;
    printf("Id %d - Start - %d | End - %d\n", x, i, startIndex + length);
    for (i = startIndex; i < startIndex + length; i++) {
        void *decodedPduStruct = decodeSingle(&pduContexts[i]);
        if (decodedPduStruct != 0) {
            DecodedContext *decodedContext = (DecodedContext *) decodedPduStruct;
            decodedPduStructList[i] = *decodedContext;
        }
    }
    printf("\n");
}

void *decodeSingle(PduContext *pduContext) {
    char *correlationId = pduContext->correlationId;
    long length = pduContext->pduBufferLength;
    uint8_t *pduBuffer = pduContext->pduBuffer;
//    printf("Correlation Id - %s\n", correlationId);
//    printf("Started decoding - %d\n", length);
    ByteBufferContext bufferContext = {pduBuffer, 0, length};

    SmppHeader *smppHeader = malloc(sizeof(SmppHeader));

    smppHeader->commandLength = readUint32(&bufferContext);
    smppHeader->commandId = readUint32(&bufferContext);
    smppHeader->commandStatus = readUint32(&bufferContext);
    smppHeader->sequenceNumber = readUint32(&bufferContext);

//    printf("CommandLength - %d\n", smppHeader->commandLength);
//    printf("CommandId - %d\n", smppHeader->commandId);
//    printf("CommandStatus - %d\n", smppHeader->commandStatus);
//    printf("SequenceNumber - %d\n", smppHeader->sequenceNumber);

    if (smppHeader->commandId == 4) {
        SubmitSmReq *submitSmReq = decodeSubmitSm(pduContext, &bufferContext);
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
    return submitSm;
}

void printAddress(Address *address) {
//    printf("ton - %d\n", address->ton);
//    printf("npi - %d\n", address->npi);
//    printf("address - %s\n", (char *) address->addressValue);
}
