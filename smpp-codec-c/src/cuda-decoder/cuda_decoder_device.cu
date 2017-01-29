//
// Created by psaw on 9/17/16.
//

#include "cuda_decoder_device.h"
#include <stdio.h>
#include <host_defines.h>

__device__ void decodeSubmitSm(CudaBaseSmReq *submitSmReq,
		ByteBufferContext *context);

__device__ char * strcpy(char *dest, const char *src) {
	int i = 0;
	do {
		dest[i] = src[i];
	} while (src[i++] != 0);
	return dest;
}

__device__ void decodeSinglePdu(CudaPduContext *pduContext,
		CudaDecodedContext *decodedPduStruct, uint8_t *pduBuffer) {
	char *correlationId = pduContext->correlationId;
	int length = pduContext->length;
	int startIndex = pduContext->start;
//    printf("Started decoding Cuda - %d\n", length);
//    printf("Correlation Id - %s\n", correlationId);
	ByteBufferContext bufferContext;
	bufferContext.buffer = pduBuffer;
	bufferContext.readIndex = (uint64_t) startIndex;
	bufferContext.limit = (uint64_t) length;
//    printf("LIMIT %ld | Length %ld\n",bufferContext.limit,length);
	SmppHeader *smppHeader = &((decodedPduStruct->pduStruct).header);

	smppHeader->commandLength = readUint32(&bufferContext);
	smppHeader->commandId = readUint32(&bufferContext);
	smppHeader->commandStatus = readUint32(&bufferContext);
	smppHeader->sequenceNumber = readUint32(&bufferContext);

//    printf("CommandLength - %d\n", smppHeader->commandLength);
//    printf("CommandId - %d\n", smppHeader->commandId);
//    printf("CommandStatus - %d\n", smppHeader->commandStatus);
//    printf("SequenceNumber - %d\n", smppHeader->sequenceNumber);
	if (smppHeader->commandId == 4 || smppHeader->commandId == 5) {
		decodeSubmitSm(&(decodedPduStruct->pduStruct), &bufferContext);
		CudaBaseSmReq *submitSmReq = &(decodedPduStruct->pduStruct);
		decodedPduStruct->commandId = smppHeader->commandId;
		strcpy(decodedPduStruct->correlationId, pduContext->correlationId);
	}
}

__device__ void decodeSubmitSm(CudaBaseSmReq *submitSm,
		ByteBufferContext *context) {
	ByteBufferContext bufferContext = *context;
//    printf("Decoding SubmitSm\n");
	readNullTerminatedString(&bufferContext, 6, submitSm->serviceType);
	/*
	 if (submitSm->serviceType != -1) {
	 printf("Service Type - %s\n", submitSm->serviceType);
	 }
	 */
	CudaAddress *sourceAddress = &(submitSm->sourceAddress);
	sourceAddress->ton = readUint8(&bufferContext);
	sourceAddress->npi = readUint8(&bufferContext);
	readNullTerminatedString(&bufferContext, 21, sourceAddress->addressValue);
//    printAddress(sourceAddress);

	CudaAddress *destinationAddress = &(submitSm->destinationAddress);
	destinationAddress->ton = readUint8(&bufferContext);
	destinationAddress->npi = readUint8(&bufferContext);
	readNullTerminatedString(&bufferContext, 21,
			destinationAddress->addressValue);
//    printAddress(destinationAddress);

	submitSm->esmClass = readUint8(&bufferContext);
	submitSm->protocolId = readUint8(&bufferContext);
	submitSm->priority = readUint8(&bufferContext);
	readNullTerminatedString(&bufferContext, 17,
			submitSm->scheduleDeliveryTime);
	readNullTerminatedString(&bufferContext, 17, submitSm->validityPeriod);
	submitSm->registeredDelivery = readUint8(&bufferContext);
	submitSm->replaceIfPresent = readUint8(&bufferContext);
	submitSm->dataCoding = readUint8(&bufferContext);
	submitSm->defaultMsgId = readUint8(&bufferContext);
	submitSm->smLength = readUint8(&bufferContext);
	if (submitSm->smLength > 0) {
		readNBytes(&bufferContext, submitSm->smLength, submitSm->shortMessage);
	}
}

