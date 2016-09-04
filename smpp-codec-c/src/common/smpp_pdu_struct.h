#ifndef N_THREAD
#define N_THREAD 3
#endif

#ifndef SMPP_CODEC_C_SMPP_PDU_STRUCT_H
#define SMPP_CODEC_C_SMPP_PDU_STRUCT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#include "com_cloudhopper_smpp_transcoder_asynchronous_DefaultAsynchronousDecoder.h"

typedef struct jmethod_cache {
    jmethodID listSizeMethod;
    jmethodID decodedContextContainerMethod;
    jmethodID addDecodedContextMethodId;
    jmethodID listGetMethod;
    jmethodID decodedPduContextconstructor;
    jmethodID constructor;
    jmethodID setCommandLength;
    jmethodID setCommandStatus;
    jmethodID setSequenceNumber;
    jmethodID setServiceType;
    jmethodID setSourceAddress;
    jmethodID setDestAddress;
    jmethodID setEsmClass;
    jmethodID setProtocolId;
    jmethodID setPriority;
    jmethodID setScheduleDeliveryTime;
    jmethodID setValidityPeriod;
    jmethodID setRegisteredDelivery;
    jmethodID setReplaceIfPresent;
    jmethodID setDataCoding;
    jmethodID setDefaultMsgId;
    jmethodID setShortMessage;
    jmethodID addressConstructor;
    jmethodID setTon;
    jmethodID setNpi;
    jmethodID setAddress;
} JmethodCache;

typedef struct jfield_cache {
    jfieldID correlationIdFieldId;
    jfieldID byteBufferFieldId;

}JfieldCache;

typedef struct jclass_cache {
    jclass arrayListClass;
    jclass pduContextClass;
    jclass decodedPduContextClass;
    jclass submitSmClass;
    jclass addressClass;
} JclassCache;

typedef struct pdu_context_struct {
    char *correlationId;
    uint8_t *pduBuffer;
    long pduBufferLength;
    jbyteArray *byteArrayNative;
    jstring correlationIdString;
} PduContext;

typedef struct decoded_context_struct {
    char *correlationId;
    uint32_t commandId;
    void *pduStruct;
} DecodedContext;

typedef struct smpp_header_struct {
    uint32_t commandLength;
    uint32_t commandId;
    uint32_t commandStatus;
    uint32_t sequenceNumber;


} SmppHeader;

typedef struct address_struct {
    int8_t ton;
    int8_t npi;
    char *addressValue;
} Address;

typedef struct tlv_struct {
    uint16_t tag;
    uint32_t length;
    uint8_t *value;
    char *tagName;
} Tlv;

typedef struct submit_sm_req_struct {
    SmppHeader *header;
    char *serviceType;
    Address *sourceAddress;
    Address *destinationAddress;
    uint8_t esmClass;
    uint8_t protocolId;
    uint8_t priority;
    char *scheduleDeliveryTime;
    char *validityPeriod;
    uint8_t registeredDelivery;
    uint8_t replaceIfPresent;
    uint8_t dataCoding;
    uint8_t defaultMsgId;
    uint8_t smLength;
    uint8_t *shortMessage;
    uint16_t tlvCount;
    Tlv *tlvList;
} SubmitSmReq;

typedef struct thread_param {
    int startIndex;
    int length;
    PduContext *pduContexts;
    DecodedContext *decodedPduStructList;
} ThreadParam;

//SubmitSmReq *decodeSubmitSm(PduContext *pduContext);
void *decodeSingle(PduContext *pduContext);

void *decode(void *threadParam);

int test();

#endif
