#include <time.h>
#include "smpp_codec.h"

jobject *createPdu(JNIEnv *env, DecodedContext *decodedContext);

jobject *getAddress(JNIEnv *env, Address *address);

jobject *getCudaAddress(JNIEnv *env, CudaAddress *address);

jobject *createPduOnCuda(JNIEnv *env, CudaDecodedContext *decodedContext);

void cacheJClass(JNIEnv *env);

void cacheJMethod(JNIEnv *env);

void cacheJField(JNIEnv *env);

void freeDecodedContext(DecodedContext *decodedContext);

jobject startDecodingBatch(JNIEnv *env, jobject thisObj, jobject param);

jobject
decodeWithPthread(JNIEnv *env, jobject pduContainerBuffer, jint size, jint correlationIdLength);

jobject
decodeWithCuda(JNIEnv *env, jobject pduContainerBuffer, jint size, jint correlationIdLength);

static JclassCache jClassCache;

static JmethodCache jmethodCache;

static JfieldCache jfieldCache;

static time_t start_t, end_t;
static double diff_t;
static struct timespec tstart = {0, 0}, tend = {0, 0};

log4c_category_t *statLogger;

JNIEXPORT void JNICALL Java_com_cloudhopper_smpp_transcoder_asynchronous_DefaultAsynchronousDecoder_initialize
        (JNIEnv *env, jobject thisObj, jstring configurationFilePath) {
    printf("Caching JClass 1, JMethodId and JFieldId values.\n");
    cacheJClass(env);
    cacheJMethod(env);
    cacheJField(env);
    statLogger = log4c_category_get("stat_logger");
    jboolean isCopy = 1;
    const char *propertyFilePath = (*env)->GetStringUTFChars(env, configurationFilePath, &isCopy);
    CodecConfiguration configuration = {propertyFilePath};
    init(&configuration);
}

JNIEXPORT jobject JNICALL Java_com_cloudhopper_smpp_transcoder_asynchronous_DefaultAsynchronousDecoder_decodePDUDirect
        (JNIEnv *env, jobject thisObj, jobject pduContainerBuffer, jint size, jint correlationIdLength) {
    time(&start_t);
    clock_gettime(CLOCK_MONOTONIC, &tstart);
    if(cpuDecodeThreshold == 0){
     return decodePduDirect(env, pduContainerBuffer, size, correlationIdLength);
    }
    if(size < cpuDecodeThreshold){
      return decodeWithPthread(env, pduContainerBuffer, size, correlationIdLength);
    } else {
          return decodeWithCuda(env, pduContainerBuffer, size, correlationIdLength);
    }
}

JNIEXPORT void JNICALL Java_com_cloudhopper_smpp_transcoder_asynchronous_DefaultAsynchronousDecoder_startTuner(
        JNIEnv *env, jobject thisObj, jobject pduContainerBuffer, jint size, jint correlationIdLength) {
    log4c_category_log(statLogger, LOG4C_PRIORITY_DEBUG, "Preparing for performance tuning .....");
    printf("Starting performance tuning \n");
    fflush(stdout);
    jlong bufferCapacity = (*env)->GetDirectBufferCapacity(env, pduContainerBuffer);
    uint8_t *pduBuffers = (uint8_t *) (*env)->GetDirectBufferAddress(env, pduContainerBuffer);
    DecoderMetadata decoderMetadata = {pduBuffers, (uint32_t) size, (uint64_t) bufferCapacity, (uint32_t) correlationIdLength};
    if (useGpu) {
        startPerfTuner(decoderMetadata);
    } else {
        startPerfTunerPthread(decoderMetadata);
    }
}

jobject
decodeWithCuda(JNIEnv *env, jobject pduContainerBuffer, jint size, jint correlationIdLength) {
    log4c_category_log(statLogger, LOG4C_PRIORITY_DEBUG, "Decoding Cuda");

    jlong bufferCapacity = (*env)->GetDirectBufferCapacity(env, pduContainerBuffer);
    uint8_t *pduBuffers = (uint8_t *) (*env)->GetDirectBufferAddress(env, pduContainerBuffer);
    DecoderMetadata decoderMetadata = {pduBuffers, (uint32_t) size, (uint64_t) bufferCapacity, (uint32_t) correlationIdLength};
    decoderMetadata.blockDim = blockDimProdction;
    decoderMetadata.gridDim = gridDimProduction;
    CudaDecodedContext *decodedPduStructList = decodeGpu(decoderMetadata);

    time(&end_t);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    diff_t = difftime(end_t, start_t);

    printf("Cuda Decoding Completed - TimeTaken %.5f \n",
           ((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) -
           ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

    jclass decodedContextContainerClass = jClassCache.arrayListClass;
    jmethodID decodedContextContainerMethod = jmethodCache.decodedContextContainerMethod;
    jobject decodedContextContainer = (*env)->NewObject(env, decodedContextContainerClass, decodedContextContainerMethod);
    jmethodID addDecodedContextMethodId = jmethodCache.addDecodedContextMethodId;

    int i;
    for (i = 0; i < 1; i++) {
        CudaDecodedContext *decodedContext = &decodedPduStructList[i];
        if (decodedContext != 0) {
            jobject *pdu = createPduOnCuda(env, decodedContext);
            if (pdu != 0) {
                jclass decodedPduContextClass = jClassCache.decodedPduContextClass;
                jmethodID decodedPduContextconstructor = jmethodCache.decodedPduContextconstructor;
                jobject decodedPduContextObject = (*env)->NewObject(env,
                                                                    decodedPduContextClass,
                                                                    decodedPduContextconstructor,
                                                                    (*env)->NewStringUTF(env,
                                                                                         decodedContext->correlationId),
                                                                    pdu);
                if ((*env)->ExceptionOccurred(env)) {
                    printf("Exception occurred\n\n");
                    return decodedContextContainer;
                }
                fflush(stdout);
                (*env)->CallObjectMethod(env, decodedContextContainer, addDecodedContextMethodId,
                                         decodedPduContextObject);
                printf("PDU Success. | Command Id %d \n", decodedContext->commandId);
                if ((*env)->ExceptionOccurred(env)) {
                    printf("Exception occurred\n\n");
                    return decodedContextContainer;
                }

            } else {
                printf("PDU context is null. | Command Id %d \n", decodedContext->commandId);
            }
        } else {
            printf("NULL decodedPduStructure\n");
        }
    }
    free(decodedPduStructList);
    printf("Returning\n");
    fflush(stdout);
    return decodedContextContainer;
}

jobject
decodeWithPthread(JNIEnv *env, jobject pduContainerBuffer, jint size, jint correlationIdLength) {
    printf("Decoding Pthread\n");
    jlong bufferCapacity = (*env)->GetDirectBufferCapacity(env, pduContainerBuffer);
    uint8_t *pduBuffers = (uint8_t *) (*env)->GetDirectBufferAddress(env, pduContainerBuffer);

    DecoderMetadata decoderMetadata = {pduBuffers, (uint32_t) size, (uint64_t) bufferCapacity, (uint32_t) correlationIdLength};
    DecodedContext *decodedPduStructList = decodePthread(decoderMetadata);

    time(&end_t);
    clock_gettime(CLOCK_MONOTONIC, &tend);
    diff_t = difftime(end_t, start_t);
    printf("Pthread Decoding Completed - TimeTaken %.5f \n",
           ((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) -
           ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

    jclass decodedContextContainerClass = jClassCache.arrayListClass;
    jmethodID decodedContextContainerMethod = jmethodCache.decodedContextContainerMethod;
    jobject decodedContextContainer = (*env)->NewObject(env, decodedContextContainerClass,
                                                        decodedContextContainerMethod);
    jmethodID addDecodedContextMethodId = jmethodCache.addDecodedContextMethodId;
    int i;
    for (i = 0; i < 1; i++) {
        DecodedContext *decodedContext = &decodedPduStructList[i];
        if (decodedContext != 0) {
            jobject *pdu = createPdu(env, decodedContext);
            if (pdu != 0) {
                jclass decodedPduContextClass = jClassCache.decodedPduContextClass;
                jmethodID decodedPduContextconstructor = jmethodCache.decodedPduContextconstructor;
                jobject decodedPduContextObject = (*env)->NewObject(env,
                                                                    decodedPduContextClass,
                                                                    decodedPduContextconstructor,
                                                                    (*env)->NewStringUTF(env,
                                                                                         decodedContext->correlationId),
                                                                    pdu);
                if ((*env)->ExceptionOccurred(env)) {
                    printf("Exception occurred\n\n");
                    freeDecodedContext(decodedContext);
                }
                fflush(stdout);
                (*env)->CallObjectMethod(env, decodedContextContainer, addDecodedContextMethodId,
                                         decodedPduContextObject);
            } else {
                printf("PDU context is null. | Command Id %d \n", decodedContext->commandId);
            }
        } else {
            printf("NULL decodedPduStructure\n");
        }
    }
    free(decodedPduStructList);
    printf("Returning\n");
    fflush(stdout);
    return decodedContextContainer;
}

jobject *createPdu(JNIEnv *env, DecodedContext *decodedContext) {
    printf("SMPP PDU type %d\n", decodedContext->commandId);
    if (decodedContext->commandId == 4 || decodedContext->commandId == 5) {
        BaseSmReq baseSmReq = *(BaseSmReq *) decodedContext->pduStruct;
        printf("BaseSm Request - %d\n", baseSmReq.esmClass);
        printf("BaseSm setCommandLength - %d\n", baseSmReq.header->commandLength);

        jclass baseSmClass;
        baseSmClass = (decodedContext->commandId == 4) ? jClassCache.submitSmClass : jClassCache.deliverSmClass;
//        printf("SubmitSm class found - %s\n", baseSmClass);

        jmethodID constructor = jmethodCache.constructor;
        jobject baseSmObject = (*env)->NewObject(env, baseSmClass, constructor);

        printf("BaseSm setCommandLength - %d\n", baseSmReq.header->commandLength);
        jmethodID setCommandLength = jmethodCache.setCommandLength;
        (*env)->CallObjectMethod(env, baseSmObject, setCommandLength, baseSmReq.header->commandLength);

        printf("BaseSm setCommandStatus - %d\n", baseSmReq.header->commandStatus);
        jmethodID setCommandStatus = jmethodCache.setCommandStatus;
        (*env)->CallObjectMethod(env, baseSmObject, setCommandStatus, baseSmReq.header->commandStatus);

        printf("BaseSm sequenceNumber - %d\n", baseSmReq.header->sequenceNumber);
        jmethodID setSequenceNumber = jmethodCache.setSequenceNumber;
        (*env)->CallObjectMethod(env, baseSmObject, setSequenceNumber, baseSmReq.header->sequenceNumber);

        printf("BaseSm service type- %s\n", baseSmReq.serviceType);
        jmethodID setServiceType = jmethodCache.setServiceType;
        (*env)->CallObjectMethod(env, baseSmObject, setServiceType,
                                 (*env)->NewStringUTF(env, baseSmReq.serviceType));

        printf("BaseSm Source Address - %s\n", baseSmReq.sourceAddress->addressValue);
        jmethodID setSourceAddress = jmethodCache.setSourceAddress;
        jobject *srcAddress = getAddress(env, baseSmReq.sourceAddress);
        (*env)->CallObjectMethod(env, baseSmObject, setSourceAddress, srcAddress);

        printf("BaseSm Destination ton - %d\n", baseSmReq.destinationAddress->ton);
        jmethodID setDestAddress = jmethodCache.setDestAddress;
        (*env)->CallObjectMethod(env, baseSmObject, setDestAddress, getAddress(env, baseSmReq.destinationAddress));
        printf("BaseSm Destination Address - %s\n", baseSmReq.destinationAddress->addressValue);

        jmethodID setEsmClass = jmethodCache.setEsmClass;
        (*env)->CallObjectMethod(env, baseSmObject, setEsmClass, baseSmReq.esmClass);

        jmethodID setProtocolId = jmethodCache.setProtocolId;
        (*env)->CallObjectMethod(env, baseSmObject, setProtocolId, baseSmReq.protocolId);

        jmethodID setPriority = jmethodCache.setPriority;
        (*env)->CallObjectMethod(env, baseSmObject, setPriority, baseSmReq.priority);

        jmethodID setScheduleDeliveryTime = jmethodCache.setScheduleDeliveryTime;
        (*env)->CallObjectMethod(env, baseSmObject, setScheduleDeliveryTime,
                                 (*env)->NewStringUTF(env, baseSmReq.scheduleDeliveryTime));

        jmethodID setValidityPeriod = jmethodCache.setValidityPeriod;
        (*env)->CallObjectMethod(env, baseSmObject, setValidityPeriod,
                                 (*env)->NewStringUTF(env, baseSmReq.validityPeriod));

        jmethodID setRegisteredDelivery = jmethodCache.setRegisteredDelivery;
        (*env)->CallObjectMethod(env, baseSmObject, setRegisteredDelivery, baseSmReq.registeredDelivery);

        jmethodID setReplaceIfPresent = jmethodCache.setReplaceIfPresent;
        (*env)->CallObjectMethod(env, baseSmObject, setReplaceIfPresent, baseSmReq.replaceIfPresent);


        jmethodID setDataCoding = jmethodCache.setDataCoding;
        (*env)->CallObjectMethod(env, baseSmObject, setDataCoding, baseSmReq.dataCoding);


        jmethodID setDefaultMsgId = jmethodCache.setDefaultMsgId;
        (*env)->CallObjectMethod(env, baseSmObject, setDefaultMsgId, baseSmReq.defaultMsgId);

        if (baseSmReq.smLength > 0) {
            jmethodID setShortMessage = jmethodCache.setShortMessage;
            jbyteArray shortMsgArray = (*env)->NewByteArray(env, baseSmReq.smLength);
            (*env)->SetByteArrayRegion(env, shortMsgArray, 0, baseSmReq.smLength, (const jbyte *) baseSmReq.shortMessage);
            (*env)->CallObjectMethod(env, baseSmObject, setShortMessage, shortMsgArray);
//            printf("Short Message- %s\n", baseSmReq.shortMessage);
        }
        return baseSmObject;
    }
    return 0;
}

jobject *createPduOnCuda(JNIEnv *env, CudaDecodedContext *decodedContext) {
    printf("SMPP PDU type %d\n", decodedContext->commandId);
    if (decodedContext->commandId == 4 || decodedContext->commandId == 5) {
        CudaBaseSmReq baseSmReq = decodedContext->pduStruct;
//        printf("SubmitSm Request - %d\n", baseSmReq.esmClass);
//        printf("SubmitSm setCommandLength - %ld\n", baseSmReq.header.commandLength);
        jclass baseSmClass;
        baseSmClass = (decodedContext->commandId == 4) ? jClassCache.submitSmClass : jClassCache.deliverSmClass;
//        printf("SubmitSm class found - %s\n", submitSmClass);

        jmethodID constructor = jmethodCache.constructor;
        jobject baseSmObject = (*env)->NewObject(env, baseSmClass, constructor);

//        printf("SubmitSm setCommandLength - %d\n", baseSmReq.header.commandLength);
        jmethodID setCommandLength = jmethodCache.setCommandLength;
        (*env)->CallObjectMethod(env, baseSmObject, setCommandLength, baseSmReq.header.commandLength);

//        printf("SubmitSm setCommandStatus - %d\n", baseSmReq.header.commandStatus);
        jmethodID setCommandStatus = jmethodCache.setCommandStatus;
        (*env)->CallObjectMethod(env, baseSmObject, setCommandStatus, baseSmReq.header.commandStatus);

//        printf("SubmitSm sequenceNumber - %d\n", baseSmReq.header.sequenceNumber);
        jmethodID setSequenceNumber = jmethodCache.setSequenceNumber;
        (*env)->CallObjectMethod(env, baseSmObject, setSequenceNumber, baseSmReq.header.sequenceNumber);

        jmethodID setServiceType = jmethodCache.setServiceType;
        (*env)->CallObjectMethod(env, baseSmObject, setServiceType,
                                 (*env)->NewStringUTF(env, baseSmReq.serviceType));

        jmethodID setSourceAddress = jmethodCache.setSourceAddress;
        jobject *srcAddress = getCudaAddress(env, &baseSmReq.sourceAddress);

        (*env)->CallObjectMethod(env, baseSmObject, setSourceAddress, srcAddress);

//        printf("SubmitSm Destination Address - %s\n", baseSmReq.destinationAddress.addressValue);
        jmethodID setDestAddress = jmethodCache.setDestAddress;
        (*env)->CallObjectMethod(env, baseSmObject, setDestAddress, getCudaAddress(env, &baseSmReq.destinationAddress));
//        printf("SubmitSm Destination Address - %s\n", baseSmReq.destinationAddress.addressValue);

        jmethodID setEsmClass = jmethodCache.setEsmClass;
        (*env)->CallObjectMethod(env, baseSmObject, setEsmClass, baseSmReq.esmClass);
//        printf("Esm Class - %d\n", baseSmReq.esmClass);

        jmethodID setProtocolId = jmethodCache.setProtocolId;
        (*env)->CallObjectMethod(env, baseSmObject, setProtocolId, baseSmReq.protocolId);

        jmethodID setPriority = jmethodCache.setPriority;
        (*env)->CallObjectMethod(env, baseSmObject, setPriority, baseSmReq.priority);

        jmethodID setScheduleDeliveryTime = jmethodCache.setScheduleDeliveryTime;
        (*env)->CallObjectMethod(env, baseSmObject, setScheduleDeliveryTime,
                                 (*env)->NewStringUTF(env, baseSmReq.scheduleDeliveryTime));

        jmethodID setValidityPeriod = jmethodCache.setValidityPeriod;
        (*env)->CallObjectMethod(env, baseSmObject, setValidityPeriod,
                                 (*env)->NewStringUTF(env, baseSmReq.validityPeriod));

        jmethodID setRegisteredDelivery = jmethodCache.setRegisteredDelivery;
        (*env)->CallObjectMethod(env, baseSmObject, setRegisteredDelivery, baseSmReq.registeredDelivery);

        jmethodID setReplaceIfPresent = jmethodCache.setReplaceIfPresent;
        (*env)->CallObjectMethod(env, baseSmObject, setReplaceIfPresent, baseSmReq.replaceIfPresent);


        jmethodID setDataCoding = jmethodCache.setDataCoding;
        (*env)->CallObjectMethod(env, baseSmObject, setDataCoding, baseSmReq.dataCoding);


        jmethodID setDefaultMsgId = jmethodCache.setDefaultMsgId;
        (*env)->CallObjectMethod(env, baseSmObject, setDefaultMsgId, baseSmReq.defaultMsgId);
//        printf("Msg Id - %d\n", baseSmReq.defaultMsgId);
        if (baseSmReq.smLength > 0) {
//            printf("Message Exists- %d\n", baseSmReq.smLength);
            jmethodID setShortMessage = jmethodCache.setShortMessage;
            jbyteArray shortMsgArray = (*env)->NewByteArray(env, baseSmReq.smLength);
            (*env)->SetByteArrayRegion(env, shortMsgArray, 0, baseSmReq.smLength, (const jbyte *) baseSmReq.shortMessage);
            (*env)->CallObjectMethod(env, baseSmObject, setShortMessage, shortMsgArray);
//            printf("Short Message- %s\n", baseSmReq.shortMessage);
        }
        return baseSmObject;
    }
    return 0;
}

jobject *getAddress(JNIEnv *env, Address *address) {

    jclass addressClass = jClassCache.addressClass;

    jmethodID addressConstructor = jmethodCache.addressConstructor;
    jobject addressObject = (*env)->NewObject(env, addressClass, addressConstructor);

    jmethodID setTon = jmethodCache.setTon;
    (*env)->CallObjectMethod(env, addressObject, setTon, address->ton);

    jmethodID setNpi = jmethodCache.setNpi;
    (*env)->CallObjectMethod(env, addressObject, setNpi, address->npi);

    jmethodID setAddress = jmethodCache.setAddress;
    jstring addressJstring = (*env)->NewStringUTF(env, address->addressValue);
    (*env)->CallObjectMethod(env, addressObject, setAddress, addressJstring);

    return &addressObject;
}

jobject *getCudaAddress(JNIEnv *env, CudaAddress *address) {

    jclass addressClass = jClassCache.addressClass;

    jmethodID addressConstructor = jmethodCache.addressConstructor;
    jobject addressObject = (*env)->NewObject(env, addressClass, addressConstructor);

    jmethodID setTon = jmethodCache.setTon;
    (*env)->CallObjectMethod(env, addressObject, setTon, address->ton);

    jmethodID setNpi = jmethodCache.setNpi;
    (*env)->CallObjectMethod(env, addressObject, setNpi, address->npi);

    jmethodID setAddress = jmethodCache.setAddress;
    jstring addressJstring = (*env)->NewStringUTF(env, address->addressValue);
    (*env)->CallObjectMethod(env, addressObject, setAddress, addressJstring);

    return &addressObject;
}

void cacheJClass(JNIEnv *env) {
    jClassCache = (JclassCache) {};

    jclass arrayListClassLocal = (*env)->FindClass(env, "Ljava/util/ArrayList;");
    jClassCache.arrayListClass = (*env)->NewGlobalRef(env, arrayListClassLocal);

    jclass pduContextClassLocal = (*env)->FindClass(env,
                                                    "Lcom/cloudhopper/smpp/transcoder/asynchronous/AsynchronousContext;");
    jClassCache.pduContextClass = (*env)->NewGlobalRef(env, pduContextClassLocal);

    jclass decodedPduContextClassLocal = (*env)->FindClass(env,
                                                           "Lcom/cloudhopper/smpp/transcoder/asynchronous/DecodedPduContext;");
    jClassCache.decodedPduContextClass = (*env)->NewGlobalRef(env, decodedPduContextClassLocal);

    jclass submitSmClassLocal = (*env)->FindClass(env, "Lcom/cloudhopper/smpp/pdu/SubmitSm;");
    jClassCache.submitSmClass = (*env)->NewGlobalRef(env, submitSmClassLocal);

    jclass deliverSmClassLocal = (*env)->FindClass(env, "Lcom/cloudhopper/smpp/pdu/DeliverSm;");
    jClassCache.deliverSmClass = (*env)->NewGlobalRef(env, deliverSmClassLocal);

    jclass addressClassLocal = (*env)->FindClass(env, "Lcom/cloudhopper/smpp/type/Address;");
    jClassCache.addressClass = (*env)->NewGlobalRef(env, addressClassLocal);
}

void cacheJMethod(JNIEnv *env) {
    jmethodCache = (JmethodCache) {};

    jclass arrayListClass = jClassCache.arrayListClass;
    jmethodCache.listSizeMethod = (*env)->GetMethodID(env, arrayListClass, "size", "()I");

    jclass decodedContextContainerClass = jClassCache.arrayListClass;
    jmethodCache.decodedContextContainerMethod =
            (*env)->GetMethodID(env, decodedContextContainerClass, "<init>", "()V");
    jmethodCache.addDecodedContextMethodId = (*env)->GetMethodID(env, decodedContextContainerClass, "add",
                                                                 "(Ljava/lang/Object;)Z");

    jmethodCache.listGetMethod = (*env)->GetMethodID(env, arrayListClass, "get", "(I)Ljava/lang/Object;");

    jclass decodedPduContextClass = jClassCache.decodedPduContextClass;
    jmethodCache.decodedPduContextconstructor = (*env)->GetMethodID(env,
                                                                    decodedPduContextClass,
                                                                    "<init>",
                                                                    "(Ljava/lang/String;Lcom/cloudhopper/smpp/pdu/Pdu;)V");
    jclass submitSmClass = jClassCache.submitSmClass;
    jmethodCache.constructor = (*env)->GetMethodID(env, submitSmClass, "<init>", "()V");
    jmethodCache.setCommandLength = (*env)->GetMethodID(env, submitSmClass, "setCommandLength", "(I)V");
    jmethodCache.setCommandStatus = (*env)->GetMethodID(env, submitSmClass, "setCommandStatus", "(I)V");
    jmethodCache.setSequenceNumber = (*env)->GetMethodID(env, submitSmClass, "setSequenceNumber", "(I)V");
    jmethodCache.setServiceType = (*env)->GetMethodID(env, submitSmClass, "setServiceType", "(Ljava/lang/String;)V");
    jmethodCache.setSourceAddress = (*env)->GetMethodID(env, submitSmClass, "setSourceAddress",
                                                        "(Lcom/cloudhopper/smpp/type/Address;)V");
    jmethodCache.setDestAddress = (*env)->GetMethodID(env, submitSmClass, "setDestAddress",
                                                      "(Lcom/cloudhopper/smpp/type/Address;)V");
    jmethodCache.setEsmClass = (*env)->GetMethodID(env, submitSmClass, "setEsmClass", "(B)V");
    jmethodCache.setProtocolId = (*env)->GetMethodID(env, submitSmClass, "setProtocolId", "(B)V");
    jmethodCache.setPriority = (*env)->GetMethodID(env, submitSmClass, "setPriority", "(B)V");
    jmethodCache.setScheduleDeliveryTime = (*env)->GetMethodID(env, submitSmClass, "setScheduleDeliveryTime",
                                                               "(Ljava/lang/String;)V");
    jmethodCache.setValidityPeriod = (*env)->GetMethodID(env, submitSmClass, "setValidityPeriod",
                                                         "(Ljava/lang/String;)V");
    jmethodCache.setRegisteredDelivery = (*env)->GetMethodID(env, submitSmClass, "setRegisteredDelivery", "(B)V");
    jmethodCache.setReplaceIfPresent = (*env)->GetMethodID(env, submitSmClass, "setReplaceIfPresent", "(B)V");
    jmethodCache.setDataCoding = (*env)->GetMethodID(env, submitSmClass, "setDataCoding", "(B)V");
    jmethodCache.setDefaultMsgId = (*env)->GetMethodID(env, submitSmClass, "setDefaultMsgId", "(B)V");
    jmethodCache.setShortMessage = (*env)->GetMethodID(env, submitSmClass, "setShortMessage", "([B)V");

    jclass addressClass = jClassCache.addressClass;
    jmethodCache.addressConstructor = (*env)->GetMethodID(env, addressClass, "<init>", "()V");
    jmethodCache.setTon = (*env)->GetMethodID(env, addressClass, "setTon", "(B)V");
    jmethodCache.setNpi = (*env)->GetMethodID(env, addressClass, "setNpi", "(B)V");
    jmethodCache.setAddress = (*env)->GetMethodID(env, addressClass, "setAddress", "(Ljava/lang/String;)V");

}

void cacheJField(JNIEnv *env) {
    jfieldCache = (JfieldCache) {};
    jclass pduContextClass = jClassCache.pduContextClass;
    jfieldCache.correlationIdFieldId = (*env)->GetFieldID(env, pduContextClass, "channelId", "Ljava/lang/String;");
    jfieldCache.byteBufferFieldId = (*env)->GetFieldID(env, pduContextClass, "pduBuffer", "[B");
}

void freeDecodedContext(DecodedContext *decodedContext) {
    uint32_t commandId = decodedContext->commandId;
    if (commandId == 4) {
        BaseSmReq *baseSmReq = (BaseSmReq *) decodedContext->pduStruct;
        free(baseSmReq->shortMessage);
        free(baseSmReq->scheduleDeliveryTime);
        free(baseSmReq->validityPeriod);
        free(baseSmReq->serviceType);

        Address *srcAddress = baseSmReq->sourceAddress;
        free(srcAddress->addressValue);
        free(srcAddress);

        Address *destAddress = baseSmReq->destinationAddress;
        free(destAddress->addressValue);
        free(destAddress);
        free(baseSmReq->header);
        free(baseSmReq);
    }
}
