#include "smpp_pdu_struct.h"

jobject *createPdu(JNIEnv *env, DecodedContext *decodedContext);

jobject *getAddress(JNIEnv *env, Address *address);

void cacheJClass(JNIEnv *env);

void cacheJMethod(JNIEnv *env);

void cacheJField(JNIEnv *env);

void freeDecodedContext(DecodedContext *decodedContext);

jobject startDecodingBatch(JNIEnv *env, jobject thisObj, jobject param);

static JclassCache jClassCache;

static JmethodCache jmethodCache;

static JfieldCache jfieldCache;

JNIEXPORT void JNICALL Java_com_cloudhopper_smpp_transcoder_asynchronous_DefaultAsynchronousDecoder_initialize
        (JNIEnv *env, jobject thisObj) {
    printf("Caching JClass , JMethodId and JFieldId values.\n");
    cacheJClass(env);
    cacheJMethod(env);
    cacheJField(env);
}

JNIEXPORT jobject
JNICALL Java_com_cloudhopper_smpp_transcoder_asynchronous_DefaultAsynchronousDecoder_decodePDU
        (JNIEnv *env, jobject thisObj, jobject param) {
    return startDecodingBatch(env, thisObj, param);
}

jobject startDecodingBatch(JNIEnv *env, jobject thisObj, jobject param) {
    printf("Started decoding the batch\n");
    test();
    jclass arrayListClass = jClassCache.arrayListClass;
    jmethodID listSizeMethod = jmethodCache.listSizeMethod;
    jint pduContextCount = (*env)->CallIntMethod(env, param, listSizeMethod);
    printf("PduContext Count -> %d\n", pduContextCount);

    jclass decodedContextContainerClass = jClassCache.arrayListClass;
    jmethodID decodedContextContainerMethod = jmethodCache.decodedContextContainerMethod;
    jobject decodedContextContainer = (*env)->NewObject(env, decodedContextContainerClass,
                                                        decodedContextContainerMethod);
    jmethodID addDecodedContextMethodId = jmethodCache.addDecodedContextMethodId;

    PduContext *pduContexts = malloc(sizeof(PduContext) * pduContextCount);

    int i;
    for (i = 0; i < pduContextCount; i++) {
        jmethodID listGetMethod = jmethodCache.listGetMethod;
        jobject pduContext = (*env)->CallObjectMethod(env, param, listGetMethod, i);
        jclass pduContextClass = jClassCache.pduContextClass;
        jfieldID correlationIdFieldId = jfieldCache.correlationIdFieldId;
        jstring correlationIdString = (*env)->GetObjectField(env, pduContext, correlationIdFieldId);
        const char *correlationId = (*env)->GetStringUTFChars(env, correlationIdString, NULL);
//        printf("CorrelationId - %s\n", correlationId);

        jfieldID byteBufferFieldId = jfieldCache.byteBufferFieldId;
        jbyteArray *byteArrayNative = (*env)->GetObjectField(env, pduContext, byteBufferFieldId);
        jbyte *byteArrayBuffer = (*env)->GetByteArrayElements(env, byteArrayNative, NULL);
        jsize byteBufferLength = (*env)->GetArrayLength(env, byteArrayNative);
//        printf("byteBuffer Size - %d\n", byteBufferLength);

        PduContext context = {correlationId, byteArrayBuffer, byteBufferLength, byteArrayNative, correlationIdString};
        pduContexts[i] = context;
    }


    int nThread = N_THREAD;
    int index = 0;
    int batchSize = pduContextCount > nThread ? pduContextCount / nThread : pduContextCount;
    DecodedContext *decodedPduStructList = malloc(sizeof(DecodedContext) * pduContextCount);
    ThreadParam *threadParams[nThread];

    pthread_t threads[nThread];
    int threadIndex = 0;
    while (index < pduContextCount) {
        int startIndex = index;
        int length = (pduContextCount - index) <= batchSize ? (pduContextCount - index) : batchSize;
        index += length;
        printf("start- %d | end- %d | next start - %d \n", startIndex, startIndex + length, index);
//       ThreadParam threadParam = threadParams[threadIndex];
        ThreadParam *threadParam = malloc(sizeof(ThreadParam));
        threadParam->startIndex = startIndex;
        threadParam->length = length;
        threadParam->pduContexts = pduContexts;
        threadParam->decodedPduStructList = decodedPduStructList;
        threadParams[threadIndex] = threadParam;
        int state = pthread_create(&threads[threadIndex], NULL, decode, (void *) threadParam);
        threadIndex++;
    }

    for (i = 0; i < threadIndex; i++) {
        pthread_join(threads[i], NULL);
    }
//    free(threadParams);
    for (i = 0; i < pduContextCount; i++) {
        DecodedContext *decodedContext = &decodedPduStructList[i];
        PduContext context = pduContexts[i];
        if (decodedContext != 0) {
//        printf("Not Null decodedPduStructure\n");
//        printf("SMPP PDU type %d\n", decodedContext->commandId);
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
//        printf("Adding decoded PDU context.\n");
                if ((*env)->ExceptionOccurred(env)) {
                    printf("Exception occurred\n\n");
                    freeDecodedContext(decodedContext);
                    (*env)->ReleaseStringUTFChars(env, context.correlationIdString, decodedContext->correlationId);
                    (*env)->ReleaseByteArrayElements(env, context.byteArrayNative, context.pduBuffer, 0);
                    return;
                }
//        printf("Storing decoded PDU context.\n");
                fflush(stdout);
                (*env)->CallObjectMethod(env, decodedContextContainer, addDecodedContextMethodId,
                                         decodedPduContextObject);
            } else {
                printf("PDU context is null.\n");
            }
            freeDecodedContext(decodedContext);
            (*env)->ReleaseStringUTFChars(env, context.correlationIdString, decodedContext->correlationId);
            (*env)->ReleaseByteArrayElements(env, context.byteArrayNative, context.pduBuffer, 0);
        } else {
            printf("NULL decodedPduStructure\n");
        }
    }

    printf("Returning\n");
    fflush(stdout);
    return decodedContextContainer;
}

jobject *createPdu(JNIEnv *env, DecodedContext *decodedContext) {
    printf("SMPP PDU type %d\n", decodedContext->commandId);
    if (decodedContext->commandId == 4) {
        SubmitSmReq submitSmReq = *(SubmitSmReq *) decodedContext->pduStruct;
//        printf("SubmitSm Request - %d\n", submitSmReq.esmClass);
//        printf("SubmitSm setCommandLength - %d\n", submitSmReq.header->commandLength);
        jclass submitSmClass = jClassCache.submitSmClass;
//        printf("SubmitSm class found - %s\n", submitSmClass);

        jmethodID constructor = jmethodCache.constructor;
        jobject submitSmObject = (*env)->NewObject(env, submitSmClass, constructor);

//        printf("SubmitSm setCommandLength - %d\n", submitSmReq.header->commandLength);
        jmethodID setCommandLength = jmethodCache.setCommandLength;
        (*env)->CallObjectMethod(env, submitSmObject, setCommandLength, submitSmReq.header->commandLength);

//        printf("SubmitSm setCommandStatus - %d\n", submitSmReq.header->commandStatus);
        jmethodID setCommandStatus = jmethodCache.setCommandStatus;
        (*env)->CallObjectMethod(env, submitSmObject, setCommandStatus, submitSmReq.header->commandStatus);

//        printf("SubmitSm sequenceNumber - %d\n", submitSmReq.header->sequenceNumber);
        jmethodID setSequenceNumber = jmethodCache.setSequenceNumber;
        (*env)->CallObjectMethod(env, submitSmObject, setSequenceNumber, submitSmReq.header->sequenceNumber);

        jmethodID setServiceType = jmethodCache.setServiceType;
        (*env)->CallObjectMethod(env, submitSmObject, setServiceType,
                                 (*env)->NewStringUTF(env, submitSmReq.serviceType));

        jmethodID setSourceAddress = jmethodCache.setSourceAddress;
        jobject *srcAddress = getAddress(env, submitSmReq.sourceAddress);

        (*env)->CallObjectMethod(env, submitSmObject, setSourceAddress, *srcAddress);

//        printf("SubmitSm Destination Address - %s\n", submitSmReq.destinationAddress->addressValue);
        jmethodID setDestAddress = jmethodCache.setDestAddress;
        (*env)->CallObjectMethod(env, submitSmObject, setDestAddress, *getAddress(env, submitSmReq.destinationAddress));

        jmethodID setEsmClass = jmethodCache.setEsmClass;
        (*env)->CallObjectMethod(env, submitSmObject, setEsmClass, submitSmReq.esmClass);

        jmethodID setProtocolId = jmethodCache.setProtocolId;
        (*env)->CallObjectMethod(env, submitSmObject, setProtocolId, submitSmReq.protocolId);

        jmethodID setPriority = jmethodCache.setPriority;
        (*env)->CallObjectMethod(env, submitSmObject, setPriority, submitSmReq.priority);

        jmethodID setScheduleDeliveryTime = jmethodCache.setScheduleDeliveryTime;
        (*env)->CallObjectMethod(env, submitSmObject, setScheduleDeliveryTime,
                                 (*env)->NewStringUTF(env, submitSmReq.scheduleDeliveryTime));

        jmethodID setValidityPeriod = jmethodCache.setValidityPeriod;
        (*env)->CallObjectMethod(env, submitSmObject, setValidityPeriod,
                                 (*env)->NewStringUTF(env, submitSmReq.validityPeriod));

        jmethodID setRegisteredDelivery = jmethodCache.setRegisteredDelivery;
        (*env)->CallObjectMethod(env, submitSmObject, setRegisteredDelivery, submitSmReq.registeredDelivery);

        jmethodID setReplaceIfPresent = jmethodCache.setReplaceIfPresent;
        (*env)->CallObjectMethod(env, submitSmObject, setReplaceIfPresent, submitSmReq.replaceIfPresent);


        jmethodID setDataCoding = jmethodCache.setDataCoding;
        (*env)->CallObjectMethod(env, submitSmObject, setDataCoding, submitSmReq.dataCoding);


        jmethodID setDefaultMsgId = jmethodCache.setDefaultMsgId;
        (*env)->CallObjectMethod(env, submitSmObject, setDefaultMsgId, submitSmReq.defaultMsgId);

        if (submitSmReq.smLength > 0) {
            jmethodID setShortMessage = jmethodCache.setShortMessage;
            jbyteArray shortMsgArray = (*env)->NewByteArray(env, submitSmReq.smLength);
            (*env)->SetByteArrayRegion(env, shortMsgArray, 0, submitSmReq.smLength, submitSmReq.shortMessage);
            (*env)->CallObjectMethod(env, submitSmObject, setShortMessage, shortMsgArray);
        }
        return submitSmObject;
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
        SubmitSmReq *submitSmReq = (SubmitSmReq *) decodedContext->pduStruct;
        free(submitSmReq->shortMessage);
        free(submitSmReq->scheduleDeliveryTime);
        free(submitSmReq->validityPeriod);
        free(submitSmReq->serviceType);

        Address *srcAddress = submitSmReq->sourceAddress;
        free(srcAddress->addressValue);
        free(srcAddress);

        Address *destAddress = submitSmReq->destinationAddress;
        free(destAddress->addressValue);
        free(destAddress);
        free(submitSmReq->header);
        free(submitSmReq);
    }
}
