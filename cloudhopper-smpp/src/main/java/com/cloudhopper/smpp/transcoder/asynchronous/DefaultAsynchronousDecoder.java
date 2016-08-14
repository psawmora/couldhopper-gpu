package com.cloudhopper.smpp.transcoder.asynchronous;

import com.cloudhopper.smpp.impl.SmppSessionChannelListener;
import com.cloudhopper.smpp.pdu.Pdu;
import com.cloudhopper.smpp.transcoder.PduTranscoder;
import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.buffer.ChannelBuffers;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import static java.nio.ByteOrder.BIG_ENDIAN;

/**
 * <p>
 * <code>DefaultAsynchronousDecoder</code> -
 * The default implementation.
 * </p>
 *
 * @author prabath.
 */
public class DefaultAsynchronousDecoder implements AsynchronousDecoder {

    private static final Logger logger = LoggerFactory.getLogger(DefaultAsynchronousDecoder.class);

    private Map<String, SmppSessionChannelListener> sessionChannelListeners;

    private PduTranscoder pduTranscoder;

    private ExecutorService executorService;

    private List<AsynchronousContext> bufferContexts;

    private ReentrantReadWriteLock.ReadLock readLock;

    private ReentrantReadWriteLock.WriteLock writeLock;

    private ChannelBuffer pduContainerBuffer;

    private ByteBuffer pduContainerBufferActive;

    private int batchSize = 600;

    private AtomicInteger currentBatchSize;

    private ChannelBuffer pdu;

    private native List<DecodedPduContext> decodePDU(List<AsynchronousContext> bufferContexts);

    private native List<DecodedPduContext> decodePDUDirect(ByteBuffer containerBuffer, int batchSize, int correlationIdLength);

    private native void initialize();

    static {
        System.loadLibrary("SMPPDecoder");
    }

    public DefaultAsynchronousDecoder(PduTranscoder pduTranscoder, ExecutorService executorService) {
        this.pduTranscoder = pduTranscoder;
        this.sessionChannelListeners = new ConcurrentHashMap<>();
        this.executorService = executorService;
        this.bufferContexts = new ArrayList<>();
        ReentrantReadWriteLock readWriteLock = new ReentrantReadWriteLock();
        this.readLock = readWriteLock.readLock();
        this.writeLock = readWriteLock.writeLock();
        this.currentBatchSize = new AtomicInteger(0);
        this.pduContainerBuffer = ChannelBuffers.directBuffer(BIG_ENDIAN, 1024 * 1024 * 1);
        //        this.pduContainerBufferSecondary = ChannelBuffers.directBuffer(BIG_ENDIAN, 1024 * 1024 * 1000);
        //        this.pduContainerBufferActive = pduContainerBuffer;
        this.pduContainerBufferActive = ByteBuffer.allocateDirect(75 * 601);
        //        this.pduContainerBufferActive = ByteBuffer.allocateDirect(1024 * 1024 * 50);
        initialize();
    }

    @Override
    public void submitToDecode(final AsynchronousContext asynchronousContext) {
        final String channelId = asynchronousContext.getChannelId();
        final SmppSessionChannelListener listener = sessionChannelListeners.get(channelId);
        if (listener == null) {
            logger.debug("SessionChannelListener for the channelId [{}] is null. Not proceeding further.",
                    channelId);
            return;
        }
        sendAsynch(asynchronousContext, listener);
    }

    private Future<?> sendAsynch(final AsynchronousContext asynchronousContext, final SmppSessionChannelListener listener) {
        return executorService.submit(new Runnable() {
            @Override
            public void run() {
                try {
                    ChannelBuffer buffer = asynchronousContext.getChannelBuffer();
                    buffer.markReaderIndex();
                    int type = buffer.readInt();
                    buffer.resetReaderIndex();
                    if (type != 4) {
                        ChannelBuffer channelBuffer = asynchronousContext.getChannelBuffer();
                        try {
                            Pdu pdu = pduTranscoder.decode(channelBuffer);
                            listener.firePduReceived(pdu);
                        } catch (Exception e) {
                            logger.error("Recoverable exception", e);
                        }
                        return;
                    }
                    if (currentBatchSize.compareAndSet(batchSize, 0)) {
                        logger.debug("Sending the batch to the decoder");
                        List<DecodedPduContext> decodedPduContexts =
                                decodePDUDirect(pduContainerBufferActive, batchSize, 14);
                        pduContainerBufferActive.position(0);
                        notifyListeners(decodedPduContexts);
                    } else {
                        pdu = asynchronousContext.getChannelBuffer().copy();
                        String channelId = asynchronousContext.getChannelId();
                        pduContainerBufferActive.put(channelId.getBytes());
                        pduContainerBufferActive.put(pdu.toByteBuffer(pdu.readerIndex(), pdu.readableBytes()));
                        currentBatchSize.incrementAndGet();
                    }
                } catch (Exception e) {
                    logger.error("Error occurred while decoding the ChannelBuffer", e);
                    listener.fireExceptionThrown(e);
                }
            }
        });
    }

    private void notifyListeners(List<DecodedPduContext> decodedPduContexts) {
        for (DecodedPduContext decodedPduContext : decodedPduContexts) {
            SmppSessionChannelListener listener =
                    sessionChannelListeners.get(decodedPduContext.getCorrelationId());
            if (listener != null && decodedPduContext.getPdu() != null) {
                listener.firePduReceived(decodedPduContext.getPdu());
            }
        }
    }

    @Override
    public void registerSessionListener(String channelId, SmppSessionChannelListener sessionChannelListener) {
        logger.info("Registering SmppSessionChannelListener for the channel id [{}]", channelId);
        sessionChannelListeners.put(channelId, sessionChannelListener);
    }

    @Override
    public void deregister(String channelId) {
        logger.info("DeRegistering SessionChannelListener for the channel id [{}]", channelId);
        sessionChannelListeners.remove(channelId);
    }

    public void setPduTranscoder(PduTranscoder pduTranscoder) {
        this.pduTranscoder = pduTranscoder;
    }
}
