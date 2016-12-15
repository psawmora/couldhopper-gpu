package com.cloudhopper.smpp.transcoder.asynchronous;

import com.cloudhopper.smpp.impl.SmppSessionChannelListener;
import com.cloudhopper.smpp.pdu.Pdu;
import com.cloudhopper.smpp.pdu.SubmitSm;
import com.cloudhopper.smpp.transcoder.PduTranscoder;
import org.jboss.netty.buffer.ChannelBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

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

    public static int MAX_PDU_LENGTH = 200;

    private Map<String, SmppSessionChannelListener> sessionChannelListeners;

    private PduTranscoder pduTranscoder;

    private ExecutorService executorService;

    private List<AsynchronousContext> bufferContexts;

    private ReentrantReadWriteLock.ReadLock readLock;

    private ReentrantReadWriteLock.WriteLock writeLock;

    private ByteBuffer pduContainerBufferActive;

    private int batchSize = 200000;

    private AtomicInteger currentBatchSize;

    private ChannelBuffer pdu;

    private volatile boolean isTuningMode = false;

    private volatile boolean isAccept = true;

    private String configurationFilePath;

    private native List<DecodedPduContext> decodePDU(List<AsynchronousContext> bufferContexts);

    private native List<DecodedPduContext> decodePDUDirect(ByteBuffer containerBuffer, int batchSize, int correlationIdLength);

    private native void startTuner(ByteBuffer containerBuffer, int batchSize, int correlationIdLength);

    private native void initialize(String configurationFilePath);

    static {
        System.loadLibrary("SMPPDecoder");
    }

    public DefaultAsynchronousDecoder(PduTranscoder pduTranscoder, ExecutorService executorService, int batchSize) {
        this.pduTranscoder = pduTranscoder;
        this.sessionChannelListeners = new ConcurrentHashMap<>();
        this.executorService = executorService;
        this.bufferContexts = new ArrayList<>();
        ReentrantReadWriteLock readWriteLock = new ReentrantReadWriteLock();
        this.readLock = readWriteLock.readLock();
        this.writeLock = readWriteLock.writeLock();
        this.currentBatchSize = new AtomicInteger(0);
        this.batchSize = batchSize;
        this.pduContainerBufferActive = ByteBuffer.allocateDirect(MAX_PDU_LENGTH * batchSize);
        try {
            Properties properties = new Properties();
            properties.load(this.getClass().getResourceAsStream("/system.properties"));
            configurationFilePath = properties.getProperty("codec.configuration.file.path",
                    "/home/prabath/Projects/Msc/couldhopper-gpu/codec-core/src/main/resources/codec-mode.cfg");
            isTuningMode = Boolean.parseBoolean(properties.getProperty("tuning.mode.on", "false"));
        } catch (IOException e) {
            logger.error("Error occurred while loading properties", e);
            System.exit(-1);
        }
        initialize(configurationFilePath);
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
        if (!isAccept) {
            logger.debug("Not accepting requests as the performance tuning is on the way");
            return null;
        }
        return executorService.submit(() -> {
            try {
                ChannelBuffer buffer = asynchronousContext.getChannelBuffer();
                buffer.markReaderIndex();
                buffer.skipBytes(4);
                int type = buffer.readInt();
                buffer.resetReaderIndex();
                if (type != 4) {
                    ChannelBuffer channelBuffer = asynchronousContext.getChannelBuffer();
                    try {
                        Pdu pdu1 = pduTranscoder.decode(channelBuffer);
                        listener.firePduReceived(pdu1);
                    } catch (Exception e) {
                        logger.error("Recoverable exception", e);
                    }
                    return;
                }
                if (currentBatchSize.compareAndSet(batchSize, 0)) {
                    if (isTuningMode) {
                        isAccept = false;
                        logger.debug("Starting performance tuning and sending batch");
                        System.out.println("Starting performance tuning and sending batch");
                        startTuner(pduContainerBufferActive, batchSize, 15);
                        pduContainerBufferActive.position(0);
                        logger.debug("Performance tuning ended.");
                    } else {
                        logger.debug("Sending the batch to the decoder");
                        System.out.println("Sending the batch to the decoder");
                        List<DecodedPduContext> decodedPduContexts = decodePDUDirect(pduContainerBufferActive, batchSize, 15);
                        System.out.println("Got Response " + decodedPduContexts.size());
                        pduContainerBufferActive.position(0);
                        notifyListeners(decodedPduContexts);
                    }
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
        });
    }

    private void notifyListeners(List<DecodedPduContext> decodedPduContexts) {
        for (DecodedPduContext decodedPduContext : decodedPduContexts) {
            SmppSessionChannelListener listener =
                    sessionChannelListeners.get(decodedPduContext.getCorrelationId());
            if (listener != null && decodedPduContext.getPdu() != null) {
                System.out.println("MESSAGE - " + new String(((SubmitSm) decodedPduContext.getPdu()).getShortMessage()));
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

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
