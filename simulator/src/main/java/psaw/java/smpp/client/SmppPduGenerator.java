package psaw.java.smpp.client;

import com.cloudhopper.smpp.SmppSession;
import com.cloudhopper.smpp.SmppSessionConfiguration;
import com.cloudhopper.smpp.SmppSessionHandler;
import com.cloudhopper.smpp.impl.DefaultSmppClient;
import com.cloudhopper.smpp.pdu.SubmitSm;
import com.cloudhopper.smpp.type.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import psaw.java.smpp.service.SmppClientConfiguration;
import psaw.java.smpp.service.SmppRequestGenerator;
import psaw.java.smpp.util.SequenceNumberGenerator;
import psaw.java.smpp.util.TpsLogger;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;

import static com.cloudhopper.smpp.SmppBindType.TRANSCEIVER;

/**
 * <p>
 * <code>SmppPduGenerator</code> -
 * Basic SMPP Client implementation.
 * </p>
 *
 * @author prabath.
 */
public class SmppPduGenerator {

    private static final Logger logger = LogManager.getLogger(SmppPduGenerator.class);

    private SmppSession session;

    private DefaultSmppClient smppClient;

    private SmppSessionConfiguration sessionConfiguration;

    private SmppSessionHandler sessionHandler;

    private SequenceNumberGenerator sequenceNumberGenerator;

    private SmppClientConfiguration smppClientConfiguration;

    private SmppRequestGenerator smppRequestGenerator;

    private long unbindTimeout = 2000;

    private volatile boolean isConnected;

    public SmppPduGenerator(SequenceNumberGenerator sequenceNumberGenerator) {
        this.sequenceNumberGenerator = sequenceNumberGenerator;
    }

    public void init(SmppClientConfiguration smppClientConfiguration) throws Exception {
        ScheduledThreadPoolExecutor scheduleThread =
                (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(1, r -> {
                    Thread thread = new Thread(r);
                    thread.setName("SMPP-Client-Thread");
                    return thread;
                });
        this.smppClientConfiguration = smppClientConfiguration;
        this.sessionConfiguration = createSessionConfiguration(smppClientConfiguration);
        this.sessionHandler = new ClientSmppSessionHandler();
        smppClient = new DefaultSmppClient(Executors.newCachedThreadPool(), 1, scheduleThread);
        bind();
    }

    private SmppSessionConfiguration createSessionConfiguration(SmppClientConfiguration smppClientConfiguration) {
        SmppSessionConfiguration sessionConfiguration =
                new SmppSessionConfiguration(TRANSCEIVER,
                        smppClientConfiguration.getSystemId(),
                        smppClientConfiguration.getPassword());
        sessionConfiguration.setBindTimeout(2000);
        sessionConfiguration.setRequestExpiryTimeout(2000);
        sessionConfiguration.setHost(smppClientConfiguration.getHost());
        sessionConfiguration.setPort(smppClientConfiguration.getPort());
        sessionConfiguration.setConnectTimeout(2000);
        sessionConfiguration.setWindowSize(20000);
        sessionConfiguration.setSystemType("Test-Client");
        sessionConfiguration.setWindowWaitTimeout(2000);
        sessionConfiguration.setWindowMonitorInterval(1000);
        return sessionConfiguration;
    }

    public void bind() {
        try {
            if (session != null && (session.isOpen() || session.isBound())) {
                unbindSession();
            }
            session = smppClient.bind(sessionConfiguration, sessionHandler);
            isConnected = true;
        } catch (SmppTimeoutException
                | SmppChannelException
                | InterruptedException
                | UnrecoverablePduException e) {
            logger.error("Error occurred while binding. SessionConfiguration : [{}]", sessionConfiguration, e);
        }
    }

    public void startSender() {
        if (isConnected) {
            smppRequestGenerator = new SmppRequestGenerator(
                    smppClientConfiguration.getTpsString(), this, sequenceNumberGenerator, smppClientConfiguration.getText());
            smppRequestGenerator.init();
            smppRequestGenerator.start();
        } else {
            logger.info("Client [{}] is not connected yet", smppClientConfiguration);
        }
    }

    public void unbindSession() {
        logger.info("Unbinding the session : [{}]", session);
        session.unbind(unbindTimeout);
    }

    public void stop() {
        logger.info("Stopping the Smpp Client. [{}]", this);
        if (session.isBound()) {
            unbindSession();
            return;
        }
        if (session.isOpen()) {
            session.close();
        }
    }

    public void sendSubmitSm(SubmitSm submitSm, long timeout) {
        try {
            submitSm.setSourceAddress(new Address((byte) 1, (byte) 2, smppClientConfiguration.getClientAddress()));
            session.sendRequestPdu(submitSm, timeout, false);
            TpsLogger.getInstance().logRequestSent(submitSm);
        } catch (RecoverablePduException
                | UnrecoverablePduException
                | SmppTimeoutException
                | SmppChannelException
                | InterruptedException e) {
            logger.error("Error occurred while sending the message", e);
        }
    }

    public boolean isConnected() {
        return isConnected;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("SmppPduGenerator{");
        sb.append("sessionConfiguration=").append(sessionConfiguration);
        sb.append('}');
        return sb.toString();
    }
}
