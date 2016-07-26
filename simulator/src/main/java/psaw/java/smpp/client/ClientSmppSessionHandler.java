package psaw.java.smpp.client;

import com.cloudhopper.smpp.PduAsyncResponse;
import com.cloudhopper.smpp.impl.DefaultSmppSessionHandler;
import com.cloudhopper.smpp.pdu.PduRequest;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import psaw.java.smpp.service.SmppResponseNotifier;
import psaw.java.smpp.util.TpsLogger;

/**
 * <p>
 * <code>ClientSmppSessionHandler</code> -
 * Session handler for the SMPP client.
 * </p>
 */
public class ClientSmppSessionHandler extends DefaultSmppSessionHandler {

    private static final Logger logger = LogManager.getLogger(ClientSmppSessionHandler.class);

    private SmppResponseNotifier smppResponseNotifier;

    public ClientSmppSessionHandler() {
        super();
    }

    @Override
    public void fireChannelUnexpectedlyClosed() {
        super.fireChannelUnexpectedlyClosed();
    }

    @Override
    public void firePduRequestExpired(PduRequest pduRequest) {
        logger.debug("SMPP request expired [{}]", pduRequest);
//        smppResponseNotifier.notifyFailedResponse(String.valueOf(pduRequest.getSequenceNumber()));
    }

    @Override
    public void fireExpectedPduResponseReceived(PduAsyncResponse pduAsyncResponse) {
        logger.debug("Pdu Response Received : [{}]", pduAsyncResponse.getResponse());
        TpsLogger.getInstance().logResponseReceived(pduAsyncResponse.getResponse());
//        smppResponseNotifier.notifyResponse(pduAsyncResponse.getResponse());
    }

    public void setSmppResponseNotifier(SmppResponseNotifier smppResponseNotifier) {
        this.smppResponseNotifier = smppResponseNotifier;
    }
}
