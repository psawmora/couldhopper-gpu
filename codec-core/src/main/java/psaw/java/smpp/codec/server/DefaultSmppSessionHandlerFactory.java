package psaw.java.smpp.codec.server;

import com.cloudhopper.smpp.SmppSession;
import com.cloudhopper.smpp.impl.DefaultSmppSessionHandler;
import com.cloudhopper.smpp.pdu.DeliverSm;
import com.cloudhopper.smpp.pdu.PduRequest;
import com.cloudhopper.smpp.pdu.PduResponse;
import com.cloudhopper.smpp.pdu.SubmitSm;
import psaw.java.smpp.codec.util.SmppLoggerUtil;

import java.lang.ref.WeakReference;

/**
 * <p>
 * <code>{@link DefaultSmppSessionHandlerFactory}</code> -
 * Default implementation.
 * </p>
 *
 * @author psaw.
 */
public class DefaultSmppSessionHandlerFactory implements SmppSessionHandlerFactory {

    @Override
    public DefaultSmppSessionHandler createSmppSessionHandler(SmppSession smppSession) {
        return new SimulatedSmppSessionHandler(smppSession);
    }

    public static class SimulatedSmppSessionHandler extends DefaultSmppSessionHandler {

        private WeakReference<SmppSession> sessionRef;

        public SimulatedSmppSessionHandler(SmppSession session) {
            this.sessionRef = new WeakReference<SmppSession>(session);
        }

        @Override
        public PduResponse firePduRequestReceived(PduRequest pduRequest) {
            if (pduRequest instanceof SubmitSm) {
                SmppLoggerUtil.logSubmitSm((SubmitSm) pduRequest);
            } else if (pduRequest instanceof DeliverSm) {
                SmppLoggerUtil.logDeliverSm((DeliverSm) pduRequest);
            }
            return null;
        }
    }

}
