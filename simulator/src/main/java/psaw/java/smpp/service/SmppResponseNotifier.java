package psaw.java.smpp.service;

import com.cloudhopper.smpp.pdu.PduResponse;

/**
 * <p>
 * <code>SmppResponseNotifier</code> -
 * Notifies responses.
 * </p>
 *
 * @author prabath.
 */
public interface SmppResponseNotifier {

    void notifySuccessResponse(String msgId);

    void notifyFailedResponse(String sequenceNumber);

    void notifyResponse(PduResponse response);
}
