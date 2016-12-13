package com.cloudhopper.smpp.transcoder.asynchronous;

import com.cloudhopper.smpp.pdu.Pdu;

/**
 * <p>
 * <code>DecodedPduContext</code> -
 * Contains details for the decoded pdu context.
 * </p>
 *
 * @author prabath.
 */
public class DecodedPduContext {

    private String correlationId;

    private Pdu pdu;

    public DecodedPduContext(String correlationId, Pdu pdu) {
        this.correlationId = correlationId;
        this.pdu = pdu;
    }

    public String getCorrelationId() {
        return correlationId;
    }

    public void setCorrelationId(String correlationId) {
        this.correlationId = correlationId;
    }

    public Pdu getPdu() {
        return pdu;
    }

    public void setPdu(Pdu pdu) {
        this.pdu = pdu;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("DecodedPduContext{");
        sb.append("correlationId='").append(correlationId).append('\'');
        sb.append(", pdu=").append(pdu);
        sb.append('}');
        return sb.toString();
    }
}
