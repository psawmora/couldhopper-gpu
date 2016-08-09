package com.cloudhopper.smpp.transcoder.asynchronous;

import com.cloudhopper.smpp.impl.SmppSessionChannelListener;

/**
 * <p>
 * <code>AsynchronousDecoder</code> -
 * Decoder interface for decoding PDUs asynchronously. The decoded PDUs would be notified to the relevant listener.
 * </p>
 *
 * @author prabath.
 */
public interface AsynchronousDecoder {

    void registerSessionListener(String channelId, SmppSessionChannelListener sessionChannelListener);

    void submitToDecode(AsynchronousContext asynchronousContext);

    void deregister(String channelId);
}
