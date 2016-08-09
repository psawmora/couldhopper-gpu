package com.cloudhopper.smpp.transcoder.asynchronous;

import com.cloudhopper.smpp.impl.SmppSessionChannelListener;
import com.cloudhopper.smpp.pdu.Pdu;
import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * <code>AsynchronousSmppSessionWrapper</code> -
 * Acts as the buffer point for delegating PDUs for encoding/decoding.
 * </p>
 *
 * @author prabath.
 */
public class AsynchronousSmppSessionWrapper extends SimpleChannelUpstreamHandler {

    private static final Logger logger = LoggerFactory.getLogger(AsynchronousSmppSessionWrapper.class);

    private SmppSessionChannelListener listener;

    private AsynchronousDecoder asynchronousDecoder;

    private String channelId;

    public AsynchronousSmppSessionWrapper(SmppSessionChannelListener listener,
                                          AsynchronousDecoder asynchronousDecoder,
                                          String channelId) {
        this.listener = listener;
        this.asynchronousDecoder = asynchronousDecoder;
        this.channelId = channelId == null ? String.valueOf(System.nanoTime()) : channelId;
        this.asynchronousDecoder.registerSessionListener(this.channelId, listener);
    }

    @Override
    public void messageReceived(ChannelHandlerContext ctx, MessageEvent e) throws Exception {
        if (e.getMessage() instanceof Pdu) {
            Pdu pdu = (Pdu) e.getMessage();
            this.listener.firePduReceived(pdu);
        } else if (e.getMessage() instanceof ChannelBuffer) {
            AsynchronousContext asynchronousContext = new AsynchronousContext(this.channelId, (ChannelBuffer) e.getMessage());
            this.asynchronousDecoder.submitToDecode(asynchronousContext);
        }
    }

    /**
     * Invoked when an exception was raised by an I/O thread or an upstream handler.
     */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, ExceptionEvent e) throws Exception {
        //logger.warn("Exception triggered in upstream ChannelHandler: {}", e.getCause());
        this.listener.fireExceptionThrown(e.getCause());
    }

    /**
     * Invoked when a Channel was disconnected from its remote peer.
     */
    @Override
    public void channelDisconnected(ChannelHandlerContext ctx, ChannelStateEvent e) throws Exception {
        //logger.info(e.toString());
        //ctx.sendUpstream(e);
        this.asynchronousDecoder.deregister(channelId);
    }

    /**
     * Invoked when a Channel was unbound from the current local address.
     */
    @Override
    public void channelUnbound(ChannelHandlerContext ctx, ChannelStateEvent e) throws Exception {
        //logger.info(e.toString());
        this.asynchronousDecoder.deregister(channelId);
    }

    /**
     * Invoked when a Channel was closed and all its related resources were released.
     */
    @Override
    public void channelClosed(ChannelHandlerContext ctx, ChannelStateEvent e) throws Exception {
        //logger.info(e.toString());
        this.listener.fireChannelClosed();
        this.asynchronousDecoder.deregister(channelId);
    }


}
