package com.cloudhopper.smpp.transcoder.asynchronous;

import org.jboss.netty.buffer.ChannelBuffer;

/**
 * <p>
 * <code>AsynchronousContext</code> -
 * Context for the asynchronous ChannelBuffer for the PDU.
 * </p>
 *
 * @author prabath.
 */
public class AsynchronousContext {

    private String channelId;

    private ChannelBuffer channelBuffer;

    public byte[] pduBuffer;

    public boolean test(){
        return true;
    }

    public AsynchronousContext(String channelId, ChannelBuffer channelBuffer) {
        this.channelId = channelId;
        this.channelBuffer = channelBuffer;
        this.pduBuffer = channelBuffer.array();
    }

    public String getChannelId() {
        return channelId;
    }

    public ChannelBuffer getChannelBuffer() {
        return channelBuffer;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("AsynchronousContext{");
        sb.append("channelId='").append(channelId).append('\'');
        sb.append(", channelBuffer=").append(channelBuffer);
        sb.append('}');
        return sb.toString();
    }
}
