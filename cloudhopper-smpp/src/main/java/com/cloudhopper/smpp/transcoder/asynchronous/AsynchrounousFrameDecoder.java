package com.cloudhopper.smpp.transcoder.asynchronous;

import com.cloudhopper.commons.util.HexUtil;
import com.cloudhopper.smpp.SmppConstants;
import com.cloudhopper.smpp.transcoder.PduTranscoder;
import com.cloudhopper.smpp.type.UnrecoverablePduException;
import org.jboss.netty.buffer.ChannelBuffer;
import org.jboss.netty.channel.Channel;
import org.jboss.netty.channel.ChannelHandlerContext;
import org.jboss.netty.handler.codec.frame.FrameDecoder;

import static com.cloudhopper.smpp.SmppConstants.PDU_INT_LENGTH;

/**
 * <p>
 * <code>AsynchrounousFrameDecoder</code> -
 * Only checks for the completeness of the packet.
 * </p>
 *
 * @author prabath.
 */
public class AsynchrounousFrameDecoder extends FrameDecoder {

    private PduTranscoder transcoder;

    public AsynchrounousFrameDecoder(PduTranscoder transcoder) {
        super(false);
        this.transcoder = transcoder;
    }

    @Override
    protected Object decode(ChannelHandlerContext channelHandlerContext, Channel channel, ChannelBuffer buffer)
            throws Exception {
        //        System.out.println("Frame Decoder");
        // wait until the length prefix is available
        return decodeTest(buffer);
    }

    private Object decodeTest(ChannelBuffer buffer) throws UnrecoverablePduException {
        if (buffer.readableBytes() < PDU_INT_LENGTH) {
            return null;
        }

        // parse the command length (first 4 bytes)
        int commandLength = buffer.getInt(buffer.readerIndex());
        //logger.trace("PDU commandLength [" + commandLength + "]");

        // valid command length is >= 16 bytes
        if (commandLength < SmppConstants.PDU_HEADER_LENGTH) {
            throw new UnrecoverablePduException("Invalid PDU length [0x" + HexUtil.toHexString(commandLength) + "] parsed");
        }

        // wait until the whole pdu is available (entire pdu)
        if (buffer.readableBytes() < commandLength) {
            return null;
        }

        // at this point, we have the entire PDU and length already in the buffer
        // we'll create a new "view" of this PDU and read the data from the actual buffer
        // NOTE: this should be super fast since the underlying byte array doesn't get copied
        ChannelBuffer pduContent = buffer.copy(buffer.readerIndex(), commandLength);
        buffer.skipBytes(commandLength);
        return pduContent;
    }

}
