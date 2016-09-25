package psaw.java.smpp.codec.server;

import com.cloudhopper.smpp.SmppSession;
import com.cloudhopper.smpp.impl.DefaultSmppSessionHandler;

/**
 * <p>
 * <code>{@link SmppSessionHandlerFactory}</code> -
 * Factory for Smpp server session handlers.
 * </p>
 *
 * @author psaw.
 */
public interface SmppSessionHandlerFactory {

    DefaultSmppSessionHandler createSmppSessionHandler(SmppSession smppSession);
}
