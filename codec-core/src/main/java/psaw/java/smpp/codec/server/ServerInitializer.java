package psaw.java.smpp.codec.server;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;

/**
 * <p>
 * <code>{@link ServerInitializer}</code> -
 * Starts and stops servers.
 * </p>
 *
 * @author psaw.
 */
public class ServerInitializer {

    private static final Logger logger = LogManager.getLogger(ServerInitializer.class);

    private PerformanceSimulatorServer server;

    private SmppSessionHandlerFactory sessionHandlerFactory;

    public void start() throws IOException {
        logger.info("Initializing Smpp Server for the performance test.");
        server = new AsynchServerMain();
        sessionHandlerFactory = new DefaultSmppSessionHandlerFactory();
        server.init(sessionHandlerFactory);
        server.start();
    }

    public void stop() {
        server.stop();
    }
}
