package psaw.java.smpp.codec.server;

import java.io.IOException;

/**
 * <p>
 * <code>{@link PerformanceSimulatorServer}</code> -
 * Interface for SmppServer implementation for the performance simulation.
 * </p>
 *
 * @author psaw.
 */
public interface PerformanceSimulatorServer {

    void init(SmppSessionHandlerFactory sessionHandlerFactory) throws IOException;

    boolean start();

    void stop();
}
