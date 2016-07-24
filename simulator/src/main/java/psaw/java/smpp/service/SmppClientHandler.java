package psaw.java.smpp.service;

import psaw.java.smpp.client.SmppPduGenerator;

import java.util.List;

/**
 * <p>
 * <code>SmppClientHandler</code> -
 * Handles <code>SmppPduGenerator</code> related house keeping.
 * - Session bind initialization.
 * - Connection retries.
 * - Connection teardown.
 * - Cleaning up Smpp Clients at system shutdowns.
 * </p>
 *
 * @author prabath.
 */
public interface SmppClientHandler {

    /**
     * <p>
     * Handles every aspect of <code>SmppPduGenerator</code> mentioned above.
     * </p>
     *
     * @param smppPduGenerators
     */
    void handleSmppClients(List<SmppPduGenerator> smppPduGenerators);

}
