package psaw.java.smpp.service;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import psaw.java.smpp.client.SmppPduGenerator;
import psaw.java.smpp.util.SequenceNumberGenerator;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * <p>
 * <code>SmppSimulatorManager</code> -
 * Encapsulate the logic for generating PDUs.
 * </p>
 *
 * @author prabath.
 */
public class SmppSimulatorManager {

    private static final Logger logger = LogManager.getLogger(SmppSimulatorManager.class);

    private String smppClientConfigFile = "/smpp-client.properties";

    private List<SmppPduGenerator> smppPduGenerators;

    private SequenceNumberGenerator sequenceNumberGenerator;

    public void startSimulation() throws Exception {
        logger.info("Starting the simulator.");
        smppPduGenerators = new ArrayList<>();
        sequenceNumberGenerator = new SequenceNumberGenerator(1000000, 1);
        initializePduGenerators();
    }

    public void stopSimulator() {
        logger.info("Stopping the simulator.");
        for (SmppPduGenerator smppPduGenerator : smppPduGenerators) {
            try {
                smppPduGenerator.stop();
            } catch (Exception e) {
                logger.error("Error occurred while stopping the SmppPduGenerator", e);
            }
        }
    }

    private void initializePduGenerators() throws Exception {
        InputStream inputStream = SmppSimulatorManager.class.getResourceAsStream(smppClientConfigFile);
        Properties properties = new Properties();
        try {
            properties.load(inputStream);
        } catch (IOException e) {
            logger.error("Error in loading SMPP client properties.", e);
            throw e;
        }
        int noOfClients = Integer.parseInt(properties.getProperty("no.of.clients"));
        logger.info("Number of configured clients : [{}]", noOfClients);
        List<SmppClientConfiguration> clientConfigurations = new ArrayList<>();
        for (int i = 0; i < noOfClients; i++) {
            SmppClientConfiguration clientConfiguration = createSmppClientConfiguration(i, properties);
            logger.debug("Client Configuration : [{}] for index [{}]", clientConfiguration, i);
            clientConfigurations.add(clientConfiguration);
        }
        startPduGenerators(clientConfigurations);
    }

    private void startPduGenerators(List<SmppClientConfiguration> clientConfigurations) throws Exception {
        for (SmppClientConfiguration smppClientConfiguration : clientConfigurations) {
            SmppPduGenerator smppPduGenerator = new SmppPduGenerator(sequenceNumberGenerator);
            smppPduGenerators.add(smppPduGenerator);
            smppPduGenerator.init(smppClientConfiguration);
            smppPduGenerator.startSender();
        }
    }

    private SmppClientConfiguration createSmppClientConfiguration(int index, Properties properties) {
        return SmppClientConfiguration.newBuilder()
                .withClientAddress(properties.getProperty("client.address." + index))
                .withDestincationNumberRange(properties.getProperty("dest.number.range." + index))
                .withHost(properties.getProperty("host." + index))
                .withPort(Integer.parseInt(properties.getProperty("port." + index)))
                .withPassword(properties.getProperty("password." + index))
                .withSystemId(properties.getProperty("system.id." + index))
                .withText(properties.getProperty("text.msg." + index))
                .withTpsString(properties.getProperty("send.tps." + index)).build();
    }

    public void setSmppClientConfigFile(String smppClientConfigFile) {
        this.smppClientConfigFile = smppClientConfigFile;
    }
}
