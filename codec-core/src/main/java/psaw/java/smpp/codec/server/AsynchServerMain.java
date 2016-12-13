package psaw.java.smpp.codec.server;

import com.cloudhopper.smpp.*;
import com.cloudhopper.smpp.impl.DefaultSmppServer;
import com.cloudhopper.smpp.pdu.BaseBind;
import com.cloudhopper.smpp.pdu.BaseBindResp;
import com.cloudhopper.smpp.type.SmppChannelException;
import com.cloudhopper.smpp.type.SmppProcessingException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.util.Properties;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.atomic.AtomicInteger;

import static java.lang.Integer.parseInt;

public class AsynchServerMain implements PerformanceSimulatorServer {

    private static final Logger logger = LogManager.getLogger(AsynchServerMain.class);

    private DefaultSmppServer smppServer;

    private SmppSessionHandlerFactory sessionHandlerFactory;

    private SmppServerConfiguration configuration;

    public void init(SmppSessionHandlerFactory sessionHandlerFactory) throws IOException {
        logger.info("Creating the Smpp Server.");
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newCachedThreadPool();
        ScheduledThreadPoolExecutor monitorExecutor =
                (ScheduledThreadPoolExecutor) Executors.newScheduledThreadPool(1, new ThreadFactory() {
                    private AtomicInteger sequence = new AtomicInteger(0);

                    @Override
                    public Thread newThread(Runnable r) {
                        Thread t = new Thread(r);
                        t.setName("SmppServerSessionWindowMonitorPool-" + sequence.getAndIncrement());
                        return t;
                    }
                });

        Properties properties = new Properties();
        properties.load(this.getClass().getResourceAsStream("/system.properties"));
        configuration = new SmppServerConfiguration();
        configuration.setPort(parseInt(properties.getProperty("smpp.server.port")));
        configuration.setNonBlockingSocketsEnabled(true);
        configuration.setDefaultRequestExpiryTimeout(parseInt(properties.getProperty("request.expiry.timeout")));
        configuration.setDefaultWindowMonitorInterval(parseInt(properties.getProperty("window.monitor.interval")));
        configuration.setDefaultWindowSize(parseInt(properties.getProperty("max.window.size")));
        configuration.setDefaultWindowWaitTimeout(configuration.getDefaultRequestExpiryTimeout());
        configuration.setDefaultSessionCountersEnabled(true);
        configuration.setMaxConnectionSize(10);
        configuration.setJmxEnabled(true);
        int maxBatchSize = parseInt(properties.getProperty("max.batch.size"));

        this.sessionHandlerFactory = sessionHandlerFactory;
        this.smppServer =
                new DefaultSmppServer(
                        configuration,
                        new DefaultSmppServerHandler(sessionHandlerFactory), executor, monitorExecutor, true, 1, maxBatchSize);
        logger.info("Starting SMPP server...");
    }

    public boolean start() {
        try {
            logger.info("Starting the Smpp Server - [{}]", configuration);
            smppServer.start();
            return true;
        } catch (SmppChannelException e) {
            logger.error("Error occurred while starting the server", e);
            return false;
        }
    }

    public void stop() {
        logger.info("Stopping the Smpp Server.");
        smppServer.stop();
    }

    private static class DefaultSmppServerHandler implements SmppServerHandler {

        private SmppSessionHandlerFactory sessionHandlerFactory;

        public DefaultSmppServerHandler(SmppSessionHandlerFactory sessionHandlerFactory) {
            this.sessionHandlerFactory = sessionHandlerFactory;
        }

        @Override
        public void sessionBindRequested(Long sessionId,
                                         SmppSessionConfiguration sessionConfiguration,
                                         final BaseBind bindRequest) throws SmppProcessingException {
            sessionConfiguration.setName("Application.SMPP." + sessionConfiguration.getSystemId());
        }

        @Override
        public void sessionCreated(Long sessionId, SmppServerSession session, BaseBindResp preparedBindResponse)
                throws SmppProcessingException {
            logger.info("Session created: {}", session);
            session.serverReady(sessionHandlerFactory.createSmppSessionHandler(session));
        }

        @Override
        public void sessionDestroyed(Long sessionId, SmppServerSession session) {
            logger.info("Session destroyed: {}", session);
            if (session.hasCounters()) {
                logger.info(" final session rx-submitSM: {}", session.getCounters().getRxSubmitSM());
            }
            session.destroy();
        }
    }
}
