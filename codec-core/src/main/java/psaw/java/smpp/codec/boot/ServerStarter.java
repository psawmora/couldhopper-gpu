package psaw.java.smpp.codec.boot;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.tanukisoftware.wrapper.WrapperListener;
import org.tanukisoftware.wrapper.WrapperManager;
import psaw.java.smpp.codec.server.ServerInitializer;

import java.io.IOException;

/**
 * <p>
 * <code>{@link ServerStarter}</code> -
 * Java service wrapper implementation.
 * </p>
 *
 * @author psaw.
 */
public class ServerStarter implements WrapperListener {

    private static final Logger logger = LogManager.getLogger(ServerStarter.class);

    private AbstractApplicationContext applicationContext;

    public static void main(String[] args) {
        WrapperManager.start(new ServerStarter(), args);

    }

    @Override
    public Integer start(String[] args) {
        logger.info("Initializing SMPP Server.....");
        logger.info("===========================================");
        logger.info("======  SMPP Server Starting =====");
        applicationContext = new ClassPathXmlApplicationContext(new String[]{"classpath:smpp-server-integration-spring.xml"});
        ServerInitializer serverInitializer = (ServerInitializer) applicationContext.getBean("serverInitializer");
        try {
            serverInitializer.start();
        } catch (IOException e) {
            logger.error("Error occurred while starting the server.", e);
            stop(0);
            return -1;
        }
        logger.info("===========================================");
        logger.info("===========================================");
        logger.info("======  SMPP Server Started   ========");
        logger.info("===========================================");
        return null;
    }

    @Override
    public int stop(int i) {
        logger.info("===========================================");
        logger.info("====== SMPP Server Stopping ======");
        logger.info("===========================================");
        if (applicationContext != null && applicationContext.isRunning()) {
            applicationContext.stop();
        }
        logger.info("===========================================");
        logger.info("====== SMPP Server Stopped=======");
        logger.info("===========================================");
        return 0;
    }

    @Override
    public void controlEvent(int event) {
        if ((event == WrapperManager.WRAPPER_CTRL_LOGOFF_EVENT)
                && WrapperManager.isLaunchedAsService()) {
        } else {
            WrapperManager.stop(0);
        }

    }
}
