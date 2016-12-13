package psaw.java.smpp.boot;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.springframework.context.support.AbstractApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.tanukisoftware.wrapper.WrapperListener;
import org.tanukisoftware.wrapper.WrapperManager;

/**
 * <p>
 * <code>SimulatorStarter</code> -
 * Starter of the SMPP Simulator.
 * </p>
 *
 * @author prabath.
 */
public class SimulatorStarter implements WrapperListener {

    private static final Logger logger = LogManager.getLogger(SimulatorStarter.class);

    private AbstractApplicationContext applicationContext;

    public static void main(String[] args) {
        WrapperManager.start(new SimulatorStarter(), args);
    }

    @Override
    public Integer start(String[] strings) {
        try {
            logger.info("Initializing SMPP Simulator.....");
            applicationContext =
                    new ClassPathXmlApplicationContext(new String[]{"classpath:simulator-integration-spring.xml"});
            logger.info("===========================================");
            logger.info("======  SMPP Simulator Starting =====");
            logger.info("===========================================");
            applicationContext.start();
            logger.info("===========================================");
            logger.info("======  SMPP Simulator Started   ========");
            logger.info("===========================================");
            Thread.sleep(100000);

        } catch (Exception e) {
            logger.error("===== SMPP Simulator Initialization Failed  =====", e);
        }
        return null;
    }

    @Override
    public int stop(int i) {
        logger.info("===========================================");
        logger.info("====== SMPP Simulator Stopping ======");
        logger.info("===========================================");

        if (applicationContext != null && applicationContext.isRunning()) {
            applicationContext.stop();
        }
        logger.info("===========================================");
        logger.info("====== SMPP Simulator Stopped=======");
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
