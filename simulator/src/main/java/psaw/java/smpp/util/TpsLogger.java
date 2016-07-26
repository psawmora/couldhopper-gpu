package psaw.java.smpp.util;

import com.cloudhopper.smpp.pdu.Pdu;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicInteger;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

/**
 * <p>
 * <code>TpsLogger</code> -
 * Logs request and response TPS.
 * </p>
 *
 * @author prabath.
 */
public class TpsLogger {

    private static final Logger logger = LogManager.getLogger("TPS_LOG");

    private static TpsLogger tpsLogger;

    private AtomicInteger requestTps;

    private AtomicInteger responseTps;

    public static TpsLogger getInstance() {
        if (tpsLogger == null) {
            synchronized (TpsLogger.class) {
                if (tpsLogger == null) {
                    tpsLogger = new TpsLogger();
                    tpsLogger.init();
                }
            }
        }
        return tpsLogger;
    }

    public void init() {
        requestTps = new AtomicInteger(0);
        responseTps = new AtomicInteger(0);
        ScheduledExecutorService tpsLogScheduler = Executors.newScheduledThreadPool(1);
        tpsLogScheduler.scheduleAtFixedRate(() -> {
            int requestTps = this.requestTps.getAndSet(0);
            int responseTps = this.responseTps.getAndSet(0);
            logger.info("Request Tps - [{}] | Response Tps - [{}]", requestTps, responseTps);

        }, 1000, 1000, MILLISECONDS);
    }

    public void logRequestSent(Pdu pdu) {
        requestTps.incrementAndGet();
    }

    public void logResponseReceived(Pdu pdu) {
        responseTps.incrementAndGet();
    }
}
