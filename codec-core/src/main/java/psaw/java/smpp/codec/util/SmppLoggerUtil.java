package psaw.java.smpp.codec.util;

import com.cloudhopper.smpp.pdu.DeliverSm;
import com.cloudhopper.smpp.pdu.SubmitSm;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * <p>
 * <code>{@link SmppLoggerUtil}</code> -
 * Tps logger for Smpp Decoders.
 * </p>
 *
 * @author psaw.
 */
public class SmppLoggerUtil {

    private static final Logger TPS_LOGGER = LogManager.getLogger("TPS_LOGGER");

    private static AtomicInteger submitSmCount;

    private static AtomicInteger deliverSmCount;

    static {
        init();
    }

    public static void init() {
        submitSmCount = new AtomicInteger(0);
        deliverSmCount = new AtomicInteger(0);
        ScheduledExecutorService executorService = Executors.newScheduledThreadPool(1);
        executorService.scheduleAtFixedRate(() -> {
            TPS_LOGGER.info("Submit-Sm[{}],Deliver-Sm[{}]", submitSmCount.getAndSet(0), deliverSmCount.getAndSet(0));

        }, 1000, 1000*5, TimeUnit.MILLISECONDS);
    }

    /**
     * <p>
     * format - submit_sm_count,deliver_sm_count
     * </p>
     *
     * @param submitSms
     */
    public static void logSubmitSm(List<SubmitSm> submitSms) {
        submitSmCount.addAndGet(submitSms.size());
    }

    public static void logSubmitSm(SubmitSm submitSm) {
        submitSmCount.incrementAndGet();
    }

    /**
     * <p>
     * format - submit_sm_count,deliver_sm_count
     * </p>
     *
     * @param deliverSms
     */
    public static void logDeliverSm(List<DeliverSm> deliverSms) {
        deliverSmCount.addAndGet(deliverSms.size());
    }

    public static void logDeliverSm(DeliverSm deliverSms) {
        deliverSmCount.incrementAndGet();
    }

}
