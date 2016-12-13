package psaw.java.smpp.service;

import com.cloudhopper.smpp.SmppConstants;
import com.cloudhopper.smpp.pdu.SubmitSm;
import com.cloudhopper.smpp.type.SmppInvalidArgumentException;
import psaw.java.smpp.client.SmppPduGenerator;
import psaw.java.smpp.util.SequenceNumberGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static java.util.concurrent.TimeUnit.MICROSECONDS;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

/**
 * <p>
 * <code>SmppRequestGenerator</code> -
 * Generates SMPP requests as per a given pattern.
 * </p>
 *
 * @author prabath.
 */
public class SmppRequestGenerator {

    private String pattern;

    private final SmppPduGenerator pduGenerator;

    private final String message;

    private List<TpsAndTimeLimit> tpsAndTimeLimits;

    private ScheduledExecutorService messageGenerateScheduler;

    private ScheduledExecutorService tpsResetScheduler;

    private volatile ScheduledFuture messageGeneratorFuture;

    private SequenceNumberGenerator sequenceNumberGenerator;

    private List<Integer> tpsChain;

    private AtomicInteger nextIndex;

    private int currentTps;

    private AtomicInteger count = new AtomicInteger(0);

    public SmppRequestGenerator(String pattern,
                                SmppPduGenerator pduGenerator,
                                SequenceNumberGenerator sequenceNumberGenerator,
                                String message) {
        this.pattern = pattern;
        this.pduGenerator = pduGenerator;
        this.sequenceNumberGenerator = sequenceNumberGenerator;
        this.message = message;
    }

    public void init() {
        this.nextIndex = new AtomicInteger(0);
        this.messageGenerateScheduler = Executors.newScheduledThreadPool(1);
        this.tpsResetScheduler = Executors.newScheduledThreadPool(1);
        this.tpsAndTimeLimits = new ArrayList<>();
        String[] tpsTimeValues = pattern.split("\\|");
        for (String tpsTimeValue : tpsTimeValues) {
            String[] tpsAndTimeDuo = tpsTimeValue.split(":");
            tpsAndTimeLimits.add(new TpsAndTimeLimit(Integer.parseInt(tpsAndTimeDuo[0]), Integer.parseInt(tpsAndTimeDuo[1])));
        }
        this.tpsChain = createTpsChain();
    }

    private List<Integer> createTpsChain() {
        List<Integer> tpsChain = new ArrayList<>(0);
        for (TpsAndTimeLimit timeLimit : tpsAndTimeLimits) {
            Integer[] tpsValues = new Integer[timeLimit.timeLimit];
            Arrays.fill(tpsValues, timeLimit.tps);
            tpsChain.addAll(Arrays.asList(tpsValues));
        }
        return tpsChain;
    }

    public void start() {
        long period = calculatePeriod(tpsAndTimeLimits.get(0).tps);
        currentTps = tpsAndTimeLimits.get(0).tps;
        scheduleMessagesAtFixRate(period);
        tpsResetScheduler.scheduleAtFixedRate(new Runnable() {
            @Override
            public void run() {
                int index = nextIndex.getAndIncrement() % tpsChain.size();
                int nextTps = tpsChain.get(index);
                if (nextTps != currentTps) {
                    currentTps = nextTps;
                    resetTps(nextTps);
                }

            }
        }, 1000, 1000, MILLISECONDS);
    }

    private void resetTps(int nextTps) {
        if (messageGeneratorFuture != null) {
            messageGeneratorFuture.cancel(false);
        }
        scheduleMessagesAtFixRate(calculatePeriod(nextTps));
    }

    private void scheduleMessagesAtFixRate(long period) {
        messageGeneratorFuture = messageGenerateScheduler.scheduleAtFixedRate((Runnable) () -> {
            try {
                SubmitSm submitSm = new SubmitSm();
                submitSm.setSequenceNumber(sequenceNumberGenerator.getSequenceNumber());
                submitSm.setEsmClass(SmppConstants.ESM_CLASS_MT_SMSC_DELIVERY_RECEIPT);
                submitSm.setServiceType("t1");
                submitSm.setShortMessage(message.getBytes());
//                submitSm.addOptionalParameter(new Tlv(SmppConstants.TAG_USER_MESSAGE_REFERENCE, "test-tlv".getBytes()));
//                submitSm.addOptionalParameter(new Tlv(SmppConstants.TAG_SOURCE_PORT, "1122".getBytes()));
                //                if(count.incrementAndGet() <= 100000){
                pduGenerator.sendSubmitSm(submitSm, 2000);
                //                }
                //                pduGenerator.sendSubmitSm(submitSm, 2000);
            } catch (SmppInvalidArgumentException e) {
                return;
            }
        }, 1000, period, MICROSECONDS);
    }

    private long calculatePeriod(int tps) {
        return 1000 * 1000 / tps;
    }

    private static class TpsAndTimeLimit {

        int tps;

        int timeLimit;

        public TpsAndTimeLimit(int tps, int timeLimit) {
            this.tps = tps;
            this.timeLimit = timeLimit;
        }
    }
}
