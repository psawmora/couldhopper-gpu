package psaw.java.smpp.util;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * <p>
 * Thread safe sequence number generator for the Smpp Gateway.
 * </p>
 *
 * @author: psaw
 */
public class SequenceNumberGenerator {

    private int MAX_VAL = 100000;

    private int MIN_VAL = 1;

    private AtomicInteger sequenceNumber;

    public SequenceNumberGenerator() {
    }

    public SequenceNumberGenerator(int MAX_VAL, int MIN_VAL) {
        this.MAX_VAL = MAX_VAL;
        this.MIN_VAL = MIN_VAL;
        sequenceNumber = new AtomicInteger(1);
    }

    /**
     * <p>
     * Initialize the sequence number counter.
     * </p>
     */
    public void init() {
        sequenceNumber = new AtomicInteger();
    }

    /**
     * <p>
     * Returns a sequence number. If Max is reached, then sends from the beginning.
     * </p>
     *
     * @return
     */
    public int getSequenceNumber() {
        int currentValue;
        int nextValue;
        do {
            currentValue = sequenceNumber.get();
            nextValue = (currentValue + 1) == MAX_VAL ? MIN_VAL : (currentValue + 1);
        } while (!sequenceNumber.compareAndSet(currentValue, nextValue));
        return nextValue;
    }

    public void setMAX_VAL(int MAX_VAL) {
        this.MAX_VAL = MAX_VAL;
    }

    public void setMIN_VAL(int MIN_VAL) {
        this.MIN_VAL = MIN_VAL;
    }
}
