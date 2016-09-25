package com.cloudhopper.smpp.transcoder.asynchronous;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * <p>
 * <code>{@link ChannelIdUtil}</code> -
 * ChannelId generator - length=15
 * </p>
 *
 * @author psaw.
 */
public class ChannelIdUtil {

    private static DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyyMMddmmss");

    private static AtomicInteger count;

    private static NumberFormat numberFormat = new DecimalFormat("000");

    static {
        count = new AtomicInteger(0);
    }

    public static String getChannelId() {
        count.compareAndSet(999, 0);
        int nextCount = count.getAndUpdate((i) -> i >= 998 ? 0 : i + 1);
        return formatter.format(LocalDateTime.now()).concat(numberFormat.format(nextCount));
    }
}
