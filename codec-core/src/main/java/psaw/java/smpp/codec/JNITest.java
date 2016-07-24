package psaw.java.smpp.codec;

/**
 * <p></p>
 *
 * @author prabath.
 */
public class JNITest {

    public native void testCall(String name);

    public static void main(String[] args) {
        System.loadLibrary("jni-test");
    }
}
