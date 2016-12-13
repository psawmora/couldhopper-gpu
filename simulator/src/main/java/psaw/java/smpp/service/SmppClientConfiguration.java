package psaw.java.smpp.service;

/**
 * <p>
 * <code>SmppClientConfiguration</code> -
 * Holds SMPP Client Configurations.
 * </p>
 *
 * @author prabath.
 */
public class SmppClientConfiguration {

    private String host;

    private int port;

    private String systemId;

    private String password;

    private String clientAddress;

    private String destinationNumberRange;

    private String tpsString;

    private String text;

    private SmppClientConfiguration(Builder builder) {
        host = builder.host;
        port = builder.port;
        systemId = builder.systemId;
        password = builder.password;
        clientAddress = builder.clientAddress;
        destinationNumberRange = builder.destincationNumberRange;
        tpsString = builder.tpsString;
        text = builder.text;
    }

    public static Builder newBuilder() {
        return new Builder();
    }

    public static final class Builder {

        private String host;

        private int port;

        private String systemId;

        private String password;

        private String clientAddress;

        private String destincationNumberRange;

        private String tpsString;

        private String text;

        private Builder() {
        }

        public Builder withHost(String host) {
            this.host = host;
            return this;
        }

        public Builder withPort(int port) {
            this.port = port;
            return this;
        }

        public Builder withSystemId(String systemId) {
            this.systemId = systemId;
            return this;
        }

        public Builder withPassword(String password) {
            this.password = password;
            return this;
        }

        public Builder withClientAddress(String clientAddress) {
            this.clientAddress = clientAddress;
            return this;
        }

        public Builder withDestincationNumberRange(String destincationNumberRange) {
            this.destincationNumberRange = destincationNumberRange;
            return this;
        }

        public Builder withTpsString(String tpsString) {
            this.tpsString = tpsString;
            return this;
        }

        public Builder withText(String text) {
            this.text = text;
            return this;
        }

        public SmppClientConfiguration build() {
            return new SmppClientConfiguration(this);
        }
    }

    public String getHost() {
        return host;
    }

    public int getPort() {
        return port;
    }

    public String getSystemId() {
        return systemId;
    }

    public String getPassword() {
        return password;
    }

    public String getClientAddress() {
        return clientAddress;
    }

    public String getDestinationNumberRange() {
        return destinationNumberRange;
    }

    public String getTpsString() {
        return tpsString;
    }

    public String getText() {
        return text;
    }
}
