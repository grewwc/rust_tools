package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import org.junit.Test;

public final class TlsTrustConfigTest {
    private static final Path RESOURCES = Paths.get("src", "main", "res");

    @Test
    public void bundledCaIsValidAndReferencedByNetworkSecurityConfig() throws Exception {
        String config = new String(
                Files.readAllBytes(RESOURCES.resolve("xml/network_security_config.xml")),
                StandardCharsets.UTF_8);
        assertTrue(config.contains("<certificates src=\"@raw/a_serve_ca\" />"));
        assertTrue(config.contains("cleartextTrafficPermitted=\"false\""));

        try (InputStream input = Files.newInputStream(RESOURCES.resolve("raw/a_serve_ca.pem"))) {
            X509Certificate ca = (X509Certificate) CertificateFactory.getInstance("X.509")
                    .generateCertificate(input);
            assertEquals(ca.getSubjectX500Principal(), ca.getIssuerX500Principal());
            assertTrue("bundled certificate must be a CA", ca.getBasicConstraints() >= 0);
            ca.checkValidity();
        }
    }
}