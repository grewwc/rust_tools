package com.grewwc.aserve;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Modifier;
import java.nio.file.Paths;
import javax.xml.parsers.DocumentBuilderFactory;
import org.junit.Test;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

/** Manifest components are constructed by Android, outside the app's package. */
public final class ManifestComponentsTest {
    private static final String ANDROID_NS = "http://schemas.android.com/apk/res/android";

    @Test
    public void frameworkComponentsArePublicWithPublicNoArgConstructors() throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        Element manifest = factory.newDocumentBuilder().parse(
                Paths.get("src", "main", "AndroidManifest.xml").toFile()).getDocumentElement();
        int checked = 0;
        for (String tag : new String[] {"application", "activity", "service", "receiver", "provider"}) {
            NodeList components = manifest.getElementsByTagName(tag);
            for (int i = 0; i < components.getLength(); i++) {
                String name = ((Element) components.item(i)).getAttributeNS(
                        ANDROID_NS, "name");
                if (name.isEmpty()) continue;
                if (name.startsWith(".")) name = BuildConfig.APPLICATION_ID + name;
                else if (!name.contains(".")) name = BuildConfig.APPLICATION_ID + "." + name;
                Class<?> component = Class.forName(name, false, getClass().getClassLoader());
                assertTrue(name + " must be public for Android framework instantiation",
                        Modifier.isPublic(component.getModifiers()));
                assertFalse(name + " must be concrete", Modifier.isAbstract(component.getModifiers()));
                assertTrue(name + " must have a public no-argument constructor",
                        Modifier.isPublic(component.getConstructor().getModifiers()));
                checked++;
            }
        }
        assertTrue("the manifest component list must not be empty", checked > 0);
    }

    @Test
    public void phoneDataAndCallPermissionsAreDeclared() throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        Element manifest = factory.newDocumentBuilder().parse(
                Paths.get("src", "main", "AndroidManifest.xml").toFile()).getDocumentElement();
        NodeList permissions = manifest.getElementsByTagName("uses-permission");
        for (String required : new String[] {
                "android.permission.READ_CONTACTS",
                "android.permission.WRITE_CONTACTS",
                "android.permission.READ_CALL_LOG",
                "android.permission.CALL_PHONE"
        }) {
            boolean found = false;
            for (int i = 0; i < permissions.getLength(); i++) {
                if (required.equals(((Element) permissions.item(i)).getAttributeNS(ANDROID_NS, "name"))) {
                    found = true;
                    break;
                }
            }
            assertTrue("missing manifest permission " + required, found);
        }
    }
}
