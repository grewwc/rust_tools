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
                        "http://schemas.android.com/apk/res/android", "name");
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
}
