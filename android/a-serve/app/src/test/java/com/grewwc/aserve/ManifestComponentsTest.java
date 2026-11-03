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

    @Test
    public void appDiscoveryIsLimitedToLauncherActivities() throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        Element manifest = factory.newDocumentBuilder().parse(
                Paths.get("src", "main", "AndroidManifest.xml").toFile()).getDocumentElement();
        boolean launcherAction = false;
        boolean launcherCategory = false;
        for (int i = 0; i < manifest.getElementsByTagName("queries").getLength(); i++) {
            Element queries = (Element) manifest.getElementsByTagName("queries").item(i);
            NodeList actions = queries.getElementsByTagName("action");
            for (int j = 0; j < actions.getLength(); j++)
                launcherAction |= "android.intent.action.MAIN".equals(
                        ((Element) actions.item(j)).getAttributeNS(ANDROID_NS, "name"));
            NodeList categories = queries.getElementsByTagName("category");
            for (int j = 0; j < categories.getLength(); j++)
                launcherCategory |= "android.intent.category.LAUNCHER".equals(
                        ((Element) categories.item(j)).getAttributeNS(ANDROID_NS, "name"));
        }
        assertTrue("launcher action must be queryable", launcherAction);
        assertTrue("launcher category must be queryable", launcherCategory);
        NodeList permissions = manifest.getElementsByTagName("uses-permission");
        for (int i = 0; i < permissions.getLength(); i++)
            assertFalse("QUERY_ALL_PACKAGES would expose non-launchable apps",
                    "android.permission.QUERY_ALL_PACKAGES".equals(
                            ((Element) permissions.item(i)).getAttributeNS(ANDROID_NS, "name")));
    }
}
