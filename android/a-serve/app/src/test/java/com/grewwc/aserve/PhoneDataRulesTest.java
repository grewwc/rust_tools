package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class PhoneDataRulesTest {
    @Test
    public void normalizesOnlyDialableNumbersWithAtLeastThreeDigits() {
        assertEquals("+8613800138000", PhoneDataRules.normalizeDialable("+86 (138) 0013-8000"));
        assertEquals("10086", PhoneDataRules.normalizeDialable("10086"));
        assertNull(PhoneDataRules.normalizeDialable("12"));
        assertNull(PhoneDataRules.normalizeDialable("1+23"));
        assertNull(PhoneDataRules.normalizeDialable("123;456"));
    }

    @Test
    public void blocksCommonEmergencyNumbers() {
        for (String number : new String[] {"000", "110", "112", "119", "120", "122", "911", "999"}) {
            assertTrue(number, PhoneDataRules.isBlockedEmergency(number));
        }
        assertFalse(PhoneDataRules.isBlockedEmergency("10086"));
    }

    @Test
    public void mapsCallAndContactTypesWithoutExposingProviderConstants() {
        assertEquals(2, PhoneDataRules.callTypeCode("outgoing"));
        assertEquals(0, PhoneDataRules.callTypeCode("all"));
        assertEquals(-1, PhoneDataRules.callTypeCode("invalid"));
        assertEquals("missed", PhoneDataRules.callTypeName(3));
        assertEquals("unknown", PhoneDataRules.callTypeName(99));
        assertEquals("mobile", PhoneDataRules.phoneTypeName(2, null));
        assertEquals("Satellite", PhoneDataRules.phoneTypeName(0, " Satellite "));
    }
}
