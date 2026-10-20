package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import org.junit.Test;

public final class ClockRequestTest {
    @Test
    public void timerIncludesBothDurationBoundaries() {
        assertEquals(1, ClockRequest.timer(1, null).seconds);
        assertEquals(86400, ClockRequest.timer(86400L, " day ").seconds);
        assertEquals("day", ClockRequest.timer(86400, " day ").label);
        assertEquals(120, ClockRequest.timer(1, "x".repeat(120)).label.length());
    }

    @Test
    public void timerRejectsCoercionFractionsAndOverflow() {
        for (Object invalid : new Object[]{null, "60", true, 1.0, 1.5, Double.NaN,
                Double.POSITIVE_INFINITY, 0, -1, 86401, Long.MAX_VALUE}) {
            assertThrows(String.valueOf(invalid), IllegalArgumentException.class,
                    () -> ClockRequest.timer(invalid, null));
        }
    }

    @Test
    public void clockLabelsAreValidatedRatherThanSilentlyTruncated() {
        assertThrows(IllegalArgumentException.class, () -> ClockRequest.timer(60, "x".repeat(121)));
        assertThrows(IllegalArgumentException.class, () -> ClockRequest.alarm(8, 0, 123, null));
        assertThrows(IllegalArgumentException.class, () -> ClockRequest.alarm(8, 0, " ".repeat(121), null));
    }

    @Test
    public void onceOnlyAlarmDoesNotSupplyCalendarDays() {
        ClockRequest clock = ClockRequest.alarm(0, 0, null, null);
        assertTrue(clock.calendarDays().isEmpty());
        assertTrue(ClockRequest.alarm(23, 59, "", Collections.emptyList()).calendarDays().isEmpty());
    }

    @Test
    public void repeatingAlarmConvertsEveryIsoDayToAndroidCalendar() {
        ClockRequest clock = ClockRequest.alarm(23, 59, "wake", Arrays.asList(1, 2, 3, 4, 5, 6, 7));
        assertEquals(Arrays.asList(Calendar.MONDAY, Calendar.TUESDAY, Calendar.WEDNESDAY,
                Calendar.THURSDAY, Calendar.FRIDAY, Calendar.SATURDAY, Calendar.SUNDAY), clock.calendarDays());
        assertEquals(Arrays.asList(1, 2, 3, 4, 5, 6, 7), clock.weekdays);
        assertThrows(UnsupportedOperationException.class, () -> clock.weekdays.add(1));
    }

    @Test
    public void repeatingAlarmRejectsInvalidOrDuplicateWeekdays() {
        for (Object invalid : new Object[]{"1,2", 1, Arrays.asList(1, 1L), Arrays.asList(0),
                Arrays.asList(8), Arrays.asList(1.0), Arrays.asList("1"), Arrays.asList(true),
                Arrays.asList((Object) null)}) {
            assertThrows(String.valueOf(invalid), IllegalArgumentException.class,
                    () -> ClockRequest.alarm(8, 0, "", invalid));
        }
    }

    @Test
    public void hourAndMinuteAreStrictIntegersWithinRange() {
        for (Object invalid : new Object[]{null, "8", 8.0, 8.5, true, -1, 24, Long.MAX_VALUE}) {
            assertThrows(IllegalArgumentException.class, () -> ClockRequest.alarm(invalid, 0, "", null));
        }
        for (Object invalid : new Object[]{null, "30", 30.0, 30.5, false, -1, 60, Long.MAX_VALUE}) {
            assertThrows(IllegalArgumentException.class, () -> ClockRequest.alarm(8, invalid, "", null));
        }
    }
}
