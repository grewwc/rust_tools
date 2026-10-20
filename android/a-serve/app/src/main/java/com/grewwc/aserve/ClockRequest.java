package com.grewwc.aserve;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.List;

/** Validated clock inputs, independent of Android's coercive JSON accessors. */
final class ClockRequest {
    final int hour;
    final int minute;
    final int seconds;
    final String label;
    final List<Integer> weekdays;

    private ClockRequest(int hour, int minute, int seconds, String label, List<Integer> weekdays) {
        this.hour = hour;
        this.minute = minute;
        this.seconds = seconds;
        this.label = label;
        this.weekdays = Collections.unmodifiableList(weekdays);
    }

    static ClockRequest alarm(Object hour, Object minute, Object label, Object weekdays) {
        List<Integer> days = new ArrayList<>();
        if (weekdays != null) {
            if (!(weekdays instanceof List<?>)) throw new IllegalArgumentException("invalid weekdays");
            for (Object value : (List<?>) weekdays) {
                int day = integer(value, 1, 7, "weekday");
                if (days.contains(day)) throw new IllegalArgumentException("duplicate weekday");
                days.add(day);
            }
        }
        return new ClockRequest(integer(hour, 0, 23, "hour"), integer(minute, 0, 59, "minute"),
                0, label(label), days);
    }

    static ClockRequest timer(Object seconds, Object label) {
        return new ClockRequest(0, 0, integer(seconds, 1, 86400, "seconds"), label(label),
                new ArrayList<>());
    }

    ArrayList<Integer> calendarDays() {
        ArrayList<Integer> days = new ArrayList<>();
        for (int day : weekdays) days.add(day == 7 ? Calendar.SUNDAY : day + 1);
        return days;
    }

    private static int integer(Object value, int min, int max, String field) {
        // JSON strings, booleans and floating-point numbers must never be truncated or coerced.
        if (!(value instanceof Integer || value instanceof Long
                || value instanceof Short || value instanceof Byte)) {
            throw new IllegalArgumentException("invalid " + field);
        }
        long number = ((Number) value).longValue();
        if (number < min || number > max) throw new IllegalArgumentException("invalid " + field);
        return (int) number;
    }

    private static String label(Object value) {
        if (value == null) return "";
        if (!(value instanceof String) || ((String) value).length() > 120) {
            throw new IllegalArgumentException("invalid label");
        }
        return ((String) value).trim();
    }
}
