package com.grewwc.aserve;

/** Pure validation and labels shared by phone-record reads and direct calls. */
final class PhoneDataRules {
    private PhoneDataRules() {}

    static String normalizeDialable(String value) {
        if (value == null || value.isEmpty() || value.length() > 40) return null;
        StringBuilder normalized = new StringBuilder();
        int digits = 0;
        for (int i = 0; i < value.length(); i++) {
            char ch = value.charAt(i);
            if (ch >= '0' && ch <= '9') {
                normalized.append(ch);
                digits++;
            } else if (ch == '+') {
                if (normalized.length() != 0) return null;
                normalized.append(ch);
            } else if (ch != ' ' && ch != '-' && ch != '(' && ch != ')' && ch != '.') {
                return null;
            }
        }
        return digits >= 3 ? normalized.toString() : null;
    }

    static boolean isBlockedEmergency(String normalized) {
        if (normalized == null) return false;
        String digits = normalized.startsWith("+") ? normalized.substring(1) : normalized;
        return digits.equals("000") || digits.equals("110") || digits.equals("112")
                || digits.equals("119") || digits.equals("120") || digits.equals("122")
                || digits.equals("911") || digits.equals("999");
    }

    static int callTypeCode(String value) {
        if (value == null || value.equals("all")) return 0;
        switch (value) {
            case "incoming": return 1;
            case "outgoing": return 2;
            case "missed": return 3;
            case "voicemail": return 4;
            case "rejected": return 5;
            case "blocked": return 6;
            case "answered_externally": return 7;
            default: return -1;
        }
    }

    static String callTypeName(int value) {
        switch (value) {
            case 1: return "incoming";
            case 2: return "outgoing";
            case 3: return "missed";
            case 4: return "voicemail";
            case 5: return "rejected";
            case 6: return "blocked";
            case 7: return "answered_externally";
            default: return "unknown";
        }
    }

    static String phoneTypeName(int value, String customLabel) {
        switch (value) {
            case 1: return "home";
            case 2: return "mobile";
            case 3: return "work";
            case 4: return "fax_work";
            case 5: return "fax_home";
            case 6: return "pager";
            case 7: return "other";
            case 8: return "callback";
            case 9: return "car";
            case 10: return "company_main";
            case 11: return "isdn";
            case 12: return "main";
            case 13: return "other_fax";
            case 14: return "radio";
            case 15: return "telex";
            case 16: return "tty_tdd";
            case 17: return "work_mobile";
            case 18: return "work_pager";
            case 19: return "assistant";
            case 20: return "mms";
            case 0:
                return customLabel == null || customLabel.trim().isEmpty()
                        ? "custom" : customLabel.trim();
            default: return "unknown";
        }
    }
}
