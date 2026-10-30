package com.grewwc.aserve;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class SearchActivityTest {
    @Test
    public void buildsOnlyFixedSearchEngineUrls() {
        assertEquals("https://www.bing.com/search?q=rust+async",
                SearchActivity.searchUrl("bing", "rust async"));
        assertEquals("https://www.baidu.com/s?wd=%E4%B8%AD%E6%96%87",
                SearchActivity.searchUrl("baidu", "中文"));
        assertEquals("https://duckduckgo.com/?q=a%26b",
                SearchActivity.searchUrl("duckduckgo", "a&b"));
        assertEquals("https://www.google.com/search?q=x",
                SearchActivity.searchUrl("google", "x"));
        assertNull(SearchActivity.searchUrl("arbitrary", "x"));
        assertNull(SearchActivity.searchUrl("bing", " "));
    }

    @Test
    public void normalizesOnlyAllowlistedEngines() {
        assertEquals("bing", SearchActivity.normalizeEngine(null));
        assertEquals("bing", SearchActivity.normalizeEngine(" "));
        assertEquals("baidu", SearchActivity.normalizeEngine(" BAIDU "));
        assertNull(SearchActivity.normalizeEngine("arbitrary"));
    }

    @Test
    public void keepsSearchConsentAndVerificationHostsInsideTheWebView() {
        assertTrue(SearchActivity.isSearchHost("bing", "www.bing.com"));
        assertTrue(SearchActivity.isSearchHost("google", "www.google.com"));
        assertTrue(SearchActivity.isSearchHost("google", "consent.google.com"));
        assertTrue(SearchActivity.isSearchHost("google", "www.google.co.jp"));
        assertTrue(SearchActivity.isSearchHost("baidu", "wappass.baidu.com"));
        assertTrue(SearchActivity.isSearchHost("duckduckgo", "html.duckduckgo.com"));
        assertFalse(SearchActivity.isSearchHost("bing", "evil.example"));
        assertFalse(SearchActivity.isSearchHost("bing", "bing.com.evil.example"));
        assertFalse(SearchActivity.isSearchHost("bing", "www.bing.com.evil.example"));
        assertFalse(SearchActivity.isSearchHost("google", "www.google.evil.example"));
        assertFalse(SearchActivity.isSearchHost("google", "google.evil.example"));
        assertFalse(SearchActivity.isSearchHost("google", "evilgoogle.com"));
    }
}
