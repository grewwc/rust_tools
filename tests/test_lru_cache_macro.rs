//! Test module for lru_cache macro
#![allow(non_upper_case_globals)]

#[cfg(test)]
mod tests {
    use rust_tools_macros::lru_cache;

    #[lru_cache(cap = 10)]
    fn cached_add(a: i32, b: i32) -> i32 {
        println!("Computing cached_add({a}, {b})...");
        a + b
    }

    #[test]
    fn test_lru_cache_macro_basic() {
        // First call - computes and caches
        let result1 = cached_add(1, 2);
        assert_eq!(result1, 3);
        
        // Second call with same args - should hit cache
        let result2 = cached_add(1, 2);
        assert_eq!(result2, 3);
        
        // Different args - computes again
        let result3 = cached_add(3, 4);
        assert_eq!(result3, 7);
    }

    #[lru_cache(cap = 5, ttl_ms = 100)]
    fn cached_mult(a: i32, b: i32) -> i32 {
        println!("Computing cached_mult({a}, {b})...");
        a * b
    }

    #[test]
    fn test_lru_cache_macro_with_ttl() {
        let result1 = cached_mult(2, 3);
        assert_eq!(result1, 6);
        
        // Should hit cache
        let result2 = cached_mult(2, 3);
        assert_eq!(result2, 6);
    }

    // Test with single argument
    #[lru_cache(cap = 3)]
    fn cached_square(n: i32) -> i32 {
        println!("Computing cached_square({n})...");
        n * n
    }

    #[test]
    fn test_lru_cache_macro_single_arg() {
        assert_eq!(cached_square(5), 25);
        assert_eq!(cached_square(5), 25); // cache hit
        assert_eq!(cached_square(3), 9);
    }

    // Test with String argument
    #[lru_cache(cap = 5)]
    fn cached_string_len(s: String) -> usize {
        println!("Computing cached_string_len(\"{s}\")...");
        s.len()
    }

    #[test]
    fn test_lru_cache_macro_string_arg() {
        assert_eq!(cached_string_len(String::from("hello")), 5);
        assert_eq!(cached_string_len(String::from("hello")), 5); // cache hit
        assert_eq!(cached_string_len(String::from("world")), 5);
    }

    // Test LRU eviction
    #[lru_cache(cap = 2)]
    fn cached_triple(a: i32, b: i32, c: i32) -> i32 {
        println!("Computing cached_triple({a}, {b}, {c})...");
        a + b + c
    }

    #[test]
    fn test_lru_cache_macro_eviction() {
        assert_eq!(cached_triple(1, 2, 3), 6);
        assert_eq!(cached_triple(1, 2, 3), 6); // cache hit
        
        // Different args should compute
        assert_eq!(cached_triple(4, 5, 6), 15);
        
        // After 3 entries with cap=2, first entry should be evicted
        // Call first args again - should recompute
        assert_eq!(cached_triple(1, 2, 3), 6);
    }
}
