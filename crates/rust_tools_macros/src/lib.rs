use proc_macro::TokenStream;
mod agent_hang;
mod lru_cache;
mod timing;

#[proc_macro_attribute]
pub fn measure_time(attr: TokenStream, item: TokenStream) -> TokenStream {
    timing::expand_timing_attr(attr, item, false)
}

#[proc_macro_attribute]
pub fn debug_measure_time(attr: TokenStream, item: TokenStream) -> TokenStream {
    timing::expand_timing_attr(attr, item, true)
}

#[proc_macro]
pub fn agent_hang_debug(input: TokenStream) -> TokenStream {
    agent_hang::expand_agent_hang_debug(input)
}

#[proc_macro_attribute]
pub fn agent_hang_span(attr: TokenStream, item: TokenStream) -> TokenStream {
    agent_hang::expand_agent_hang_span(attr, item)
}

#[proc_macro_attribute]
pub fn lru_cache(attr: TokenStream, item: TokenStream) -> TokenStream {
    lru_cache::expand_lru_cache(attr, item)
}
