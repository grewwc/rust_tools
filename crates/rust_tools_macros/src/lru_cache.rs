use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitInt, Ident};

pub fn expand_lru_cache(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as CacheArgs);
    let input_fn = parse_macro_input!(item as syn::ItemFn);

    let fn_args = &input_fn.sig.inputs;
    let fn_body = &input_fn.block;
    let vis = &input_fn.vis;
    let attrs = &input_fn.attrs;
    let sig = &input_fn.sig;

    // Check if function has arguments (we need at least one for caching)
    let arg_count = fn_args.iter().filter(|arg| match arg {
        syn::FnArg::Receiver(_) => false,
        syn::FnArg::Typed(_) => true,
    }).count();

    if arg_count == 0 {
        return TokenStream::from(
            syn::Error::new_spanned(
                &input_fn.sig,
                "#[lru_cache] requires function with at least one argument to use as cache key",
            ).into_compile_error()
        );
    }

    // Generate cache variable name (uppercase for static conventions)
    let fn_name = &input_fn.sig.ident;
    let fn_name_upper = fn_name.to_string().to_uppercase();
    let cache_name = quote::format_ident!("__LRU_CACHE_{}", fn_name_upper);

    // Collect argument patterns for key generation
    let key_fields: Vec<_> = fn_args.iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Receiver(_) => None,
            syn::FnArg::Typed(pat_type) => Some(pat_type.pat.clone()),
        })
        .collect();

    // Generate tuple key type using references for get_ref
    let key_ref_type = if arg_count == 1 {
        let ty = fn_args.iter()
            .filter_map(|arg| match arg {
                syn::FnArg::Typed(pat_type) => Some(pat_type.ty.clone()),
                _ => None,
            })
            .next()
            .unwrap();
        quote! { #ty }
    } else {
        let types: Vec<syn::Type> = fn_args.iter()
            .filter_map(|arg| match arg {
                syn::FnArg::Typed(pat_type) => Some((*pat_type.ty).clone()),
                _ => None,
            })
            .collect();
        quote! { (#(#types),*) }
    };

    // Extract return type
    let output = &input_fn.sig.output;
    let ret_type = match output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };

    let cap = args.cap;
    let ttl_ms = args.ttl_ms;
    let cache_init = if ttl_ms >= 0 {
        quote! { std::sync::Mutex::new(LruCache::with_ttl(#cap, #ttl_ms)) }
    } else {
        quote! { std::sync::Mutex::new(LruCache::new(#cap)) }
    };

    // Generate wrapped function using get_ref for borrowing
    let wrapped = quote! {
        #(#attrs)*
        #vis #sig {
            use rust_tools::cw::lru_cache::LruCache;

            // Use LazyLock for lazy initialization of the Mutex<LruCache<...>>
            static #cache_name: std::sync::LazyLock<std::sync::Mutex<LruCache<#key_ref_type, #ret_type>>> = 
                std::sync::LazyLock::new(|| #cache_init);

            let mut cache = #cache_name.lock().unwrap();

            // Use get_ref to borrow keys without moving
            if let Some(result) = cache.get_ref(&(#(#key_fields),*)) {
                return result;
            }

            let result = (|| #fn_body)();
            cache.put((#(#key_fields),*), result.clone());
            result
        }
    };

    TokenStream::from(wrapped)
}

struct CacheArgs {
    cap: usize,
    ttl_ms: i64,
}

impl syn::parse::Parse for CacheArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut cap = 100usize;
        let mut ttl_ms = -1i64;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            let _: syn::Token![=] = input.parse()?;

            match ident.to_string().as_str() {
                "cap" => {
                    let lit: LitInt = input.parse()?;
                    cap = lit.base10_parse()?;
                }
                "ttl_ms" => {
                    let lit: LitInt = input.parse()?;
                    ttl_ms = lit.base10_parse()?;
                }
                _ => return Err(syn::Error::new(ident.span(), "expected `cap` or `ttl_ms`")),
            }

            if !input.is_empty() {
                let _: syn::Token![,] = input.parse()?;
            }
        }

        Ok(CacheArgs { cap, ttl_ms })
    }
}

impl Default for CacheArgs {
    fn default() -> Self {
        CacheArgs { cap: 100, ttl_ms: -1 }
    }
}
