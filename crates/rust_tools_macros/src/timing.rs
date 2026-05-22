use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, LitStr, parse::Parser, parse_macro_input};

fn timing_stmt(label: &str, debug_only: bool) -> proc_macro2::TokenStream {
    if debug_only {
        quote! {
            if cfg!(debug_assertions) {
                let __measure_elapsed = __measure_start.elapsed();
                ::std::eprintln!(
                    "[timing] {} took {:.2} ms",
                    #label,
                    __measure_elapsed.as_secs_f64() * 1000.0
                );
            }
        }
    } else {
        quote! {
            let __measure_elapsed = __measure_start.elapsed();
            ::std::eprintln!(
                "[timing] {} took {:.2} ms",
                #label,
                __measure_elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}

pub(crate) fn expand_timing_attr(
    attr: TokenStream,
    item: TokenStream,
    debug_only: bool,
) -> TokenStream {
    let parser = syn::punctuated::Punctuated::<LitStr, syn::Token![,]>::parse_terminated;
    let args = parser.parse(attr).unwrap_or_default();
    let input_fn = parse_macro_input!(item as ItemFn);

    let attrs = input_fn.attrs;
    let vis = input_fn.vis;
    let sig = input_fn.sig;
    let block = input_fn.block;
    let fn_name = sig.ident.to_string();
    let label = args
        .first()
        .map(|lit| lit.value())
        .unwrap_or_else(|| fn_name.clone());
    let timing_stmt = timing_stmt(&label, debug_only);

    let wrapped_block = if sig.asyncness.is_some() {
        quote!({
            let __measure_start = ::std::time::Instant::now();
            let __measure_result = (async move #block).await;
            #timing_stmt
            __measure_result
        })
    } else {
        quote!({
            let __measure_start = ::std::time::Instant::now();
            let __measure_result = (|| #block)();
            #timing_stmt
            __measure_result
        })
    };

    TokenStream::from(quote! {
        #(#attrs)*
        #vis #sig #wrapped_block
    })
}
