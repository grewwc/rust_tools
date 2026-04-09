use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, LitStr, parse::Parser, parse_macro_input};

pub(crate) enum TimingMode {
    Eprintln,
    Tracing,
}

fn timing_stmt(label: &str, debug_only: bool, mode: &TimingMode) -> proc_macro2::TokenStream {
    match (mode, debug_only) {
        (TimingMode::Eprintln, false) => quote! {
            let __measure_elapsed = __measure_start.elapsed();
            ::std::eprintln!(
                "[timing] {} took {:.2} ms",
                #label,
                __measure_elapsed.as_secs_f64() * 1000.0
            );
        },
        (TimingMode::Eprintln, true) => quote! {
            if cfg!(debug_assertions) {
                let __measure_elapsed = __measure_start.elapsed();
                ::std::eprintln!(
                    "[timing] {} took {:.2} ms",
                    #label,
                    __measure_elapsed.as_secs_f64() * 1000.0
                );
            }
        },
        (TimingMode::Tracing, false) => quote! {
            let __measure_elapsed = __measure_start.elapsed();
            ::tracing::info!(
                target: "timing",
                label = #label,
                elapsed_ms = __measure_elapsed.as_secs_f64() * 1000.0,
                "timing"
            );
        },
        (TimingMode::Tracing, true) => quote! {
            if cfg!(debug_assertions) {
                let __measure_elapsed = __measure_start.elapsed();
                ::tracing::info!(
                    target: "timing",
                    label = #label,
                    elapsed_ms = __measure_elapsed.as_secs_f64() * 1000.0,
                    "timing"
                );
            }
        },
    }
}

pub(crate) fn expand_timing_attr(
    attr: TokenStream,
    item: TokenStream,
    mode: TimingMode,
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
    let timing_stmt = timing_stmt(&label, debug_only, &mode);

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
