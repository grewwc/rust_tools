use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{
    Expr, ItemFn, LitStr, Token,
    braced,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

struct AgentHangDebugArgs {
    run_id: Expr,
    _comma1: Token![,],
    hypothesis_id: Expr,
    _comma2: Token![,],
    location: Expr,
    _comma3: Token![,],
    msg: Expr,
    _comma4: Token![,],
    data_tokens: TokenStream2,
}

impl Parse for AgentHangDebugArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let run_id = input.parse()?;
        let _comma1 = input.parse()?;
        let hypothesis_id = input.parse()?;
        let _comma2 = input.parse()?;
        let location = input.parse()?;
        let _comma3 = input.parse()?;
        let msg = input.parse()?;
        let _comma4 = input.parse()?;
        let mut data_tokens: TokenStream2 = input.parse()?;
        data_tokens = strip_trailing_top_level_comma(data_tokens);
        Ok(Self {
            run_id,
            _comma1,
            hypothesis_id,
            _comma2,
            location,
            _comma3,
            msg,
            _comma4,
            data_tokens,
        })
    }
}

fn strip_trailing_top_level_comma(tokens: TokenStream2) -> TokenStream2 {
    let mut items = tokens.into_iter().collect::<Vec<_>>();
    if matches!(
        items.last(),
        Some(TokenTree::Punct(p)) if p.as_char() == ','
    ) {
        items.pop();
    }
    items.into_iter().collect()
}

pub(crate) fn expand_agent_hang_debug(input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(input as AgentHangDebugArgs);
    let run_id = args.run_id;
    let hypothesis_id = args.hypothesis_id;
    let location = args.location;
    let msg = args.msg;
    let data_tokens = args.data_tokens;

    TokenStream::from(quote! {
        crate::ai::driver::turn_runtime::report_agent_hang_debug(
            #run_id,
            #hypothesis_id,
            #location,
            #msg,
            serde_json::json!(#data_tokens),
        )
    })
}

struct AgentHangSpanArgs {
    run_id: LitStr,
    _comma1: Token![,],
    hypothesis_id: LitStr,
    _comma2: Token![,],
    location: LitStr,
    _comma3: Token![,],
    begin_msg: LitStr,
    _comma4: Token![,],
    end_msg: LitStr,
    _comma5: Token![,],
    begin_data_tokens: TokenStream2,
    _comma6: Token![,],
    end_data_tokens: TokenStream2,
    _trailing_comma: Option<Token![,]>,
}

impl Parse for AgentHangSpanArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let run_id = input.parse()?;
        let _comma1 = input.parse()?;
        let hypothesis_id = input.parse()?;
        let _comma2 = input.parse()?;
        let location = input.parse()?;
        let _comma3 = input.parse()?;
        let begin_msg = input.parse()?;
        let _comma4 = input.parse()?;
        let end_msg = input.parse()?;
        let _comma5 = input.parse()?;
        let begin_data_tokens = parse_braced_json_body(input)?;
        let _comma6 = input.parse()?;
        let end_data_tokens = parse_braced_json_body(input)?;
        Ok(Self {
            run_id,
            _comma1,
            hypothesis_id,
            _comma2,
            location,
            _comma3,
            begin_msg,
            _comma4,
            end_msg,
            _comma5,
            begin_data_tokens,
            _comma6,
            end_data_tokens,
            _trailing_comma: input.parse().ok(),
        })
    }
}

fn parse_braced_json_body(input: ParseStream) -> syn::Result<TokenStream2> {
    let content;
    braced!(content in input);
    let body: TokenStream2 = content.parse()?;
    Ok(body)
}

pub(crate) fn expand_agent_hang_span(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AgentHangSpanArgs);
    let input_fn = parse_macro_input!(item as ItemFn);

    let attrs = input_fn.attrs;
    let vis = input_fn.vis;
    let sig = input_fn.sig;
    let block = input_fn.block;

    let run_id = args.run_id;
    let hypothesis_id = args.hypothesis_id;
    let begin_msg = args.begin_msg;
    let end_msg = args.end_msg;
    let begin_data_tokens = args.begin_data_tokens;
    let end_data_tokens = args.end_data_tokens;

    let location_base = args.location.value();
    let begin_location = LitStr::new(
        &format!("{location_base}:begin"),
        args.location.span(),
    );
    let end_location = LitStr::new(
        &format!("{location_base}:end"),
        args.location.span(),
    );

    let wrapped_block = if sig.asyncness.is_some() {
        quote!({
            #[cfg(feature = "agent-hang-debug")]
            {
                crate::ai::driver::turn_runtime::report_agent_hang_debug(
                    #run_id,
                    #hypothesis_id,
                    #begin_location,
                    #begin_msg,
                    serde_json::json!({ #begin_data_tokens }),
                );
                let __agent_hang_start = ::std::time::Instant::now();
                let __agent_hang_result = (async move #block).await;
                let __agent_hang_elapsed_ms = __agent_hang_start.elapsed().as_secs_f64() * 1000.0;
                crate::ai::driver::turn_runtime::report_agent_hang_debug(
                    #run_id,
                    #hypothesis_id,
                    #end_location,
                    #end_msg,
                    serde_json::json!({ #end_data_tokens }),
                );
                __agent_hang_result
            }
            #[cfg(not(feature = "agent-hang-debug"))]
            {
                (async move #block).await
            }
        })
    } else {
        quote!({
            #[cfg(feature = "agent-hang-debug")]
            {
                crate::ai::driver::turn_runtime::report_agent_hang_debug(
                    #run_id,
                    #hypothesis_id,
                    #begin_location,
                    #begin_msg,
                    serde_json::json!({ #begin_data_tokens }),
                );
                let __agent_hang_start = ::std::time::Instant::now();
                let __agent_hang_result = (|| #block)();
                let __agent_hang_elapsed_ms = __agent_hang_start.elapsed().as_secs_f64() * 1000.0;
                crate::ai::driver::turn_runtime::report_agent_hang_debug(
                    #run_id,
                    #hypothesis_id,
                    #end_location,
                    #end_msg,
                    serde_json::json!({ #end_data_tokens }),
                );
                __agent_hang_result
            }
            #[cfg(not(feature = "agent-hang-debug"))]
            {
                (|| #block)()
            }
        })
    };

    TokenStream::from(quote! {
        #(#attrs)*
        #vis #sig #wrapped_block
    })
}
