# Provider Guide

## Scope

Applies to `src/bin/ai/provider/**`. `mod.rs` defines the `ApiProvider` enum
(`compatible`, `alibaba`, `openai`, `opencode`) + shared types; `adapter/` holds
the `ProviderAdapter` trait and its impls — one per enum variant, plus
`openrouter` (an endpoint variant of OpenAI wire format, selected by
`adapter_for` when the endpoint contains `openrouter.ai`; no enum entry, only a
different log label) and `thinking.rs`, the orthogonal thinking-dialect axis.

## Key invariants

1. **Adapter hooks over conditionals.** Provider-specific differences belong in
   adapter hooks, not scattered conditionals across the request pipeline.
2. **Adapter vs platform.** `ApiProvider` is the request `adapter` axis; model
   metadata (reasoning flags, endpoints, rare `platform` branding) lives in
   model registry, and request behavior keys off the adapter.
3. **Wire-format tests.** Request-body or stream-format changes need focused
   tests, especially when formats differ across providers.
