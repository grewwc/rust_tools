# Provider Guide

## Scope

Applies to `src/bin/ai/provider/**`. `mod.rs` defines the `ApiProvider` enum +
shared types; `adapter/` holds the `ProviderAdapter` trait + per-provider impls
(`alibaba`, `compatible`, `openai`, `opencode`, `openrouter`, `thinking`).

## Key invariants

1. **Adapter hooks over conditionals.** Provider-specific differences belong in
   adapter hooks, not scattered conditionals across the request pipeline.
2. **Adapter vs platform.** `ApiProvider` is the request `adapter` axis; model
   metadata (reasoning flags, endpoints, rare `platform` branding) lives in
   model registry, and request behavior keys off the adapter.
3. **Wire-format tests.** Request-body or stream-format changes need focused
   tests, especially when formats differ across providers.
