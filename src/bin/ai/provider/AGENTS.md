# Provider Guide

## Scope

Applies to `src/bin/ai/provider/**`.
Layout: `mod.rs` defines the `ApiProvider` enum and re-exports; `adapter/`
holds the `ProviderAdapter` trait + per-provider impls (`alibaba`, `compatible`,
`openai`, `opencode`, `openrouter`, `thinking`). Key areas: request fields and
endpoint selection in adapter hooks, reasoning/thinking dialect per adapter, and
stream chunk parsing via `ProviderAdapter::parse_provider_chunk`.

## Key invariants

1. **Adapter hooks over conditionals.** Provider-specific differences belong in
   adapter hooks, not scattered conditionals across the request pipeline.
2. **Shared types.** `provider/mod.rs` defines shared provider enums and common
   types.
3. **Adapter vs platform.** `ApiProvider` is the request `adapter` axis. Optional
   platform branding lives in `models.json` as `platform` (rare); request behavior
   keys off the adapter, and model metadata (reasoning flags, endpoints) lives in
   `models.json`.
4. **Wire-format tests.** Request-body or stream-format changes need focused
   tests, especially when formats differ across providers.

