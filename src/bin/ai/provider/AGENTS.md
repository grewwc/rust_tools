# Provider Guide

## Scope

Applies to `src/bin/ai/provider/**`.
Read `docs/agent-guides/ai-provider.md` before changing provider-specific
request fields, endpoint selection, reasoning flags, or stream normalization.

## Key invariants

1. **Adapter hooks over conditionals.** Provider-specific differences belong in
   adapter hooks, not scattered conditionals across the request pipeline.
2. **Shared types.** `provider/mod.rs` defines shared provider enums and common
   types.
3. **Adapter vs platform.** `ApiProvider` is the request `adapter` axis. Platform
   branding lives in `models.json` as `platform`; request behavior still keys off
   the adapter.
4. **Wire-format tests.** Request-body or stream-format changes need focused
   tests, especially when formats differ across providers.

## Related detailed guide

- `docs/agent-guides/ai-provider.md`
