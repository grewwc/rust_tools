# Provider Guide

## Scope

Applies to `src/bin/ai/provider/**`.

Read `docs/agent-guides/ai-provider.md` before changing provider-specific
request fields, endpoint selection, reasoning flags, or stream normalization.

## Key invariants

1. Provider-specific differences belong in adapter hooks, not scattered
   conditionals across the pipeline.
2. `provider/mod.rs` defines the shared provider enums and common types.
3. Treat `ApiProvider` as the request `adapter` axis, not the upstream platform
   brand. Platform naming now lives in `models.json` as `platform`; request
   behavior still keys off `adapter`.
4. Request-body or stream-format changes need focused tests, especially when the
   wire format differs across providers.

## Related detailed guide

- `docs/agent-guides/ai-provider.md`
