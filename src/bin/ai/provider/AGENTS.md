# Provider Guide

## Scope

Applies to `src/bin/ai/provider/**`. `ApiProvider` (`mod.rs`) defines the request
adapter axis; `ProviderAdapter` and `adapter_for` (`adapter/mod.rs`) own wire
adaptation and dispatch. Endpoint variants such as OpenRouter can use adapter
dispatch without a new `ApiProvider` variant. Keep the thinking dialect
orthogonal to provider selection.

## Key invariants

1. **Adapter hooks over conditionals.** Provider-specific differences belong in
   adapter hooks, not scattered conditionals across the request pipeline.
2. **Adapter vs platform.** `ApiProvider` is the request `adapter` axis; model
   metadata (reasoning flags, endpoints, rare `platform` branding) lives in
   model registry, and request behavior keys off the adapter.
3. **Wire-format gate.** Request-body or stream-format behavior changes need
   focused regression coverage for the affected adapters, including differing
   wire formats. Provider wire-shape fixtures live in
   `src/bin/ai/stream/runtime/tests/`.
