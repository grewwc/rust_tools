# AI Provider Detailed Guide

## Scope

Long-form reference for `src/bin/ai/provider/**`.

## Architecture

- `provider/mod.rs` defines `ApiProvider`, `ModelQualityTier`,
  `ReasoningEffort`, and shared provider-facing types.
- `provider/adapter.rs` hosts the `ProviderAdapter` trait and the zero-state
  adapters.
- The main request pipeline stays mostly provider-agnostic and calls adapter
  hooks only where providers actually differ.

## What belongs in adapters

- request-body field names and shapes (`enable_thinking`, `enable_search`,
  `reasoning_effort`, nested `reasoning`)
- default endpoint and API-key resolution
- stream-chunk parsing differences
- waiting-hint UX differences
- rules for disabling thinking on auxiliary tasks

## What should stay out of adapters

- generic history shaping
- unrelated business logic
- duplicated request-building logic that can stay in the common pipeline

## Editing rules

- Prefer extending adapter hooks over adding scattered `match provider` branches
  across request/model/stream code.
- When wire-format behavior changes, update focused tests that guard provider
  request bodies or stream parsing.
- If endpoint detection depends on provider + endpoint combination (for example
  OpenRouter detection), keep that logic centralized.

## Useful files

- `request.rs`
- `models.rs`
- `stream/normalize.rs`
- `stream/runtime.rs`
- `driver/reflection/background.rs`
