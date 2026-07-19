# Engine KV primitives — measured semantics

> **Support is per engine, not universal.** These results come from probing
> `llama3.2:1b`. The text engines (llama, qwen2, qwen3, gemma, gemma_text,
> gpt_oss, lfm2, nanbeige, phi4) implement `set_context_length`. The
> VL/multimodal engines (qwen2vl, qwen3vl, qwen3_5vl, gemma4e) and the MoE
> engine (qwen3_6_moe) do **not** — they log "Setting context length is not
> supported" and leave the cache unchanged. `AutoModel::truncate_context()`
> detects this on first use and stops asking.

`checkpoint()`, `restore()` and `set_context_length()` are declared in
`causal_lm.hpp` but the engine implementing them ships as a prebuilt library
(`libqwen3_npu.so` and friends), so their behaviour is not visible in source.
`set_context_length()` had no callers anywhere in the tree. This is what a
direct probe against the real engine showed (llama3.2:1b, 64-token prompt).

## checkpoint() / restore()

- `checkpoint()` returns the context length it captured (64 after a 64-token
  prefill).
- Generating past the checkpoint grows `get_current_context_length()` as
  expected (69 after 5 `forward()` calls) and writes cache entries at the new
  positions.
- `restore()` returns the checkpoint length, resets
  `get_current_context_length()` to it, and leaves the cache contents at
  positions below the checkpoint byte-identical.

Single slot: each `checkpoint()` replaces the previous one.

## set_context_length(L)

**It truncates.** `set_context_length(56)` on a 64-token context set
`get_current_context_length()` to 56 and left the retained prefix intact
(fingerprint at position 10 unchanged).

The decisive test: prefill 64 tokens, truncate to 56, prefill the final 8
again. The resulting cache fingerprint at position 60 and the final context
length both match a clean run that prefilled the first 56 and then the last 8
as separate calls. **Truncate-then-reprefill reconstructs exactly the same
state as an uninterrupted prefill**, which is what partial prefix reuse needs.

### Edge cases

- `set_context_length(len + 16)` is *accepted* and reports the larger length,
  but the extra positions were never written. Growth is meaningless and must
  not be used — always clamp to the current length.
- `set_context_length(0)` is accepted and reports 0.
- Neither throws, so the caller is responsible for passing a sane value.

## Consequence

Partial prefix reuse is viable. When an incoming prompt shares a prefix of
length `skip_count` with `token_history`, the engine can be truncated to
`skip_count` and only the remaining tokens prefilled, instead of clearing the
whole context and prefilling everything.

Caller obligations:

- Clamp `L` into `[0, get_current_context_length()]`; never grow.
- Keep the host mirror consistent with the engine: `token_history`,
  `total_tokens` and `checkpoint_his` must be truncated to match, or the
  prefix-matching logic in `_shared_insert` will compare against tokens the
  cache no longer holds.
- Leave at least one token to prefill. Truncating to exactly the prompt length
  leaves nothing to run through the model and so produces no logits to sample.
