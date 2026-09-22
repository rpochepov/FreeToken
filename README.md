# Tensor parallelism (TP=4 + EP=4) for `qwen4_exp` / Qwen3.8-Flash-Next-NVFP4

> **Note on this file:** this branch replaces the repository's front-page `README.md` with the
> TP=4 writeup so it renders on the fork's landing page. FreeToken's own README is preserved
> unmodified as [`README.upstream.md`](README.upstream.md) — read that for what FreeToken is,
> installation, and the upstream CLI. Everything below concerns only this fork's `main` branch.

The **`main`** branch of this fork makes one request span **four GPUs** for
`Qwen4ExpForConditionalGeneration`, with routed experts offloaded to host RAM,
262144-token context and working tool calling — on 4× RTX 3090 (24 GiB, sm_86,
**no NVLink**).

Upstream FreeToken refuses this outright:

```
models/qwen4_exp/weight.py:  raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
```

Measured on the reference hardware (4× RTX 3090 24 GiB sm_86 on a SYS/host-bridge
interconnect, EPYC 7543 32c/32t, 503 GiB RAM, model on RAID0 of two NVMe):

| | TP=1 (upstream) | **TP=4 + EP=4 (this branch)** |
|---|---|---|
| single-stream decode | 11.0 tok/s | **64–67 tok/s** |
| expert-cache hit rate | 4.5% (1098 / 24576 slots) | **67% (4124 / 6144 owner slots)** |
| warm prefill | ~8.7k tok/s | **~14.4k tok/s** |
| cold prefill | ~785 tok/s | ~1060 tok/s |
| TTFT at 4k context | 0.72 s | **0.44 s** |
| boot time | ~3 min | **~2 min** (was ~32 min before commit 2) |
| context | 262144 | 262144 |

For scale: four *independent* TP=1 replicas behind a round-robin proxy reach
55.2 tok/s **aggregate**. One request across four GPUs beats that at 66 tok/s.

---

## Read this first: what this branch is based on

**This branch is NOT rebased onto upstream `main`.** It sits on
[`#447`](https://github.com/FlashML-org/FreeToken/pull/447)
(`feat(moe): owner-local expert parallelism and tensor parallelism`) at
**`81d034d`**, which is 18 commits behind upstream `main`. A diff against upstream `main` therefore
reads as `178 files changed, 5837 insertions(+), 7628 deletions(-)` — **the
deletions are the stale base, not removed upstream work.**

It cannot simply be rebased: owner-local expert parallelism does not exist in
upstream `main` at all. Both commits here depend on #447's `moe/ownership.py`,
`--moe-ep-size`, and the owner-aware offload cache.

Commits on this branch, oldest first:

| SHA | what | author |
|---|---|---|
| `71c8ef3` | `feat(moe): owner-local expert parallelism and tensor parallelism` | upstream #447 |
| `81d034d` | `fix(moe): address review — TP>1 loader regression, owner cache state, tests` | upstream #447 |
| `dade0f8` | **fix(qwen4_exp): make TP>2 attention sharding agree with the loader** | this fork |
| `af5a6a4` | **feat(ftw): owner-slice expert banks so owner EP can use an FTW checkpoint** | this fork |
| `1958934` | cherry-pick of `cc1f5c2` — preserve greedy sampling in mixed batches (#471) | cherry77-cloud |
| `195631a` | cherry-pick of `e0886cc` — shared experts before in-place routed experts (#463) | cherry77-cloud |
| `cdb616a` | cherry-pick of `f7dbab7` — load ModelOpt exports with no input_scale (#462) | Xiaoze Fan |

The three cherry-picks are upstream commits re-applied because #447's base predates
them. **Cherry-pick, do not `git merge` upstream `main`** — a full merge costs 12 conflict
hunks across `engine.py`, `scheduler.py`, `qwen4_exp/{config,weight}.py` and
`models/weight.py` (main added `include_vision` and a `keep` filter and
restructured the function; #447 added `tp_shard`/`tp_config`). These three applied
with zero conflicts, and two of them are correctness fixes — see
[Missing upstream fixes](#missing-upstream-fixes-that-matter).

---

## Commit 1 — `dade0f8`: the TP>2 attention sharding bug

### Symptom

```
AssertionError: shape/dtype mismatch loading 'model.layers.3.self_attn.qkv_proj.weight':
  module expects (3328, 2560) torch.bfloat16, loader emitted (3584, 2560) torch.bfloat16
```

### Cause

Two components disagree about how to shard K and V when `num_key_value_heads < tp_size`.

`qwen4_exp/attention.py` built the projection with `LinearColParallelMerged`, which
shards each segment by **dividing its row count by `tp_size`**:

```
full QKV = Q-gated (2*24*256 = 12288) + K (2*256 = 512) + V (512) = 13312 rows
13312 / 4 = 3328        -> K and V get 512/4 = 128 rows each = HALF A HEAD
```

The weight loader shards by **head**, with replication
(`_partition`: when `world_size > size`, `start = rank // (world_size // size)`),
so ranks 0,1 receive KV head 0 and ranks 2,3 receive KV head 1:

```
Q 3072 + K 256 + V 256 = 3584   (14 head-slots x 256)
```

**The two agree only when `num_key_value_heads % tp_size == 0`.** This model has
`num_key_value_heads = 2`, so at TP=2 both paths give one head per rank and no
replication is needed either way — which is exactly why #447, tested only at
TP=2, never hit it. Any TP size that does not divide 2 exposes it.

### Fix

Switch to the head-aware `LinearQKVMerged`, which already does
`local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)`:

```python
self.qkv_proj = LinearQKVMerged(
    config.hidden_size,
    config.head_dim,
    num_qo_heads=2 * config.num_qo_heads,   # doubled: the output gate makes q_proj 2x wide
    num_kv_heads=config.num_kv_heads,
    has_bias=False,
    quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
)
```

`num_qo_heads` is doubled because `q_proj` carries the attention output gate
(`_qkv_global_split` uses `2 * num_qo_heads * head_dim`). Then
`local_osize = (12 + 2*1) * 256 = 3584` and `output_sizes = [3072, 256, 256]`,
matching both the loader and `_qkv_split`.

Also in this commit:

- **`_validate_owner_ep_config`**: dropped the hardcoded `or config.tp_info.size != 2`.
  The same-group invariant (`moe_ep_size == tp_info.size`) is kept, and the real
  constraint is delegated to `ExpertOwnership`, which validates
  `global_num_experts % world_size == 0`. 512 experts divides by 1/2/4/8/16 but not
  3/5/6/7, so that check is the correct gate. `moe/ownership.py` and
  `moe/offload_cache.py` are already fully `world_size`-parameterized — nothing else
  in the owner path assumes 2.
- **`layers/base.py`**: the bare `assert param.shape == item.shape and
  param.dtype == item.dtype` is replaced with an error naming the parameter and both
  shapes/dtypes. A bare assert that cannot say *which* of 48 layers × dozens of
  projections mismatched makes a shard bug nearly undiagnosable; this is what turned
  the above into a one-shot fix.
- **`tests/engine/test_owner_ep_config.py`**: `test_a_topology_other_than_tp2_ep2_is_still_rejected`
  asserted the restriction being removed. Replaced with
  `test_wider_same_group_topologies_pass` (2/4/8) and
  `test_a_mismatched_ep_and_tp_is_still_rejected`; `test_ftw_checkpoints_are_rejected`
  became `test_ftw_checkpoints_are_accepted`.

---

## Commit 2 — `af5a6a4`: FTW owner-slicing, 32 min boot → 2 min

### The problem

Owner EP rejected FTW checkpoints:

> `owner EP is not supported for FTW checkpoints: load_ftw_banks rebuilds
> [num_experts, ...] GLOBAL expert rows with no ownership filter, so the banks
> cannot bind to the owner-local geometry`

So every boot rebuilt all expert banks from raw safetensors. Measured: **7.5 MiB/s
per rank** with four ranks contending (loadavg 112, 4× ~700% CPU = 28 of 32 cores),
**~32 minutes**, with no persistent cache. That is the difference between a
benchmark and something you would actually run.

### The fix

`load_ftw_banks` gains `owner_slice=(start, count)` and reads only that contiguous
range of expert rows from each **per-layer** bank entry, producing `[count, ...]`
banks. `moe/expert_banks.py` passes
`(ownership.global_start, ownership.local_num_experts)` instead of raising, and the
`is_ftw_checkpoint` rejection is removed from `_validate_owner_ep_config`.

It reuses the flat layout's existing aligned-window carve, so it works whether or
not a bank's row size is a multiple of `ALIGN = 4096`: the read covers
`[align_down(off), align_up(off+len))` and the owner rows are viewed out at
`head_pad`.

**The alignment arithmetic is what makes this cheap.** For this checkpoint:

| bank | shape | dtype | row bytes | ÷4096 | sliceable |
|---|---|---|---|---|---|
| `gate_up_packed` | [512,1280,1280] | uint8 | 1,638,400 | **400.0** | yes |
| `gate_up_scale` | [512,1280,160] | fp8_e4m3 | 204,800 | **50.0** | yes |
| `down_packed` | [512,2560,320] | uint8 | 819,200 | **200.0** | yes |
| `down_scale` | [512,2560,40] | fp8_e4m3 | 102,400 | **25.0** | yes |
| `gate_up_global` | [512,1280] | fp16 | 2,560 | 0.625 | no → padded window |
| `down_global` | [512,2560] | fp16 | 5,120 | 1.25 | no → padded window |

Four of six banks — **99.72% of the bytes** — have row sizes that are exact
multiples of 4096, so `head_pad` is 0 and the window is exactly the owner range.
The two fp16 `*_global` banks fall back to a padded window costing **180 MiB of
over-read per rank** across 48 layers (~13 ms at 10 GiB/s).

The two layouts that genuinely cannot be sliced per owner are **rejected loudly**
rather than silently mis-sliced: the flat `[num_layers*num_experts, ...]` row
layout (one owner's rows are not contiguous there) and alpha vectors (flat, no
per-layer row split). Triton-packed NVFP4 banks have no alphas — those are
marlin's `gate_up_alpha`/`down_alpha` — so hitting that error means the checkpoint
was converted for a different `--quant-backend` than the server is using.

**Result: 15.86 GiB per rank in ~1.6 s at ~10 GiB/s, instead of ~32 min.**

### `--expert-bank-path` — load-bearing, not a convenience

Pointing `--model-path` at an FTW under TP>1 fails with:

```
shape/dtype mismatch loading 'model.embed_tokens.weight':
  module expects (62080, 2560), loader emitted (248320, 2560)      # 248320 / 4 = 62080
```

An FTW stores dense weights **post-fusion** (`qkv_proj`, `gate_up_proj`), but
`shard_qwen4_exp_dense_tensor` operates on **raw unfused** tensors so that fused
buffers keep head boundaries. Sharding a fused tensor at load would mean
re-deriving every fusion's segment layout — precisely the change that produces
fluent-but-wrong output instead of an error. And `ft checkpoint` has no TP flag, so
every FTW in existence has global dense weights.

So this commit adds `--expert-bank-path`, splitting the two sources:

```
--model-path        <HF safetensors dir>   dense: reader shards RAW tensors, 206 shards in ~2 s
--expert-bank-path  <FTW dir>              experts: owner-sliced banks, ~1.6 s/rank
```

Both halves reuse an already-verified route, so no new numerics are introduced.

---

## Verification

### Bank slicing is byte-exact

Every rank's sliced banks compared against the corresponding rows of the unsliced
load, dtype-agnostically (raw byte comparison, so fp8/uint8/fp16 all work):

```
banks compared byte-for-byte : 1152   (4 ranks x 6 roles x 48 layers)
failures                     : 0
```

15.86 GiB per rank at ~10 GiB/s, 1.58–1.66 s. This isolates the fast-load change
from the TP change: if the aligned-window carve were off by anything, this fails.

### Unit tests

`418 passed, 56 skipped` across `tests/engine`, `tests/moe`,
`tests/models/qwen4_exp`, `tests/models/test_weight_tp_shard.py`.

**4 failures are pre-existing in #447 itself**, not introduced here:
`tests/moe/test_fused_copy.py::test_fused_copy_matches_per_bank[0,1,4,8]` →
`offload_cache.py:1106 AssertionError: no staged misses (ensure_experts/materialize_layer first)`.
Confirmed by running the bare base `81d034d` in a throwaway worktree.

### End-to-end acceptance

| gate | result |
|---|---|
| greedy determinism (temperature 0, 5 repeats × 3 prompts) | **DETERMINISTIC** — 1 distinct output each |
| tool calling | **PASS** — `finish_reason: tool_calls`, `get_weather({"city":"Berlin","units":"celsius"})` |
| streamed tool-call deltas reassembling across SSE chunks | **PASS** |
| long-context prefill | **PASS** — 6,269 / 27,969 / **80,669** tokens, 919–1475 tok/s |
| decode throughput | 64.5 / 66.4 / 67.3 tok/s |
| multi-turn GDN save→evict→restore | **PASS on content** (see caveat) |

The multi-turn gate pushed **421,200 distinct tokens** (random word soup, no shared
prefix) through 54 requests to genuinely overflow the 262144-token radix cache —
verified by `cached_tokens` falling from 1408 to `None`. After that, the same greedy
request reproduced its content **byte-identically** and retrieved all three facts
buried in turn 1. Warm-vs-warm repeats were identical too.

### Missing upstream fixes that matter

Before cherry-picking `cc1f5c2`, output was **non-deterministic at temperature 0**
(three identical greedy runs gave two different results). That is #471
"preserve greedy sampling in mixed batches", which #447's base predates. Also
picked up `e0886cc` "compute shared experts before in-place routed experts" (a MoE
correctness fix) and `f7dbab7` "load ModelOpt exports that ship no input_scale"
(NVFP4/modelopt-relevant).

---

## Known gaps — please read before trusting this

**Not verified:**

1. **Greedy equivalence against a TP=1 run of this same branch.** Note that bitwise
   identity is *not* expected across different TP sizes in any case — all-reduce
   changes floating-point summation order — so this needs judging on consistency and
   correctness, not exact match.
2. **A full 262144-token prefill.** 80,669 tokens is the largest probed.
3. **The cold-vs-warm reasoning divergence.** Same prompt at temperature 0: with a
   warm prefix cache the model produced 585 chars of `reasoning_content`
   (`completion_tokens` 177); after eviction, 393 chars (`completion_tokens` 127).
   **Final content was byte-identical and all facts correct** in both cases, so
   greedy determinism holds *within* a cache state but not across cold/warm. Plausible
   cause: a cache hit restores GDN recurrent state from a checkpoint rather than
   recomputing it, and the restored state is not bit-identical, so small logit
   differences flip chain-of-thought tokens while the answer still converges.
   **Not attributed** — there is no cold-vs-warm comparison for TP=1 to rule out that
   this is inherent to FreeToken's GDN prefix caching rather than specific to owner EP.

**Landmines:**

- **`gdn.py` carries the same bug commit 1 fixed.** It sizes heads with
  `div_even(..., allow_replicate=True)` but builds `in_proj_qkvz` / `in_proj_ba` with
  row-dividing `LinearColParallelMerged`. Harmless at TP=4 (`linear_num_key_heads` 16
  and `linear_num_value_heads` 48 both divide by 4, so row division and head division
  coincide) but **wrong for any `tp_size > 16`**, where replication engages.
- **`--ple-backend disk` is mandatory at TP>1.** The PLE n-gram table has no rank
  awareness, so it is replicated per rank; `pinned` would want 4 × 47.7 = 191 GiB of
  page-locked host RAM. `disk` maps the shards in place so all ranks share one
  page-cache copy. Measured: **no performance difference** (decode 66.7 vs 66.7,
  65.2 vs 65.4 tok/s; cold prefill 1057 vs 1042), so the 191 GiB buys nothing.
- **`--kv-reserve-tokens` must equal `--num-tokens`.** `--moe-cache-auto` sizes the
  expert cache against the *reserve* (default 8192 ≈ 192 MiB), not against
  `--num-tokens`. Without this the expert cache consumes the VRAM and
  `create_kv_pool` dies: `Tried to allocate 6.00 GiB ... 2.53 GiB is free`.
- **`--memory-ratio 0.85` + `--max-prefill-length 2048`, do not raise.** At 0.95 the
  engine boots, answers short prompts, then hard-crashes the backend worker on the
  first real prefill in the GDN/FLA transient
  (`gdn.py → chunk_gated_delta_rule → l2norm.py torch.empty_like`). It does **not**
  degrade gracefully: `Backend worker is gone and cannot be restarted`. Watch for
  `Free GPU memory after capturing CUDA graphs` ≥ ~3 GiB (this config: 3.27 GiB).
  A config that passes a 58-token smoke test can still fail here — always validate
  with a multi-thousand-token prefill.
- **Owner EP requires `--moe-strategy offload`** and rejects `--moe-cache-rate` and
  `--moe-cpu-layers` (`_decode_owner` is selected before the `is_cpu_layer` branch,
  so the latter would be accepted and silently ignored). At a 67% hit rate
  offload-only costs little; at TP=1's 4.5% it would be crippling, which is why
  TP=1 needs `hybrid`.
- **Each instance binds `port` AND `port+1`** (`server/args.py`:
  `tcp://127.0.0.1:{server_port+1}`) for ZMQ IPC. Keep both free.
- **`--quant-backend moe.nvfp4=triton`.** Marlin declares `cpu_ok=False`, and its
  donor (vLLM) is not a dependency here. The FTW banks are physically packed per
  kernel — triton's are identifiable by `gate_up_global`/`down_global` (fp16) and the
  absence of `*_alpha` — so changing backend means reconverting.

---

## Running it

Requires the FTW conversion (once) and `ft bench bw` per GPU.

```bash
# 1. Convert to FTW with the triton bank layout (banks are packed per kernel).
#    The FTW fingerprint hashes compute capability, not GPU index, so it is
#    portable across identical cards. ~275 s for 126 GiB.
ft checkpoint --model <hf-safetensors-dir> --out <ftw-dir> \
    --dtype bfloat16 --moe-backend offload \
    --quant-backend moe.nvfp4=triton --shard-gib 8 --gpu 0

# 2. Calibrate the bandwidth profile for EVERY GPU you will use. The profile is
#    keyed on GPU UUID, so it is not shared between cards even when they are
#    identical: four RTX 3090 in one host measured 4.12x / 4.08x / 4.45x / 4.33x.
ft bench bw --gpu 0 --dtype nvfp4,bf16     # ...and 1, 2, 3

# 3. Serve.
ft serve \
    --model-path        <hf-safetensors-dir> \
    --expert-bank-path  <ftw-dir> \
    --served-model-name qwen3.8-27b \
    --gpu 0,1,2,3 --tensor-parallel-size 4 --moe-ep-size 4 \
    --host 0.0.0.0 --port 8081 \
    --dtype bfloat16 \
    --max-seq-len-override 262144 --num-tokens 262144 --kv-reserve-tokens 262144 \
    --memory-ratio 0.85 --max-prefill-length 2048 \
    --moe-strategy offload --quant-backend moe.nvfp4=triton --ple-backend disk \
    --max-running-requests 2 --cuda-graph-max-bs 2 \
    --enable-special-token-ckpt --moe-prefill-hit-d2d --enable-cache-report \
    --sampling-defaults model --tool-call-parser auto --reasoning-parser auto
```

`--gpu` entries are positional TP ranks: entry *i* is rank *i*.
`--tool-call-parser auto` resolves to `qwen3_coder` and `--reasoning-parser auto` to
`qwen3` for this family.

Boot confirmation lines to look for:

```
--gpu 0,1,2,3 -> GPU-…, GPU-…, GPU-…, GPU-…
MoE experts: nvfp4 via triton
expert banks: FTW fast path, owner-local rows [0, 128) of 512
--moe-cache-auto resolved moe_cache_size=4124 num_pages=4096 (prefill_overlap=True)
Allocating 262144 tokens for KV cache, K + V = 3.19 GiB          # per rank
Free GPU memory after capturing CUDA graphs: 3.27 GiB
API server is ready to serve on 0.0.0.0:8081
```

### VRAM budget per rank

| | GiB |
|---|---|
| owned experts in host RAM (128 of 512, 48 layers) | 15.83 (host, not VRAM) |
| PLE n-gram table, FP8 | 47.72 (host, shared via page cache) |
| dense bf16 shard + replicated norms/routers/hyper-connections/QSA indexer | ~6 |
| KV cache, 262144 tokens, fp8-free bf16 | 3.19 |
| expert cache (4124 slots) | ~10 |
| CUDA graphs + activations | ~0.4 |
| **total, of 23.56 usable** | **~20.6** |

### Flags measured and rejected

`--moe-prefill-hit-d2d` is **kept**: warm prefill 8655 → 14443 tok/s (+67%), TTFT at
4k context 0.72 → 0.44 s (−39%), short-prompt TTFT 0.71 → 0.50 s, for 1.8% of decode
(65.2 → 64.0). Cold prefill is unchanged (1057 → 1080) since there are no cache hits
to copy device-side. Needs CUDA ≥ 13.

`--ple-backend pinned` is **rejected**: identical performance, 191 GiB of
non-reclaimable host RAM.

`--memory-ratio` above 0.85 is **not recommended**: it would drop free VRAM to
~2.5 GiB, and the GDN prefill transient OOMs at 2.4 GiB.

---

## Why EP and not TP for the experts

Owner-local EP gives each rank **128 whole experts** rather than a 160-wide slice of
all 512. Two consequences that made this tractable:

1. No NVFP4 kernel needs to become TP-aware. Every nvfp4 MoE kernel upstream
   declares `tp_ok=False`, and marlin additionally declares `cpu_ok=False`. EP sidesteps
   all of it — each rank runs the standard unsharded kernel on the experts it owns.
2. The expert pool per rank shrinks to 15.83 GiB, which nearly fits the ~10 GiB of
   VRAM left after dense weights and KV. That is the 4.5% → 67% cache-hit jump, and
   it is where the speed actually comes from.

The arithmetic for TP-sharding experts instead also works out — `moe_intermediate_size`
640 / 4 = 160, and the `local_intermediate % (2 * GROUP)` guard with `GROUP = 16`
needs 160 % 32 == 0 — so that route is viable too, just more invasive.

---

## MTP / speculative decoding: assessed, not implemented

This checkpoint ships an MTP head that `models/qwen4_exp/weight.py` discards
(`if raw_name.startswith(("mtp.", "model.visual.", "visual."))`). See upstream
issue [#421](https://github.com/FlashML-org/FreeToken/issues/421).

Measured from the safetensors headers: **31 `mtp.*` tensors, all bf16, 4.856 GiB**
(the quant `ignore` list covers `mtp.*`, so the head is unquantized). It is
structurally different from the `qwen3_5_moe` nextn head:

- `mtp.layers.0.mlp.experts.gate_up_proj` [512,1280,2560] = 3200 MiB and
  `down_proj` [512,2560,640] = 1600 MiB — pre-stacked, all 512 experts. This does
  **not** fit in 3.27 GiB/rank of free VRAM; only TP/EP-sharded to 1.2 GiB/rank.
- `self_attn.indexer.index_qk_proj` [640,2560] — QSA sparse attention.
- **Three** hyper-connection blocks; `pre_fc_norm_hidden` is [10240] = `hc_count` 4 ×
  2560, i.e. it norms the hyper-connection-widened hidden state.
- **Two** matrices, `fc_embedding` and `fc_hidden`, both [2560,2560] — not a single
  `fc` [H, 2H].
- `text_config.mtp.layer_types = ['full_attention']`, so the head layer is never GDN
  and needs no recurrent state of its own.

Upstream PR #69 (`DSpark`) does not supply the machinery: dSpark self-speculates from
the **tail N MoE layers of the target**, not a trained head, and its `rollback.py`
snapshots DSV4's compressor carry and Lightning Indexer ring blocks — a different
hazard from rolling back **36 GDN layers' fp32 recurrent state** on partially rejected
drafts. Nothing in FreeToken does the latter. The suggested hook is that
`cache_type='hybrid_radix'` already snapshots GDN state at chunk boundaries.

Expected payoff here is lower than the ~1.80× reported for a PCIe-fetch-bound single
GPU. At 15.2 ms/token, expert misses are only ~4.1 ms, so ~11 ms is per-forward
overhead (likely the 96 all-reduces, ~5 KB each, latency-bound with no NVLink). That
overhead amortises across verified tokens too: modelling fixed ≈ 11 ms and per-token
≈ 4.2 ms, draft-1 at α = 0.894 gives ≈ 10.2 ms/token → **~98 tok/s, ~1.48×**. α for
this checkpoint is unknown and is the number that decides whether it is worth building.

---

## Process gotchas worth recording

- **`readlink /proc/PID/exe` resolves a venv symlink to the real interpreter**
  (here `/usr/bin/python3.12`), so a `*/.venv/*` exe-pattern kill loop matches
  nothing and silently does nothing. The old server kept holding the port and the
  relaunch died on `address already in use`. Kill by explicit PID, or use
  `nvidia-smi --query-compute-apps=pid --format=csv,noheader`.
- The four TP ranks are `multiprocessing` children that **survive killing the
  parent**. Kill all of them and wait for VRAM to actually drop before relaunching,
  or the new instance OOMs against the old one's residue.
- Never pattern-match `ps` output when the pattern text also appears in your own
  command line — including inside an `echo`. It SIGKILLs the shell doing the killing.
- Backgrounding a whole `A && B && launch &` chain from a non-interactive caller can
  get the group reaped before it reaches the launch if `A` is slow. Launch the server
  on its own, detached (`setsid nohup ... &`).

---

## License and attribution

FreeToken is Apache-2.0 (© the FreeToken authors); this fork and the two commits
here are under the same license. Cherry-picked commits retain their original
authors. Nothing here is affiliated with or endorsed by the FreeToken authors.

Model: `Qwen3.8-Flash-Next-NVFP4`. Hardware for all measurements: 4× NVIDIA RTX 3090
24 GiB (sm_86, SYS interconnect, no NVLink), AMD EPYC 7543 32c/32t, 503 GiB RAM,
model on RAID0 of two NVMe (~14 GB/s), CUDA 13.3, driver r610.43.02.
