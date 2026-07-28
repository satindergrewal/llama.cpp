# L2: DSA indexer harvest port plan (the highest-mission-fit lane)

Branch `fable-dsa-harvest` off upstream master (91f8c9c5f). Goal: make GLM 5.2 decode harvest
O(k=2048) attention instead of O(n) by wiring the model's own DSA lightning indexer into a
gather-attention kernel. Per RESEARCH-DSA-INDEXER: 7.7-31x KV traffic / 13-31x FLOP at 128K.

## State of the two trees (read 2026-07-28)

**Mainline (our fork base) HAS:**
- `GGML_OP_LIGHTNING_INDEXER` + CUDA impl: `ggml/src/ggml-cuda/lightning-indexer.cu`
  (WMMA kernel L19, vec kernel L244, `ggml_cuda_lightning_indexer` L400). From #24231.
- GLM 5.2 correctness graph (#25407, merged Jul 24): indexer + top-k, but MASK-SHAPED - it
  unmasks the top-2048 inside a full-size KQ mask then runs normal FA over ALL keys. Pays
  indexer cost, harvests NO O(k) compute. PR author reports decode SLOWER than dense.

**Mainline LACKS (open PR #25917, fairydreaming):** sparse KV indices in the MMA FA kernel =
the actual O(k) gather. This is the harvest. It is unmerged.

**ik_llama.cpp HAS BOTH, and the fused gather:**
- `ggml/src/ggml-cuda/indexer_topk.cu` (279 lines): `ggml_cuda_op_indexer_topk` +
  `ggml_cuda_op_indexer_mask`.
- `ggml/src/ggml-cuda/dsa_attn.cu` (312 lines): `ggml_cuda_dsa_attn_ext(ctx, dst) -> bool`
  (returns whether it handled the op = the gather-attention that reads only top-k KV).
- Runtime flags: `-dsa` (enable), `-dsatk` (top-k override), `-fidx` (fused indexer top-k).
- ~2 weeks ahead of mainline; MIT-licensed (Iwan Kawrakow).

## The port (fork-only, per Satinder's boundary)

Two viable paths, both keep everything on our fork:

**Path A (port ik's gather):** bring `dsa_attn.cu` + `indexer_topk.cu` into the fleet build's
ggml-cuda, adapt the tensor/op plumbing to mainline's graph (mainline already emits the indexer
op; the graph builder in llama-graph.cpp / the glm-dsa model file must emit a
`dsa_attn`/gather op instead of the mask+full-FA path). ~591 lines CUDA + graph wiring +
dispatch registration in ggml-cuda.cu. Mac-Metal has no path (no Metal DSA kernel in ik either)
so this is CUDA-only, box-validated.

**Path B (adopt mainline #25917's approach):** implement the same sparse-KV-index MMA FA path
fairydreaming is building, on our fork, so it lands when we want rather than when upstream
merges. More work, closer to mainline's eventual shape.

Recommendation: **Path A** - ik's is a working, tested, standalone unit; port it, measure the
O(k) decode win vs the current mask-shaped #25407 path AND vs dense, on GLM 5.2 at 64K/128K.

## Gates (measure-first)
1. Before porting: confirm our fleet build's #25407 correctness path is coherent at 64K and
   is (per the inverted-risk hypothesis) parity-or-BETTER than dense - the correctness A/B.
   This validates the indexer is doing the right thing before we optimize its speed.
2. After porting: measure decode t/s and KV traffic at 64K/128K, gather vs mask-shaped vs
   dense. Success = decode faster than dense (the current path is slower).
3. Quality: NIAH + loop-rate on the gather path (top-k selection must not drop needles).

All three gates need both box cards (169G GLM) = a GPU window. Study + port code is CPU/Mac
work, done now on this branch; the measurements queue on a window behind v0.3.

Autonomy: experiment freely on this branch (Satinder #1109); merge-to-fleet is the only gate.

---

# PORT DESIGN (read from ik source 2026-07-28, baseline built)

## Baseline
Worktree `/mnt/nvme0/llama.cpp-dsaport` (branch `fable-dsa-harvest-box`, off fleet ffee9f47e),
CUDA build clean: DBRC=0, llama-server + llama-bench at 16:54. Opus's fleet tree untouched
(separate worktree by construction).

## ik's gather is a SELF-CONTAINED PIPELINE, not an FA modification

`ggml_cuda_dsa_attn_ext(ctx, dst) -> bool` (returns "I handled this op"), built from four small
kernels plus its own softmax:
1. `k_prepare_mask` - gathers the mask rows for the selected indices into a compact
   `[nidx x rows]` buffer (the full-size mask never drives the math).
2. `k_prepare_one_batch_kv` - gathers ONLY the selected K (and V) rows into a compact f16
   buffer. **This is the O(k) harvest: it reads top-k rows instead of streaming all of KV.**
3. `k_prepare_one_batch_q` - f32->f16 Q staging.
4. `soft_max_f16_simple` + `k_copy_dst` - softmax over the compact scores, f16->f32 out.

It **sidesteps FlashAttention entirely** for DSA layers rather than teaching FA about sparsity.
That is exactly why it ports as a unit, and why it is lower-risk than mainline's #25917 approach
(which threads sparse indices through the MMA FA kernel).

## THE INTEGRATION SEAM (the one thing that needs graph work)

ik's entry reads **`dst->src[5]` = indexer** (the top-k index tensor), i.e. their FA node carries
a 6th source. Mainline's `ggml_flash_attn_ext` node has 5 (Q,K,V,mask,sinks). So the port needs
either:
- **(A)** a new op `GGML_OP_DSA_ATTN` emitted by the glm-dsa/deepseek32 graph builders when the
  indexer top-k tensor exists, with the gather as its CUDA impl (cleanest, no FA disturbance); or
- **(B)** extend the FA node with an optional 6th src and dispatch to the gather when present
  (closer to ik, touches shared FA plumbing).
Recommendation: **(A)** - new op keeps FA untouched, matches mainline's own style of adding
`GGML_OP_LIGHTNING_INDEXER` as a discrete op, and the CPU fallback can simply be the existing
mask-shaped path.

## Guards ik enforces (must replicate, they define the fast path's domain)
- no sinks; requires Q,K,V,mask,indexer all present
- `indexer->ne[0] % 256 == 0` (top-k multiple of 256; ours is 2048 OK)
- `K->ne[1] >= 4*indexer->ne[0]` (only worth it when context >> top-k: at 2048 top-k this means
  ctx >= 8192 - below that, fall back to dense, which matches the "identity below 2048" fact)
- K/V/mask f16, Q f32; single-batch (ne[2]/ne[3] == 1)
Anything failing a guard returns false -> existing path runs. **Fail-safe by construction.**

## Order of work
1. ~~baseline build~~ DONE
2. Port the four kernels + softmax into `ggml/src/ggml-cuda/dsa-attn.cu` (new file), adapted to
   mainline's ggml-cuda conventions (ctx, stream, pool allocs).
3. Add `GGML_OP_DSA_ATTN` (option A) + dispatch registration; graph emit in `src/models/glm-dsa.cpp`
   gated on the indexer tensor + the guards above.
4. Compile-gate (CPU).
5. Measure on a GPU window: gather vs mask-shaped(#25407) vs dense, decode t/s + KV traffic at
   64K/128K, plus NIAH + loop-rate for quality. Success = faster than dense (today's path is slower).
