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
