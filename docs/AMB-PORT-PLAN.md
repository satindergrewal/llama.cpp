# -amb (attention-max-batch) port plan + the measure-first caveat

Branch `fable-amb-port` off upstream master (91f8c9c5f). Goal: cap the compute-buffer wall
(the real GLM context ceiling, per RESEARCH-LLAMACPP-KV-INVENTORY) the way ik_llama.cpp's `-amb`
does. This doc is the investigation before the code, per the room's measure-first law.

## What ik's -amb actually does (read from /mnt/nvme0/ik_llama.cpp)

- Flag: `-amb/--attention-max-batch N` (common.cpp:1923); clamps N<128 to 128.
- Threads through: params.attn_max_batch -> mparams.amb -> cparams.attn_max_batch.
- The mechanism (src/llama.cpp:3763, inside `mla_attn==3 && mla.wv_b`): when the f32 KV/wv_b
  scratch exceeds `amb` MiB, it splits the HEAD dimension into the largest divisor chunk whose
  per-chunk scratch fits `amb`, so `compute[il]` is bounded by `2 * wv_b.ne[1] * n_max_head *
  max_ctx * 4B` instead of the full-head size. It caps the worst-case compute buffer used for
  memory FITTING, and the runtime attention loop honors the same chunking.

## THE CAVEAT (must measure before porting)

ik's -amb targets two buffers: (a) the non-FA KQ matrix `[n_kv, n_ubatch, n_head]`, and (b) the
MLA `wv_b` absorbed-path intermediate. WE RUN FA-ON, so (a) never materializes for us. The GLM
compute buffer we hit (22.8GB at 128K, fixed by -ub 256) was the FA path's buffer, whose
dominant terms after mainline #25370 (f16 KQ mask) are the mask `[n_kv, n_ubatch]` and per-ubatch
activations. So:

- **-amb's head-chunking of (a) does NOT help an FA-on serve.** -ub already scales the FA buffer.
- **-amb's chunking of (b), the MLA wv_b intermediate, MAY help GLM even with FA**, because MLA's
  absorbed path produces a large `[wv_b.ne1 x n_head x ctx]` intermediate independent of the FA
  KQ matrix. THIS is the piece worth measuring and possibly porting.

## Measure-first step (do before writing the port)

On a GLM serve, dump per-op compute-buffer allocation at 64K/128K with FA on and -ub 512, and
identify whether the MLA wv_b intermediate is a top allocator. If yes: port the head-chunk of
just that intermediate to mainline's MLA graph (build_attn_mha / the deepseek MLA path in
llama-graph.cpp). If no (FA mask + activations dominate): -amb is the wrong lever for our FA-on
regime, and the real compute-buffer win is elsewhere (smaller -ub, or the #26038-class
over-reservation fixes). Report the measurement either way; do not port a mechanism aimed at a
buffer we do not materialize.

## Why this branch exists now
Autonomy grant (Satinder #1109): experiment freely on own branches, only merge-to-fleet gated.
This is the zero-GPU-contention first pickup while both box cards run approved work. The
measurement needs a GLM serve = a GPU window, so it queues behind v0.3/the DSA A/B; the port
plan is committed now so the work is resumable and the caveat is on record before any hours burn.
