#include "pagedattn.cuh"
#include "mma.cuh"
#include "cp-async.cuh"

__device__ __forceinline__ float block_reduce_sum_full(float val, float * __restrict__ smem, int tid, int head_dim) {
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = (head_dim + 31) >> 5;

    // warp-level reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    // Each warp's lane 0 depositits partial sum into shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the per-warp partial
    float warp_val = (tid < n_warps) ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val += __shfl_down_sync(0xffffffffu, warp_val, offset);
        }
        if (lane == 0) {
            smem[0] = warp_val;
        }
    }
    __syncthreads();
    return smem[0];  // this will be identical in every thread
}

__global__ void paged_attention_write_kernel(const float * __restrict__ k_new,  // [batch_size, n_heads_kv, head_dim]
                                             const float * __restrict__ v_new,  // [batch_size, n_heads_kv, head_dim]
                                             half * __restrict__ kv_cache,      // The paged cache
                                             const int * __restrict__ write_slots,  // Global slot index for each token
                                             const int * __restrict__ batch_offsets,
                                             const int * __restrict__ batch_lens,
                                             const size_t stride_token,  // Elements between tokens in a block (nb1)
                                             const size_t stride_head,   // Elements between heads (nb2)
                                             const size_t stride_block,  // Elements between physical blocks (nb3)
                                             const int    n_heads_kv,
                                             const int    block_size) {
    const int head_idx = blockIdx.x;   // 0 to n_heads_kv - 1
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;  // 0 to head_dim - 1
    const int head_dim = blockDim.x;

    const int seq_start  = batch_offsets[seq_idx];
    const int num_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_tokens; ++i) {
        const int token_batch_idx = seq_start + i;
        const int target_slot     = write_slots[token_batch_idx];

        // Map slot to block and internal offset
        const int block_id       = target_slot / block_size;
        const int token_in_block = target_slot % block_size;

        // K is at head_idx, V is at n_heads_kv + head_idx
        const size_t k_cache_idx = (size_t) block_id * stride_block + (size_t) head_idx * stride_head +
                                   (size_t) token_in_block * stride_token + tid;
        const size_t v_cache_idx = (size_t) block_id * stride_block + (size_t) (n_heads_kv + head_idx) * stride_head +
                                   (size_t) token_in_block * stride_token + tid;

        // Input offset: [token][head][dim]
        const size_t input_off = (size_t) token_batch_idx * n_heads_kv * head_dim + (size_t) head_idx * head_dim + tid;

        kv_cache[k_cache_idx] = __float2half(k_new[input_off]);
        kv_cache[v_cache_idx] = __float2half(v_new[input_off]);
    }
}

// Reference kernel (kept for head_dim > 128): one block-wide reduction with barriers PER
// KV TOKEN, serial over the context. Correct but pathological at scale -- an 8K-context
// decode step measured ~40 s/step on Blackwell (Q4 fork-cost gate, 2026-08-04). The
// warp-parallel kernel below replaces it for head_dim 64/128.
__global__ void paged_attention_decode_kernel_ref(const float * __restrict__ q,
                                              const half * __restrict__ kv_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t stride_token,
                                              const size_t stride_head,
                                              const size_t stride_block,
                                              const int    n_heads_kv,
                                              const int    block_size,
                                              const int    max_blocks,
                                              const float  scale,
                                              // banded (3b): rel_logits [rel_extent, n_heads, n_tokens] F32 or nullptr;
                                              // rel_dist = q_pos - token is LOGICAL, block scattering cannot affect it
                                              const float * __restrict__ rel,
                                              const int64_t rel_extent,
                                              const int64_t visibility_window,
                                              float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;

    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];

    for (int i = 0; i < num_new_tokens; i++) {
        const int token_batch_idx = seq_start + i;

        float q_val = q[(size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid] * scale;

        float qk_max  = -FLT_MAX;
        float exp_sum = 0.0f;
        float acc     = 0.0f;

        const int ctx_len    = context_lens[seq_idx];
        const int q_pos      = (ctx_len - num_new_tokens) + i;
        const int num_blocks = (q_pos / block_size) + 1;

        for (int bid = 0; bid < num_blocks; bid++) {
            const int physical_block = block_table[seq_idx * max_blocks + bid];
            const int start_token    = bid * block_size;
            const int end_token      = min(start_token + block_size, q_pos + 1);

            for (int token = start_token; token < end_token; ++token) {
                const int64_t rel_dist = (int64_t) q_pos - token; // >= 0 (end_token caps at q_pos + 1)

                // analytic band: skip invisible cells before paying for the cache loads.
                // NB: uniform across the block (all threads share q_pos/token), so no
                // divergence around the __syncthreads in the reduction below.
                if (visibility_window > 0 && rel_dist >= visibility_window) {
                    continue;
                }

                const int token_in_block = token % block_size;

                const size_t k_idx =
                    tid + token_in_block * stride_token + kv_head_idx * stride_head + physical_block * stride_block;

                const size_t v_idx = tid + token_in_block * stride_token + (n_heads_kv + kv_head_idx) * stride_head +
                                     physical_block * stride_block;

                float k_val = __half2float(kv_cache[k_idx]);
                float v_val = __half2float(kv_cache[v_idx]);

                // Calculate full dot product and return the same scalar in every thread
                float qk = block_reduce_sum_full(q_val * k_val, smem, tid, head_dim);

                // banded relative-position bias; own rel_extent gate, independent of the
                // visibility window. Same value in every thread (uniform indices).
                if (rel != nullptr && rel_dist < rel_extent) {
                    qk += rel[((size_t) token_batch_idx * n_heads + head_idx) * rel_extent + rel_dist];
                }

                // Online softmax update
                const float qk_max_new = fmaxf(qk_max, qk);
                const float exp_old    = __expf(qk_max - qk_max_new);
                const float exp_new    = __expf(qk - qk_max_new);

                exp_sum = exp_sum * exp_old + exp_new;
                acc     = acc * exp_old + exp_new * v_val;
                qk_max  = qk_max_new;
            }
        }

        const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;

        out[out_idx] = acc / (exp_sum + 1e-6f);
    }
}

// Warp-parallel decode (flash-decode shape): context tokens are strided across warps, each
// warp keeps its own online-softmax state (m, l, per-lane acc slices), the per-token dot is
// a warp shuffle reduction, and the warps merge ONCE per query token via log-sum-exp in
// shared memory. No __syncthreads inside the context loop -- the reference kernel's
// per-token block barrier serialized the whole context and made big-context decode
// unusable. Same visibility-window and rel-bias semantics as the reference.
// Contract: blockDim.x == head_dim, head_dim in {64, 128} (acc slices fixed at <= 4).
__global__ void paged_attention_decode_kernel(const float * __restrict__ q,
                                              const half * __restrict__ kv_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t stride_token,
                                              const size_t stride_head,
                                              const size_t stride_block,
                                              const int    n_heads_kv,
                                              const int    block_size,
                                              const int    max_blocks,
                                              const float  scale,
                                              const float * __restrict__ rel,
                                              const int64_t rel_extent,
                                              const int64_t visibility_window,
                                              float * __restrict__ out,
                                              // M7 split-K (n_splits == 1 => original path):
                                              const int n_splits,
                                              float * __restrict__ out_m,
                                              float * __restrict__ out_l) {
    extern __shared__ float smem[];

    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int split_idx = (n_splits > 1) ? (int) blockIdx.z : 0;
    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp_id   = tid >> 5;

    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;
    const int n_warps  = head_dim >> 5;       // 2 (hd 64) or 4 (hd 128)
    const int dpl      = head_dim >> 5;       // dims per lane (lane + 32*d)

    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];

    // smem layout: q_s[head_dim] | warp_m[n_warps] | warp_l[n_warps] | warp_acc[n_warps*head_dim]
    float * q_s      = smem;
    float * warp_m   = q_s + head_dim;
    float * warp_l   = warp_m + n_warps;
    float * warp_acc = warp_l + n_warps;

    // the query-token axis lives on the GRID (blockIdx.z), not in a serial loop: a 512-token
    // prefill chunk previously ran 512 full context scans back-to-back in one block, which
    // made chunked prefill O(minutes) at 8K context (Q4 gate). One block = one query token.
    {
        // split-K borrows blockIdx.z, which is free during decode (num_new_tokens == 1);
        // with n_splits == 1 it stays the query index as before
        const int i = (n_splits > 1) ? 0 : (int) blockIdx.z;
        if (i >= num_new_tokens) {
            return;
        }
        const int token_batch_idx = seq_start + i;

        q_s[tid] = q[(size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid] * scale;
        __syncthreads();

        const int ctx_len = context_lens[seq_idx];
        const int q_pos   = (ctx_len - num_new_tokens) + i;
        const int n_tok   = q_pos + 1;
        // analytic band: everything older than the window is invisible (matches the
        // reference's rel_dist >= visibility_window skip)
        const int lo = (visibility_window > 0) ? max(0, (int) (q_pos - visibility_window + 1)) : 0;

        float m_i = -FLT_MAX;
        float l_i = 0.0f;
        float acc_i[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

        // M7 split-K: when the launcher splits the context across the grid, each block
        // owns a contiguous slice [split_lo, split_hi) and emits a PARTIAL (m, l, acc);
        // the combine kernel merges partials by log-sum-exp. n_splits == 1 is the
        // original whole-context path, bit-for-bit (same warp striding, same order).
        int split_lo = lo, split_hi = n_tok;
        if (n_splits > 1) {
            const int span = (n_tok - lo + n_splits - 1) / n_splits;
            split_lo = lo + split_idx * span;
            split_hi = min(n_tok, split_lo + span);
            if (split_lo >= split_hi) {   // empty slice: emit a null partial and stop
                const size_t pidx = ((size_t) token_batch_idx * n_heads + head_idx) * n_splits + split_idx;
                if (tid == 0) {
                    out_m[pidx] = -FLT_MAX;
                    out_l[pidx] = 0.0f;
                }
                out[pidx * head_dim + tid] = 0.0f;
                return;
            }
        }

        for (int token = split_lo + warp_id; token < split_hi; token += n_warps) {
            const int    bid            = token / block_size;
            const int    physical_block = block_table[seq_idx * max_blocks + bid];
            const int    token_in_block = token % block_size;
            const size_t base   = (size_t) token_in_block * stride_token + (size_t) physical_block * stride_block;
            const size_t k_base = base + (size_t) kv_head_idx * stride_head;
            const size_t v_base = base + (size_t) (n_heads_kv + kv_head_idx) * stride_head;

            float part = 0.0f;
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                if (d < dpl) {
                    const int dim = lane + (d << 5);
                    part += q_s[dim] * __half2float(kv_cache[k_base + dim]);
                }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                part += __shfl_down_sync(0xffffffffu, part, offset);
            }
            float qk = __shfl_sync(0xffffffffu, part, 0);

            const int64_t rel_dist = (int64_t) q_pos - token;
            if (rel != nullptr && rel_dist < rel_extent) {
                qk += rel[((size_t) token_batch_idx * n_heads + head_idx) * rel_extent + rel_dist];
            }

            const float m_new = fmaxf(m_i, qk);
            const float e_old = __expf(m_i - m_new);
            const float p     = __expf(qk - m_new);
            l_i = l_i * e_old + p;
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                if (d < dpl) {
                    const int dim = lane + (d << 5);
                    acc_i[d] = acc_i[d] * e_old + p * __half2float(kv_cache[v_base + dim]);
                }
            }
            m_i = m_new;
        }

        if (lane == 0) {
            warp_m[warp_id] = m_i;
            warp_l[warp_id] = l_i;
        }
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            if (d < dpl) {
                warp_acc[warp_id * head_dim + lane + (d << 5)] = acc_i[d];
            }
        }
        __syncthreads();

        // cross-warp log-sum-exp merge; a warp that saw no tokens has m = -FLT_MAX and
        // contributes expf(-inf) = 0
        float m_tot = -FLT_MAX;
        for (int w = 0; w < n_warps; ++w) {
            m_tot = fmaxf(m_tot, warp_m[w]);
        }
        float l_tot   = 0.0f;
        float out_acc = 0.0f;
        for (int w = 0; w < n_warps; ++w) {
            const float f = __expf(warp_m[w] - m_tot);
            l_tot   += warp_l[w] * f;
            out_acc += warp_acc[w * head_dim + tid] * f;
        }

        if (n_splits > 1) {
            // emit this slice's PARTIAL: un-normalised acc plus its (m, l) so the
            // combine kernel can merge slices by log-sum-exp
            const size_t pidx = ((size_t) token_batch_idx * n_heads + head_idx) * n_splits + split_idx;
            if (tid == 0) {
                out_m[pidx] = m_tot;
                out_l[pidx] = l_tot;
            }
            out[pidx * head_dim + tid] = out_acc;
        } else {
            const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;
            out[out_idx] = out_acc / (l_tot + 1e-6f);
        }
        __syncthreads();  // q_s / warp_acc are rewritten next iteration
    }
}

// M7 phase 2: TILED PREFILL. The decode kernel gives one block per query token, so every
// query re-reads the whole KV span from HBM -- O(n^2) traffic with zero reuse (43.7 s to
// prefill 22K, ~18x off the static banded path). Here one block owns Q_TILE queries and
// each KV token it loads is scored against ALL of them, cutting KV traffic by Q_TILE.
// Per-query online-softmax state lives in registers (m, l, acc slice per lane), so Q_TILE
// is bounded by register pressure: 8 queries x 4 dims-per-lane = 32 acc registers.
// Cross-warp merge is the same log-sum-exp as the decode kernel, done per query.
#define PAGED_Q_TILE 4

// ============================ M7 phase 3: WMMA SKELETON ============================
// The f32 tile family is measured out: 2/4/8/16 -> 31,277 / 28,816 / 31,272 / 218,100 ms
// at 22K, i.e. a 1.52x ceiling where the gate wants >=5x (<= 8,745 ms). The escape is the
// same one flash-attention uses: hold the score tile in warp FRAGMENTS instead of per-lane
// accumulator arrays, so the QK^T product runs on tensor cores and register pressure stops
// being the binding constraint.
//
// Fragment shape: 16x16x16 half inputs with an f32 accumulator (nvcuda::wmma), following
// the in-tree pattern in lightning-indexer.cu (frag_q row_major / frag_k col_major /
// mma_sync into a float accumulator).
//
// ACCEPTANCE BAR, chosen up front and justified: q->data is FLOAT and wmma needs HALF
// operands, so Q must be converted at tile load. That is a real numeric change, so the
// bit-identical bar used for every other paged kernel does NOT apply here. The bar is the
// one tests/test-paged-banded.cpp already enforces for this op family --
//     max_abs < 2e-3 && nmse < 1e-6
// -- which is the half-input tolerance the suite was written around; no new tolerance is
// invented for this kernel, and the existing gate is the judge.
//
// Staging (why it is a skeleton and what the next commit fills in):
//   1. [this commit] layout + guard + bar recorded; dispatch stays on the f32 tiled path.
//   2. QK^T in fragments: q_h/k_h half tiles in smem (padded to avoid bank conflicts),
//      wmma::load_matrix_sync + mma_sync -> f32 score fragment.
//   3. online softmax over the score fragment, then P(half) x V(half) -> f32 out fragment.
//   4. flip DS4P_PAGED_QTILE=2 (fragment path) only after max_abs/nmse + >=5x + p28 3/3.
#define PAGED_WMMA_M 16
#define PAGED_WMMA_N 16
#define PAGED_WMMA_K 16
// smem for the fragment path: Q tile + K tile + V tile in half, +8 pad per row against
// bank conflicts, plus the f32 score tile. Recomputed at launch like the f32 path does.
#define PAGED_WMMA_SMEM(head_dim) \
    ((size_t) (PAGED_WMMA_M * ((head_dim) + 8) + 2 * PAGED_WMMA_N * ((head_dim) + 8)) * sizeof(half) \
     + (size_t) PAGED_WMMA_M * PAGED_WMMA_N * sizeof(float))

#include <mma.h>
namespace wmma = nvcuda::wmma;

// stage 2: QK^T for one 16x16 (query-tile x key-tile) pair, accumulated over head_dim in
// 16-wide chunks. q_h is [M][ld] row-major half; k_h is [N][ld] row-major half, and K^T is
// obtained WITHOUT a transpose copy by reading k_h as a col_major b-fragment: with base
// &k_h[0][c*16] and leading dimension ld, element [k][n] resolves to k_h[n][c*16+k],
// which is exactly K^T for that chunk. Scores land in an f32 accumulator (the whole point:
// the score tile lives in fragments, not in per-lane registers).
// One warp owns the tile; the caller supplies smem already staged.
__device__ __forceinline__ void paged_wmma_qk_range(const half * __restrict__ q_h,
                                                    const half * __restrict__ k_h,
                                                    const int    ld,
                                                    const int    head_dim,
                                                    const float  scale,
                                                    float * __restrict__ scores_out,
                                                    const int    c_begin,
                                                    const int    c_end) {
    wmma::fragment<wmma::accumulator, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, float> frag_acc;
    wmma::fill_fragment(frag_acc, 0.0f);

    for (int c = c_begin; c < c_end; c += PAGED_WMMA_K) {
        wmma::fragment<wmma::matrix_a, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, half, wmma::col_major> frag_k;
        wmma::load_matrix_sync(frag_q, q_h + c, ld);
        wmma::load_matrix_sync(frag_k, k_h + c, ld);
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
    }

    #pragma unroll
    for (int i = 0; i < frag_acc.num_elements; ++i) {
        frag_acc.x[i] *= scale;   // fold the 1/sqrt(d) here, as the f32 path does at load
    }
    wmma::store_matrix_sync(scores_out, frag_acc, PAGED_WMMA_N, wmma::mem_row_major);
}

__device__ __forceinline__ void paged_wmma_qk(const half * q_h, const half * k_h, const int ld,
                                              const int head_dim, const float scale, float * scores_out) {
    paged_wmma_qk_range(q_h, k_h, ld, head_dim, scale, scores_out, 0, head_dim);
}

// stage 3a: P x V for one tile. After the online softmax turns the score tile into
// probabilities, P (M x N_keys, half) times V (N_keys x head_dim, half) accumulates into
// the output fragment -- one f32 accumulator per 16-wide slice of head_dim. Both operands
// are row_major here: the K dimension of the product is N_keys (the tile's keys), which is
// V's leading axis, so no transpose is needed on this side either.
// out_acc is [M][ld_out] f32 and is ACCUMULATED into, because a query tile walks many key
// tiles and the online-softmax rescaling is applied by the caller between tiles.
__device__ __forceinline__ void paged_wmma_pv_range(const half * __restrict__ p_h,
                                                    const half * __restrict__ v_h,
                                                    const int    ld_v,
                                                    float * __restrict__ out_acc,
                                                    const int    ld_out,
                                                    const int    c_begin,
                                                    const int    c_end) {
    for (int c = c_begin; c < c_end; c += PAGED_WMMA_N) {
        wmma::fragment<wmma::accumulator, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, float> frag_out;
        wmma::load_matrix_sync(frag_out, out_acc + c, ld_out, wmma::mem_row_major);

        wmma::fragment<wmma::matrix_a, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, half, wmma::row_major> frag_p;
        wmma::fragment<wmma::matrix_b, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, half, wmma::row_major> frag_v;
        wmma::load_matrix_sync(frag_p, p_h, PAGED_WMMA_N);
        wmma::load_matrix_sync(frag_v, v_h + c, ld_v);
        wmma::mma_sync(frag_out, frag_p, frag_v, frag_out);

        wmma::store_matrix_sync(out_acc + c, frag_out, ld_out, wmma::mem_row_major);
    }
}

__device__ __forceinline__ void paged_wmma_pv(const half * p_h, const half * v_h, const int ld_v,
                                              const int head_dim, float * out_acc, const int ld_out) {
    paged_wmma_pv_range(p_h, v_h, ld_v, out_acc, ld_out, 0, head_dim);
}


__global__ void paged_attention_prefill_tiled_kernel(const float * __restrict__ q,
                                                     const half * __restrict__ kv_cache,
                                                     const int * __restrict__ block_table,
                                                     const int * __restrict__ context_lens,
                                                     const int * __restrict__ batch_offsets,
                                                     const int * __restrict__ batch_lens,
                                                     const size_t stride_token,
                                                     const size_t stride_head,
                                                     const size_t stride_block,
                                                     const int    n_heads_kv,
                                                     const int    block_size,
                                                     const int    max_blocks,
                                                     const float  scale,
                                                     const float * __restrict__ rel,
                                                     const int64_t rel_extent,
                                                     const int64_t visibility_window,
                                                     float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int q_tile   = blockIdx.z;
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp_id  = tid >> 5;

    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;
    const int n_warps  = head_dim >> 5;
    const int dpl      = head_dim >> 5;

    const int kv_head_idx = head_idx / (n_heads / n_heads_kv);

    const int seq_start      = batch_offsets[seq_idx];
    const int num_new_tokens = batch_lens[seq_idx];
    const int q_base         = q_tile * PAGED_Q_TILE;
    if (q_base >= num_new_tokens) {
        return;
    }
    const int q_cnt = min(PAGED_Q_TILE, num_new_tokens - q_base);

    // smem: q_s[Q_TILE * head_dim] | warp_m/l[n_warps * Q_TILE] | warp_acc[n_warps * Q_TILE * head_dim]
    float * q_s      = smem;
    float * warp_m   = q_s + PAGED_Q_TILE * head_dim;
    float * warp_l   = warp_m + n_warps * PAGED_Q_TILE;
    float * warp_acc = warp_l + n_warps * PAGED_Q_TILE;

    for (int i = 0; i < q_cnt; ++i) {
        const int tb = seq_start + q_base + i;
        q_s[i * head_dim + tid] =
            q[(size_t) tb * n_heads * head_dim + (size_t) head_idx * head_dim + tid] * scale;
    }
    __syncthreads();

    const int ctx_len   = context_lens[seq_idx];
    const int first_pos = ctx_len - num_new_tokens;   // logical pos of query 0 of the batch

    float m_i[PAGED_Q_TILE], l_i[PAGED_Q_TILE], acc_i[PAGED_Q_TILE][4];  // Q_TILE x dpl acc registers
    #pragma unroll
    for (int i = 0; i < PAGED_Q_TILE; ++i) {
        m_i[i] = -FLT_MAX; l_i[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < 4; ++d) { acc_i[i][d] = 0.0f; }
    }

    // the tile's queries span positions [first_pos+q_base, first_pos+q_base+q_cnt); the
    // last one sees the most context, so that bounds the walk
    const int n_tok_max = first_pos + q_base + q_cnt;
    const int lo_all    = (visibility_window > 0)
                        ? max(0, (int) (first_pos + q_base - visibility_window + 1)) : 0;

    for (int token = lo_all + warp_id; token < n_tok_max; token += n_warps) {
        const int    bid            = token / block_size;
        const int    physical_block = block_table[seq_idx * max_blocks + bid];
        const int    token_in_block = token % block_size;
        const size_t base   = (size_t) token_in_block * stride_token + (size_t) physical_block * stride_block;
        const size_t k_base = base + (size_t) kv_head_idx * stride_head;
        const size_t v_base = base + (size_t) (n_heads_kv + kv_head_idx) * stride_head;

        // load this KV token's lane slice ONCE, reuse across all queries in the tile
        float k_l[4], v_l[4];
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            if (d < dpl) {
                const int dim = lane + (d << 5);
                k_l[d] = __half2float(kv_cache[k_base + dim]);
                v_l[d] = __half2float(kv_cache[v_base + dim]);
            }
        }

        for (int i = 0; i < q_cnt; ++i) {
            const int q_pos = first_pos + q_base + i;
            if (token > q_pos) { continue; }                       // causal
            const int64_t rel_dist = (int64_t) q_pos - token;
            if (visibility_window > 0 && rel_dist >= visibility_window) { continue; }

            float part = 0.0f;
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                if (d < dpl) { part += q_s[i * head_dim + lane + (d << 5)] * k_l[d]; }
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                part += __shfl_down_sync(0xffffffffu, part, offset);
            }
            float qk = __shfl_sync(0xffffffffu, part, 0);

            if (rel != nullptr && rel_dist < rel_extent) {
                qk += rel[((size_t) (seq_start + q_base + i) * n_heads + head_idx) * rel_extent + rel_dist];
            }

            const float m_new = fmaxf(m_i[i], qk);
            const float e_old = __expf(m_i[i] - m_new);
            const float p     = __expf(qk - m_new);
            l_i[i] = l_i[i] * e_old + p;
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                if (d < dpl) { acc_i[i][d] = acc_i[i][d] * e_old + p * v_l[d]; }
            }
            m_i[i] = m_new;
        }
    }

    // per-query cross-warp merge
    for (int i = 0; i < q_cnt; ++i) {
        if (lane == 0) {
            warp_m[warp_id * PAGED_Q_TILE + i] = m_i[i];
            warp_l[warp_id * PAGED_Q_TILE + i] = l_i[i];
        }
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            if (d < dpl) {
                warp_acc[(warp_id * PAGED_Q_TILE + i) * head_dim + lane + (d << 5)] = acc_i[i][d];
            }
        }
    }
    __syncthreads();

    for (int i = 0; i < q_cnt; ++i) {
        float m_tot = -FLT_MAX;
        for (int w = 0; w < n_warps; ++w) { m_tot = fmaxf(m_tot, warp_m[w * PAGED_Q_TILE + i]); }
        float l_tot = 0.0f, out_acc = 0.0f;
        for (int w = 0; w < n_warps; ++w) {
            const float f = __expf(warp_m[w * PAGED_Q_TILE + i] - m_tot);
            l_tot   += warp_l[w * PAGED_Q_TILE + i] * f;
            out_acc += warp_acc[(w * PAGED_Q_TILE + i) * head_dim + tid] * f;
        }
        const int tb = seq_start + q_base + i;
        out[(size_t) tb * n_heads * head_dim + (size_t) head_idx * head_dim + tid] = out_acc / (l_tot + 1e-6f);
    }
}

// double-buffered fragment prefill: warps 1-3 stage key tile T+1 while warp 0 computes
// tile T (qk fragments -> softmax rows -> pv fragments), overlapping the HBM staging
// latency with the tensor-core work instead of barriering the whole block around both.
// Basis: the 4-warp shared-accumulator shape (31,915 ms) -- its remaining cost was every
// warp waiting through every phase. smem doubles only the K/V tiles (~+8.7 KB), keeping
// occupancy, unlike the per-warp variant whose private accumulators collapsed it (53.3 s).
// Dark: DS4P_PAGED_QTILE=2 only.
__global__ void paged_attention_prefill_wmma_kernel(const float * __restrict__ q,
                                                    const half * __restrict__ kv_cache,
                                                    const int * __restrict__ block_table,
                                                    const int * __restrict__ context_lens,
                                                    const int * __restrict__ batch_offsets,
                                                    const int * __restrict__ batch_lens,
                                                    const size_t stride_token,
                                                    const size_t stride_head,
                                                    const size_t stride_block,
                                                    const int    n_heads_kv,
                                                    const int    block_size,
                                                    const int    max_blocks,
                                                    const float  scale,
                                                    const float * __restrict__ rel,
                                                    const int64_t rel_extent,
                                                    const int64_t visibility_window,
                                                    const int    head_dim,
                                                    float * __restrict__ out) {
    extern __shared__ char smem_raw[];
    const int ld      = head_dim + 8;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;

    // q_h[M*ld] k_h[N*ld] v_h[N*ld] p_h[M*N] (half) | sc_w[4][M*N] o_acc[M*ld] m l resc (f32)
    half  * q_h  = (half *) smem_raw;
    half  * k_h  = q_h + PAGED_WMMA_M * ld;
    half  * v_h  = k_h + PAGED_WMMA_N * ld;
    half  * p_h  = v_h + PAGED_WMMA_N * ld;
    float * sc_w = (float *) (p_h + PAGED_WMMA_M * PAGED_WMMA_N);   // 4 partial tiles
    float * sc   = sc_w;                                            // combined in-place into warp 0's tile
    float * o_acc  = sc_w + 4 * PAGED_WMMA_M * PAGED_WMMA_N;
    float * m_s    = o_acc + PAGED_WMMA_M * ld;
    float * l_s    = m_s + PAGED_WMMA_M;
    float * resc_s = l_s + PAGED_WMMA_M;

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int q_tile   = blockIdx.z;
    const int n_heads  = gridDim.x;
    const int kv_head  = head_idx / (n_heads / n_heads_kv);

    const int seq_start = batch_offsets[seq_idx];
    const int n_new     = batch_lens[seq_idx];
    const int q_base    = q_tile * PAGED_WMMA_M;
    if (q_base >= n_new) { return; }
    const int q_cnt     = min(PAGED_WMMA_M, n_new - q_base);
    const int first_pos = context_lens[seq_idx] - n_new;
    const int n_tok     = first_pos + q_base + q_cnt;

    // head_dim chunks (QK reduction) and slices (PV output) partitioned across the 4 warps
    const int n_chunk   = head_dim / PAGED_WMMA_K;
    const int cpw       = (n_chunk + 3) / 4;
    const int qk_begin  = min(warp_id * cpw, n_chunk) * PAGED_WMMA_K;
    const int qk_end    = min((warp_id + 1) * cpw, n_chunk) * PAGED_WMMA_K;
    const int pv_begin  = qk_begin;   // WMMA_K == WMMA_N so the same partition serves both
    const int pv_end    = qk_end;

    for (int e = tid; e < PAGED_WMMA_M * ld; e += blockDim.x) {
        const int i = e / ld, d = e % ld;
        q_h[e]   = (i < q_cnt && d < head_dim)
                 ? __float2half(q[(size_t)(seq_start + q_base + i) * n_heads * head_dim
                                  + (size_t) head_idx * head_dim + d])
                 : __float2half(0.0f);
        o_acc[e] = 0.0f;
    }
    for (int i = tid; i < PAGED_WMMA_M; i += blockDim.x) { m_s[i] = -FLT_MAX; l_s[i] = 0.0f; }
    __syncthreads();

    for (int kt = 0; kt < n_tok; kt += PAGED_WMMA_N) {
        for (int e = tid; e < PAGED_WMMA_N * ld; e += blockDim.x) {
            const int j = e / ld, d = e % ld;
            const int tok = kt + j;
            half kv_k = __float2half(0.0f), kv_v = __float2half(0.0f);
            if (tok < n_tok && d < head_dim) {
                const int pb   = block_table[seq_idx * max_blocks + tok / block_size];
                const size_t b = (size_t)(tok % block_size) * stride_token + (size_t) pb * stride_block;
                kv_k = kv_cache[b + (size_t) kv_head * stride_head + d];
                kv_v = kv_cache[b + (size_t)(n_heads_kv + kv_head) * stride_head + d];
            }
            k_h[e] = kv_k; v_h[e] = kv_v;
        }
        __syncthreads();

        // every warp computes a partial score tile over its head_dim chunk range
        // (scale is linear, so scale*partial sums correctly across warps)
        paged_wmma_qk_range(q_h, k_h, ld, head_dim, scale, sc_w + warp_id * PAGED_WMMA_M * PAGED_WMMA_N,
                            qk_begin, qk_end);
        __syncthreads();

        for (int e = tid; e < PAGED_WMMA_M * PAGED_WMMA_N; e += blockDim.x) {
            sc[e] = sc_w[e] + sc_w[e + 256] + sc_w[e + 512] + sc_w[e + 768];
        }
        __syncthreads();

        for (int i = tid; i < PAGED_WMMA_M; i += blockDim.x) {
            const int q_pos = first_pos + q_base + i;
            float rmax = -FLT_MAX;
            for (int j = 0; j < PAGED_WMMA_N; ++j) {
                const int tok = kt + j;
                float v = sc[i * PAGED_WMMA_N + j];
                const int64_t rd = (int64_t) q_pos - tok;
                if (i >= q_cnt || tok >= n_tok || tok > q_pos ||
                    (visibility_window > 0 && rd >= visibility_window)) {
                    v = -FLT_MAX;
                } else if (rel != nullptr && rd < rel_extent) {
                    v += rel[((size_t)(seq_start + q_base + i) * n_heads + head_idx) * rel_extent + rd];
                }
                sc[i * PAGED_WMMA_N + j] = v;
                rmax = fmaxf(rmax, v);
            }
            const float m_new = fmaxf(m_s[i], rmax);
            if (m_new == -FLT_MAX) {
                for (int j = 0; j < PAGED_WMMA_N; ++j) { p_h[i * PAGED_WMMA_N + j] = __float2half(0.0f); }
                resc_s[i] = 1.0f;
                continue;
            }
            const float resc = (m_s[i] == -FLT_MAX) ? 0.0f : __expf(m_s[i] - m_new);
            float row_l = 0.0f;
            for (int j = 0; j < PAGED_WMMA_N; ++j) {
                const float v  = sc[i * PAGED_WMMA_N + j];
                const float pv = (v == -FLT_MAX) ? 0.0f : __expf(v - m_new);
                p_h[i * PAGED_WMMA_N + j] = __float2half(pv);
                row_l += pv;
            }
            resc_s[i] = resc;
            l_s[i] = l_s[i] * resc + row_l;
            m_s[i] = m_new;
        }
        __syncthreads();

        // o_acc rescale spread across the whole block (was 128 serial mults per row-lane)
        for (int e = tid; e < PAGED_WMMA_M * head_dim; e += blockDim.x) {
            const int i = e / head_dim, d = e % head_dim;
            o_acc[i * ld + d] *= resc_s[i];
        }
        __syncthreads();

        // every warp accumulates its own head_dim slice of P x V (disjoint o_acc columns)
        paged_wmma_pv_range(p_h, v_h, ld, o_acc, ld, pv_begin, pv_end);
        __syncthreads();
    }

    for (int e = tid; e < q_cnt * head_dim; e += blockDim.x) {
        const int i = e / head_dim, d = e % head_dim;
        out[(size_t)(seq_start + q_base + i) * n_heads * head_dim
            + (size_t) head_idx * head_dim + d] = o_acc[i * ld + d] / (l_s[i] + 1e-6f);
    }
}

// ---------------------------------------------------------------------------
// M7 fragment-held paged prefill (DS4P_PAGED_QTILE=3).
//
// Ported discipline from ggml/src/ggml-cuda/fattn-mma-f16.cuh (upstream
// llama.cpp, Johannes Gaessler): keep the attention accumulator in mma
// register fragments across the WHOLE key loop instead of round-tripping it
// through shared memory once per key tile, and exploit the fact that the
// m16n8k16 f32 accumulator element mapping is identical to the half2
// A-operand mapping, so the softmax probabilities feed the next mma with
// zero data movement. Opaque wmma::fragment cannot express either.
//
// Shape: one block = 4 warps = 4 independent 16-row q tiles (64 tokens).
// Warps share the staged K/V tile but own disjoint q rows, so there is no
// cross-warp softmax merge and no per-tile barrier beyond the K/V staging.
// V is staged TRANSPOSED so P x V is a plain row.col mma.
// ---------------------------------------------------------------------------
#define PAGED_MMA_WARPS 4
#define PAGED_MMA_M     16
#define PAGED_MMA_N     16
#define PAGED_MMA_KV    16   // keys staged per round (64 measured worse: smem cost occupancy)
#define PAGED_MMA_LDV   (PAGED_MMA_KV + 8)   // pad; multiple of 8 halves for ldmatrix

template <int HD>
__global__ __launch_bounds__(128, 4)   // profile: blocks/SM was register-capped at 2
void paged_attention_prefill_mma_kernel(const float * __restrict__ q,
                                        const half * __restrict__ kv_cache,
                                        const int * __restrict__ block_table,
                                        const int * __restrict__ context_lens,
                                        const int * __restrict__ batch_offsets,
                                        const int * __restrict__ batch_lens,
                                        const size_t stride_token,
                                        const size_t stride_head,
                                        const size_t stride_block,
                                        const int    n_heads_kv,
                                        const int    block_size,
                                        const int    max_blocks,
                                        const float  scale,
                                        const float * __restrict__ rel,
                                        const int64_t rel_extent,
                                        const int64_t visibility_window,
                                        float * __restrict__ out,
                                        const int    n_splits,
                                        float * __restrict__ part_m,
                                        float * __restrict__ part_l,
                                        const bool   use_cp_async) {
#ifdef TURING_MMA_AVAILABLE
    using namespace ggml_cuda_mma;
    typedef tile<16, 16, float> tile_acc;   // score tile and output tiles
    typedef tile<16,  8, half2> tile_ab;    // A and B operands

    constexpr int NC  = HD / 16;            // head_dim chunks
    constexpr int ld  = HD + 8;             // q/k row pitch in halves

    extern __shared__ char smem_raw[];
    // q_h is dead once the Q fragments are in registers, so K/V OVERLAY it:
    // smem = max(q, k+v) instead of q+k+v, which buys blocks per SM.
    half * q_h = (half *) smem_raw;                        // [64][ld], live until q_frag loaded
    half * k_h = (half *) smem_raw;                        // [PAGED_MMA_KV][ld]
    half * v_h = k_h + PAGED_MMA_KV * ld;                  // [PAGED_MMA_KV][ld], transposed on LOAD

    const int lane    = threadIdx.x;                 // mma.cuh indexes by threadIdx.x
    const int warp_id = threadIdx.y;
    const int tid     = warp_id * 32 + lane;
    const int nthr    = PAGED_MMA_WARPS * 32;

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int n_heads  = gridDim.x;
    const int kv_head  = head_idx / (n_heads / n_heads_kv);

    const int seq_start = batch_offsets[seq_idx];
    const int n_new     = batch_lens[seq_idx];
    const int q_tile    = (n_splits > 1) ? (int) blockIdx.z / n_splits : (int) blockIdx.z;
    const int split_idx = (n_splits > 1) ? (int) blockIdx.z % n_splits : 0;
    const int q_base    = q_tile * (PAGED_MMA_WARPS * PAGED_MMA_M);
    if (q_base >= n_new) { return; }
    const int q_cnt_blk = min(PAGED_MMA_WARPS * PAGED_MMA_M, n_new - q_base);
    const int first_pos = context_lens[seq_idx] - n_new;
    const int n_tok     = first_pos + q_base + q_cnt_blk;

    // this warp's q rows
    const int q_base_w = q_base + warp_id * PAGED_MMA_M;
    const int q_cnt_w  = min(PAGED_MMA_M, max(0, n_new - q_base_w));

    for (int e = tid; e < PAGED_MMA_WARPS * PAGED_MMA_M * ld; e += nthr) {
        const int i = e / ld, d = e % ld;
        q_h[e] = (q_base + i < n_new && d < HD)
               ? __float2half(q[(size_t)(seq_start + q_base + i) * n_heads * HD
                                + (size_t) head_idx * HD + d])
               : __float2half(0.0f);
    }
    __syncthreads();

    // Q fragments are loaded ONCE and stay in registers for the whole key loop.
    tile_ab  q_frag[NC];
    tile_acc o_frag[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        load_ldmatrix(q_frag[c], (const half2 *) (q_h + (size_t) warp_id * PAGED_MMA_M * ld + c * 16), ld / 2);
#pragma unroll
        for (int l = 0; l < tile_acc::ne; ++l) { o_frag[c].x[l] = 0.0f; }
    }

    float m_r[2] = { -FLT_MAX, -FLT_MAX };
    float l_r[2] = { 0.0f, 0.0f };

    // Analytic band: everything older than the window is invisible, so SKIP it instead of
    // masking it. The old grid-per-token kernel does this (its `lo`), and without it a
    // banded model would do the full O(n^2) and throw ~95% away -- a silent regression the
    // unbanded wall cannot see. Block-level bound = the block's SMALLEST q position;
    // per-row edges are still handled by the mask below.
    int lo_blk = 0;
    if (visibility_window > 0) {
        lo_blk = max(0, first_pos + q_base - (int) visibility_window + 1);
        lo_blk = (lo_blk / PAGED_MMA_N) * PAGED_MMA_N;   // keep tiles aligned
    }

    // split-K: this block owns keys [kt_lo, kt_hi); partials merged by the combine kernel.
    // The span is rounded to the mma tile so no split straddles a tile.
    const int span  = (n_splits > 1)
                    ? ((n_tok - lo_blk + n_splits - 1) / n_splits + PAGED_MMA_N - 1) / PAGED_MMA_N * PAGED_MMA_N
                    : n_tok;
    const int kt_lo = (n_splits > 1) ? lo_blk + split_idx * span : lo_blk;
    const int kt_hi = (n_splits > 1) ? min(n_tok, kt_lo + span) : n_tok;

    for (int kt0 = kt_lo; kt0 < kt_hi; kt0 += PAGED_MMA_KV) {
        __syncthreads();
        if (use_cp_async) {
            // 8 halves (16 B) per instruction, straight global->shared with no register
            // round-trip. Every offset here is a multiple of 8 halves, so both ends are
            // 16 B aligned: ld = HD + 8, stride_head = HD, stride_token = 2*n_heads_kv*HD.
            for (int e = tid * 8; e < PAGED_MMA_KV * HD; e += nthr * 8) {
                const int j = e / HD, d = e % HD;
                const int tok = kt0 + j;
                if (tok < n_tok) {
                    const int pb   = block_table[seq_idx * max_blocks + tok / block_size];
                    const size_t b = (size_t)(tok % block_size) * stride_token + (size_t) pb * stride_block;
                    cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(k_h + j * ld + d),
                                        kv_cache + b + (size_t) kv_head * stride_head + d);
                    cp_async_cg_16<128>(ggml_cuda_cvta_generic_to_shared(v_h + j * ld + d),
                                        kv_cache + b + (size_t)(n_heads_kv + kv_head) * stride_head + d);
                } else {
                    for (int u = 0; u < 8; ++u) {
                        k_h[j * ld + d + u] = __float2half(0.0f);
                        v_h[j * ld + d + u] = __float2half(0.0f);
                    }
                }
            }
            cp_async_wait_all();
        } else {
            for (int e = tid; e < PAGED_MMA_KV * HD; e += nthr) {
                const int j = e / HD, d = e % HD;
                const int tok = kt0 + j;
                half kv_k = __float2half(0.0f), kv_v = __float2half(0.0f);
                if (tok < n_tok) {
                    const int pb   = block_table[seq_idx * max_blocks + tok / block_size];
                    const size_t b = (size_t)(tok % block_size) * stride_token + (size_t) pb * stride_block;
                    kv_k = kv_cache[b + (size_t) kv_head * stride_head + d];
                    kv_v = kv_cache[b + (size_t)(n_heads_kv + kv_head) * stride_head + d];
                }
                k_h[j * ld + d] = kv_k;
                v_h[j * ld + d] = kv_v;   // contiguous; ldmatrix.trans supplies the transpose
            }
        }
        __syncthreads();

        // sub-tiles run back to back with NO barriers: all state is in registers
        for (int sub = 0; sub < PAGED_MMA_KV / PAGED_MMA_N; ++sub) {
        const int kt = kt0 + sub * PAGED_MMA_N;
        if (kt >= kt_hi) { break; }   // uniform across the block

        tile_acc s_frag;
#pragma unroll
        for (int l = 0; l < tile_acc::ne; ++l) { s_frag.x[l] = 0.0f; }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            tile_ab k_frag;
            load_ldmatrix(k_frag, (const half2 *) (k_h + (size_t) sub * PAGED_MMA_N * ld + c * 16), ld / 2);
            mma(s_frag, q_frag[c], k_frag);
        }

        // mask + bias + online softmax, entirely in registers
        float rmax[2] = { -FLT_MAX, -FLT_MAX };
#pragma unroll
        for (int l = 0; l < tile_acc::ne; ++l) {
            const int i   = tile_acc::get_i(l);
            const int j   = tile_acc::get_j(l);
            const int tok = kt + j;
            const int q_pos = first_pos + q_base_w + i;
            float v = s_frag.x[l] * scale;
            const int64_t rd = (int64_t) q_pos - tok;
            if (i >= q_cnt_w || tok >= n_tok || tok > q_pos ||
                (visibility_window > 0 && rd >= visibility_window)) {
                v = -FLT_MAX;
            } else if (rel != nullptr && rd < rel_extent) {
                v += rel[((size_t)(seq_start + q_base_w + i) * n_heads + head_idx) * rel_extent + rd];
            }
            s_frag.x[l] = v;
            const int r = (l / 2) % 2;
            rmax[r] = fmaxf(rmax[r], v);
        }
        // the 4 lanes holding one row differ only in the low 2 lane bits
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1, 32));
            rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2, 32));
        }

        float m_new[2], resc[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            m_new[r] = fmaxf(m_r[r], rmax[r]);
            resc[r]  = (m_r[r] == -FLT_MAX) ? 0.0f : __expf(m_r[r] - m_new[r]);
        }

        float rsum[2] = { 0.0f, 0.0f };
        tile_ab p_frag;
#pragma unroll
        for (int l = 0; l < tile_acc::ne; l += 2) {
            const int r = (l / 2) % 2;
            const float p0 = (s_frag.x[l]     == -FLT_MAX || m_new[r] == -FLT_MAX)
                           ? 0.0f : __expf(s_frag.x[l]     - m_new[r]);
            const float p1 = (s_frag.x[l + 1] == -FLT_MAX || m_new[r] == -FLT_MAX)
                           ? 0.0f : __expf(s_frag.x[l + 1] - m_new[r]);
            rsum[r] += p0 + p1;
            // accumulator element pair (l, l+1) IS half2 A-operand element l/2
            p_frag.x[l / 2] = make_half2(p0, p1);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            rsum[r] += __shfl_xor_sync(0xffffffffu, rsum[r], 1, 32);
            rsum[r] += __shfl_xor_sync(0xffffffffu, rsum[r], 2, 32);
            l_r[r] = l_r[r] * resc[r] + rsum[r];
            m_r[r] = m_new[r];
        }

        // rescale + accumulate: registers only, no shared-memory round trip.
        // The rescale is 8*NC register multiplies per tile, but it is only NEEDED when a
        // row max actually grew. In causal prefill the max stops growing after the first
        // few tiles, so skip it warp-uniformly when no row in this warp changed.
        const bool need_resc = (resc[0] != 1.0f) | (resc[1] != 1.0f);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (need_resc) {
#pragma unroll
                for (int l = 0; l < tile_acc::ne; ++l) { o_frag[c].x[l] *= resc[(l / 2) % 2]; }
            }
            tile_ab v_frag;
            load_ldmatrix_trans(v_frag, (const half2 *) (v_h + (size_t) sub * PAGED_MMA_N * ld + c * 16), ld / 2);
            mma(o_frag[c], p_frag, v_frag);
        }
        }   // sub
    }

    if (n_splits > 1) {
        // raw (unnormalised) partials; paged_attention_combine_kernel folds them by log-sum-exp
#pragma unroll
        for (int c = 0; c < NC; ++c) {
#pragma unroll
            for (int l = 0; l < tile_acc::ne; ++l) {
                const int i = tile_acc::get_i(l);
                if (i >= q_cnt_w) { continue; }
                const int d = c * 16 + tile_acc::get_j(l);
                const size_t pidx = ((size_t)(seq_start + q_base_w + i) * n_heads + head_idx) * n_splits + split_idx;
                out[pidx * HD + d] = o_frag[c].x[l];
            }
        }
        // one thread per row writes that row's (m, l)
        if (lane % 4 == 0) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int i = r * 8 + lane / 4;
                if (i < q_cnt_w) {
                    const size_t pidx = ((size_t)(seq_start + q_base_w + i) * n_heads + head_idx) * n_splits + split_idx;
                    part_m[pidx] = m_r[r];
                    part_l[pidx] = l_r[r];
                }
            }
        }
        return;
    }

#pragma unroll
    for (int c = 0; c < NC; ++c) {
#pragma unroll
        for (int l = 0; l < tile_acc::ne; ++l) {
            const int i = tile_acc::get_i(l);
            if (i >= q_cnt_w) { continue; }
            const int d = c * 16 + tile_acc::get_j(l);
            out[(size_t)(seq_start + q_base_w + i) * n_heads * HD + (size_t) head_idx * HD + d]
                = o_frag[c].x[l] / (l_r[(l / 2) % 2] + 1e-6f);
        }
    }
#else
    GGML_UNUSED_VARS(q, kv_cache, block_table, context_lens, batch_offsets, batch_lens,
                     stride_token, stride_head, stride_block, n_heads_kv, block_size,
                     max_blocks, scale, rel, rel_extent, visibility_window, out,
                     n_splits, part_m, part_l, use_cp_async);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

// M7: merge the split-K partials for one (token, head) by log-sum-exp.
// grid (n_heads, n_tokens), blockDim head_dim.
__global__ void paged_attention_combine_kernel(const float * __restrict__ part_acc,
                                               const float * __restrict__ part_m,
                                               const float * __restrict__ part_l,
                                               const int   n_splits,
                                               float * __restrict__ out) {
    const int head_idx = blockIdx.x;
    const int tok_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int n_heads  = gridDim.x;
    const int head_dim = blockDim.x;

    const size_t base = ((size_t) tok_idx * n_heads + head_idx) * n_splits;

    float m_tot = -FLT_MAX;
    for (int s = 0; s < n_splits; ++s) {
        m_tot = fmaxf(m_tot, part_m[base + s]);
    }
    float l_tot = 0.0f;
    float acc   = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float f = __expf(part_m[base + s] - m_tot);
        l_tot += part_l[base + s] * f;
        acc   += part_acc[(base + s) * head_dim + tid] * f;
    }

    out[((size_t) tok_idx * n_heads + head_idx) * head_dim + tid] = acc / (l_tot + 1e-6f);
}

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // banded variant (rel_logits at src[10], 3b): rel_extent / visibility_window at
    // op_params bytes [16,24)/[24,32) -- the banded-FA layout
    const ggml_tensor * rel = dst->src[10];
    int64_t rel_extent        = 0;
    int64_t visibility_window = 0;
    memcpy(&rel_extent,        &dst->op_params[4], sizeof(rel_extent));
    memcpy(&visibility_window, &dst->op_params[6], sizeof(visibility_window));
    if (rel) {
        GGML_ASSERT(rel->type == GGML_TYPE_F32 && ggml_is_contiguous(rel) &&
                    "paged banded attention: rel_logits must be contiguous F32");
        GGML_ASSERT(rel->ne[0] == rel_extent);
    }

    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_new         = dst->src[1];
    const ggml_tensor * v_new         = dst->src[2];
    const ggml_tensor * kv_cache      = dst->src[3];  // KV interleaved layout
    const ggml_tensor * block_table   = dst->src[5];
    const ggml_tensor * write_slots   = dst->src[6];
    const ggml_tensor * context_lens  = dst->src[7];
    const ggml_tensor * batch_offsets = dst->src[8];
    const ggml_tensor * batch_lens    = dst->src[9];

    const float * op_params_f = (const float *) (dst->op_params);
    const float   scale       = op_params_f[0];
    const int     block_size  = ((const int32_t *) (op_params_f + 1))[0];
    const int     max_blocks  = ((const int32_t *) (op_params_f + 2))[0];

    const int head_dim   = q->ne[0];
    const int n_heads    = q->ne[1];
    const int n_seq      = batch_lens->ne[0];
    const int n_heads_kv = k_new->ne[1];

    GGML_ASSERT(n_heads != 0 && "n_head cannot be 0.");
    GGML_ASSERT(n_heads_kv != 0 && "n_heads_kv cannot be 0.");
    GGML_ASSERT(head_dim <= 1024 && "head_dim exceeds maximum supported (1024)");
    GGML_ASSERT(n_heads % n_heads_kv == 0 && "n_heads must be divisible by n_heads_kv");

    // Extracting strides
    const size_t stride_token = kv_cache->nb[1] / sizeof(half);
    const size_t stride_head  = kv_cache->nb[2] / sizeof(half);
    const size_t stride_block = kv_cache->nb[3] / sizeof(half);

    dim3 block_dims(head_dim);       // one thread per dimension of head
    dim3 grid_dims(n_heads, n_seq);  // one block per head per sequence

    // Write kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_write_kernel<<<dim3(n_heads_kv, n_seq), dim3(head_dim), 0, ctx.stream()>>>(
        (const float *) k_new->data, (const float *) v_new->data, (half *) kv_cache->data,
        (const int *) write_slots->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size);

    if (head_dim == 64 || head_dim == 128) {
        // warp-parallel kernel: q_s[head_dim] + warp_m/l[n_warps each] + warp_acc[n_warps*head_dim]
        const size_t n_warps    = (size_t) head_dim / 32;
        const size_t smem_bytes = (head_dim + 2 * n_warps + n_warps * head_dim) * sizeof(float);

        // grid.z = query tokens: covers the largest per-seq chunk; blocks whose z exceeds
        // their seq's batch_lens early-return (batch_lens lives on device, so the exact
        // per-seq max is not host-visible -- total n_tokens is a safe upper bound)
        const int n_tokens_total = (int) q->ne[2];

        // M7 split-K: decode-only batches (one query per seq) leave blockIdx.z free and
        // are exactly the case that starves for parallelism at long context -- a single
        // (head, seq) block walking 22K tokens measured 220 ms/token (18-28x off the
        // static banded path). Split the context across the grid and merge partials.
        // DS4P_PAGED_SPLITK=0 disables; default splits only when the context is long
        // enough for the extra launch + combine to pay.
        static const int splitk_env = []() {
            const char * s = getenv("DS4P_PAGED_SPLITK");
            return s ? atoi(s) : -1;   // -1 = auto
        }();
        // M7 phase 2: tiled prefill, OFF by default until gated (DS4P_PAGED_QTILE=1).
        // A wrong attention kernel is the expensive class of bug this lane keeps proving,
        // so it ships dark: the measured path is unchanged unless the flag is set.
        static const bool qtile_on = []() {
            const char * s = getenv("DS4P_PAGED_QTILE");
            return s && atoi(s) != 0;
        }();
        // Prefill kernel selection. 3 = the register-held mma path and now the DEFAULT:
        // 12.8x over the grid-per-token path (3,426 vs 43,726 ms at 22K), equivalence
        // max_abs 4.067e-05, byte-identical generation, P2-8 arms 3/3. 0 restores the
        // original grid-per-token prefill, 1 the f32 tile path, 2 the wmma path.
        // Unsupported head_dims fall through to 0 on their own.
        static const int qtile_mode = []() {
            const char * s = getenv("DS4P_PAGED_QTILE");
            return s ? atoi(s) : 3;
        }();
        if (qtile_mode == 3 && n_tokens_total > n_seq && (head_dim == 64 || head_dim == 128)) {
            const int    rows_blk  = PAGED_MMA_WARPS * PAGED_MMA_M;
            const int    n_q_tiles = (n_tokens_total + rows_blk - 1) / rows_blk;
            const int    ld        = head_dim + 8;
            const size_t smem_m    = sizeof(half) * (size_t) std::max(rows_blk * ld, 2 * PAGED_MMA_KV * ld);
            if (smem_m > 48 * 1024) {
                GGML_ASSERT(smem_m <= 96 * 1024 && "mma prefill smem exceeds 96KB");
                if (head_dim == 64) {
                    CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<64>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_m));
                } else {
                    CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_mma_kernel<128>,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_m));
                }
            }
            // The profile said this kernel is starved, not slow: 256 blocks on ~188 SMs is
            // 0.7 waves and 5.44 active warps/SM. Split the KEY range across extra blocks so
            // the warps exist, then fold the partials with the same log-sum-exp combine
            // kernel the decode path already uses.
            const int  base_blocks = n_heads * n_seq * n_q_tiles;
            const int  nsm         = ggml_cuda_info().devices[ctx.device].nsm;
            static const int psplit_env = []() {
                const char * s = getenv("DS4P_PAGED_PSPLIT");
                return s ? atoi(s) : -1;
            }();
            int n_splits = 1;
            if (psplit_env >= 0) {
                n_splits = std::max(1, psplit_env);
            } else if (base_blocks < 4 * nsm) {
                n_splits = std::min(8, (4 * nsm + base_blocks - 1) / base_blocks);
            }

            // cp.async staging: refuted at 5.44 warps/SM (nothing to hide behind), worth
            // repricing now that split-K + the register cap tripled occupancy.
            static const int cpasync_env = []() {
                const char * s = getenv("DS4P_PAGED_CPASYNC");
                return s ? atoi(s) : 1;
            }();
            const bool cpa = cpasync_env != 0 && ggml_cuda_info().devices[ctx.device].cc >= GGML_CUDA_CC_AMPERE;

            const dim3 grid(n_heads, n_seq, n_q_tiles * n_splits);
            const dim3 blk(32, PAGED_MMA_WARPS);   // mma.cuh indexes fragments by threadIdx.x

            ggml_cuda_pool_alloc<float> part_acc(ctx.pool());
            ggml_cuda_pool_alloc<float> part_m(ctx.pool());
            ggml_cuda_pool_alloc<float> part_l(ctx.pool());
            float * p_out = (float *) dst->data;
            float * p_m   = nullptr;
            float * p_l   = nullptr;
            if (n_splits > 1) {
                const size_t n_part = (size_t) n_tokens_total * n_heads * n_splits;
                p_out = part_acc.alloc(n_part * head_dim);
                p_m   = part_m.alloc(n_part);
                p_l   = part_l.alloc(n_part);
            }

#define DS4P_LAUNCH_MMA(HD)                                                                          \
            paged_attention_prefill_mma_kernel<HD><<<grid, blk, smem_m, ctx.stream()>>>(              \
                (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data, \
                (const int *) context_lens->data, (const int *) batch_offsets->data,                 \
                (const int *) batch_lens->data, stride_token, stride_head, stride_block, n_heads_kv,  \
                block_size, max_blocks, scale, rel ? (const float *) rel->data : nullptr, rel_extent, \
                visibility_window, p_out, n_splits, p_m, p_l, cpa)
            if (head_dim == 64) { DS4P_LAUNCH_MMA(64); } else { DS4P_LAUNCH_MMA(128); }
#undef DS4P_LAUNCH_MMA

            if (n_splits > 1) {
                paged_attention_combine_kernel<<<dim3(n_heads, n_tokens_total), dim3(head_dim), 0, ctx.stream()>>>(
                    p_out, p_m, p_l, n_splits, (float *) dst->data);
            }
            return;
        }

        if (qtile_mode == 2 && n_tokens_total > n_seq) {
            const int    n_q_tiles = (n_tokens_total + PAGED_WMMA_M - 1) / PAGED_WMMA_M;
            const int    ld        = head_dim + 8;
            // q_h + K + V + P (half) | 4 partial score tiles + o_acc + m + l + resc (f32)
            const size_t smem_w    = (size_t) (PAGED_WMMA_M * ld + 2 * PAGED_WMMA_N * ld
                                             + PAGED_WMMA_M * PAGED_WMMA_N) * sizeof(half)
                                   + (size_t) (4 * PAGED_WMMA_M * PAGED_WMMA_N + PAGED_WMMA_M * ld
                                             + 3 * PAGED_WMMA_M) * sizeof(float);
            if (smem_w > 48 * 1024) {
                GGML_ASSERT(smem_w <= 96 * 1024 && "wmma prefill smem exceeds 96KB");
                CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_wmma_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_w));
            }
            // 4 warps, ALL issuing mma: QK as per-warp partial score tiles summed in smem,
            // PV as per-warp head_dim slices; softmax rows + o_acc rescale block-wide.
            paged_attention_prefill_wmma_kernel<<<dim3(n_heads, n_seq, n_q_tiles), dim3(128), smem_w, ctx.stream()>>>(
                (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                rel ? (const float *) rel->data : nullptr, rel_extent, visibility_window, head_dim,
                (float *) dst->data);
            return;
        }

        if (qtile_on && n_tokens_total > n_seq) {   // multi-query (prefill) batch
            const int    n_q_tiles  = (n_tokens_total + PAGED_Q_TILE - 1) / PAGED_Q_TILE;
            const size_t n_warps_t  = (size_t) head_dim / 32;
            const size_t smem_tiled = (PAGED_Q_TILE * head_dim
                                     + 2 * n_warps_t * PAGED_Q_TILE
                                     + n_warps_t * PAGED_Q_TILE * head_dim) * sizeof(float);
            if (smem_tiled > 48 * 1024) {
                CUDA_CHECK(cudaFuncSetAttribute(paged_attention_prefill_tiled_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem_tiled));
            }
            paged_attention_prefill_tiled_kernel<<<dim3(n_heads, n_seq, n_q_tiles), dim3(head_dim), smem_tiled, ctx.stream()>>>(
                (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                rel ? (const float *) rel->data : nullptr, rel_extent, visibility_window, (float *) dst->data);
            return;
        }

        int n_splits = 1;
        if (splitk_env != 0 && n_tokens_total == n_seq) {   // decode-only batch
            const int ctx_hint = (int) kv_cache->ne[3] * block_size;  // pool span upper bound
            const int nsm      = ggml_cuda_info().devices[ctx.device].nsm;
            // The old hard-coded 8 never adapted to the device or the head count, and it
            // was leaving 3.1x on the floor: measured at 22K context on one Blackwell card,
            // decode ms/tok by split count was 8 -> 36.4, 16 -> 21.2, 32 -> 13.7,
            // 48 -> 12.3, 64 -> 11.6, 96 -> 12.6. Each decode block does one token's work,
            // so it takes ~12 blocks/SM to saturate; past the knee the combine pass and the
            // shrinking per-block work take it back.
            const int fill = (12 * nsm) / std::max(1, n_heads * n_seq);
            const int cap  = std::max(1, ctx_hint / 256);   // >=256 keys per slice
            const int want = splitk_env > 0 ? splitk_env
                                            : std::max(1, std::min(std::min(fill, cap), 64));
            if (splitk_env > 0 || ctx_hint >= 4096) {
                n_splits = want;
            }
        }

        if (n_splits > 1) {
            const size_t n_part = (size_t) n_tokens_total * n_heads * n_splits;
            ggml_cuda_pool_alloc<float> part_acc(ctx.pool(), n_part * head_dim);
            ggml_cuda_pool_alloc<float> part_m(ctx.pool(), n_part);
            ggml_cuda_pool_alloc<float> part_l(ctx.pool(), n_part);

            paged_attention_decode_kernel<<<dim3(n_heads, n_seq, n_splits), dim3(head_dim), smem_bytes, ctx.stream()>>>(
                (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
                (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
                stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
                rel ? (const float *) rel->data : nullptr, rel_extent, visibility_window,
                part_acc.get(), n_splits, part_m.get(), part_l.get());

            paged_attention_combine_kernel<<<dim3(n_heads, n_tokens_total), dim3(head_dim), 0, ctx.stream()>>>(
                part_acc.get(), part_m.get(), part_l.get(), n_splits, (float *) dst->data);
            return;
        }

        paged_attention_decode_kernel<<<dim3(n_heads, n_seq, n_tokens_total), dim3(head_dim), smem_bytes, ctx.stream()>>>(
            (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
            (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
            stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
            rel ? (const float *) rel->data : nullptr, rel_extent, visibility_window, (float *) dst->data,
            /*n_splits =*/ 1, nullptr, nullptr);
        return;
    }

    // reference kernel fallback (head_dim > 128): correct but serial over context
    const size_t n_warps    = ((size_t) head_dim + 31) / 32;
    const size_t smem_bytes = n_warps * sizeof(float);

    // Manually request extended shared memory if needed (>48 KB)
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
    if (smem_bytes > 48 * 1024) {
        GGML_ASSERT(smem_bytes <= 96 * 1024 && "smem exceeds 96KB limit");
        CUDA_CHECK(cudaFuncSetAttribute(paged_attention_decode_kernel_ref, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int) smem_bytes));
    }

    // Read kernel - Grid (n_heads_kv, n_seq), Block (head_dim)
    paged_attention_decode_kernel_ref<<<dim3(n_heads, n_seq), dim3(head_dim), smem_bytes, ctx.stream()>>>(
        (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
        (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
        stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
        rel ? (const float *) rel->data : nullptr, rel_extent, visibility_window, (float *) dst->data);
}
