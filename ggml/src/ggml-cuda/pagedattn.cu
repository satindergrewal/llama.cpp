#include "pagedattn.cuh"

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
__device__ __forceinline__ void paged_wmma_qk(const half * __restrict__ q_h,
                                              const half * __restrict__ k_h,
                                              const int    ld,          // head_dim + pad
                                              const int    head_dim,
                                              const float  scale,
                                              float * __restrict__ scores_out) {  // [M*N] row-major
    wmma::fragment<wmma::accumulator, PAGED_WMMA_M, PAGED_WMMA_N, PAGED_WMMA_K, float> frag_acc;
    wmma::fill_fragment(frag_acc, 0.0f);

    for (int c = 0; c < head_dim; c += PAGED_WMMA_K) {
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

// stage 3a: P x V for one tile. After the online softmax turns the score tile into
// probabilities, P (M x N_keys, half) times V (N_keys x head_dim, half) accumulates into
// the output fragment -- one f32 accumulator per 16-wide slice of head_dim. Both operands
// are row_major here: the K dimension of the product is N_keys (the tile's keys), which is
// V's leading axis, so no transpose is needed on this side either.
// out_acc is [M][ld_out] f32 and is ACCUMULATED into, because a query tile walks many key
// tiles and the online-softmax rescaling is applied by the caller between tiles.
__device__ __forceinline__ void paged_wmma_pv(const half * __restrict__ p_h,     // [M][N] row-major
                                              const half * __restrict__ v_h,     // [N][ld_v] row-major
                                              const int    ld_v,
                                              const int    head_dim,
                                              float * __restrict__ out_acc,      // [M][ld_out]
                                              const int    ld_out) {
    for (int c = 0; c < head_dim; c += PAGED_WMMA_N) {
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
            const int want     = splitk_env > 0 ? splitk_env : 8;
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
