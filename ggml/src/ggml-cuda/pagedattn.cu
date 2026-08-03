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
                                              float * __restrict__ out) {
    extern __shared__ float smem[];

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;
    const int lane     = tid & 31;
    const int warp_id  = tid >> 5;

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
        const int i = blockIdx.z;
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

        for (int token = lo + warp_id; token < n_tok; token += n_warps) {
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

        const int out_idx = (size_t) token_batch_idx * n_heads * head_dim + (size_t) head_idx * head_dim + tid;
        out[out_idx] = out_acc / (l_tot + 1e-6f);
        __syncthreads();  // q_s / warp_acc are rewritten next iteration
    }
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

        paged_attention_decode_kernel<<<dim3(n_heads, n_seq, n_tokens_total), dim3(head_dim), smem_bytes, ctx.stream()>>>(
            (const float *) q->data, (const half *) kv_cache->data, (const int *) block_table->data,
            (const int *) context_lens->data, (const int *) batch_offsets->data, (const int *) batch_lens->data,
            stride_token, stride_head, stride_block, n_heads_kv, block_size, max_blocks, scale,
            rel ? (const float *) rel->data : nullptr, rel_extent, visibility_window, (float *) dst->data);
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
