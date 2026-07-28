//
// Sparse ("DSA") flash attention: gather the indexer-selected top-k K/V rows and
// run attention only over those, instead of a full-size KQ with a sparse mask.
//
// SPDX-License-Identifier: MIT
//
// The kernels and the batched-cuBLAS pipeline in this file are transplanted from
// ik_llama.cpp, ggml/src/ggml-cuda/dsa_attn.cu -- Copyright (C) 2024 Iwan Kawrakow,
// MIT license. Adapted to llama.cpp conventions (op plumbing via
// GGML_OP_FLASH_ATTN_EXT_DSA, ggml_cuda_pool_alloc, ctx.stream(), CUDA_CHECK /
// CUBLAS_CHECK, explicit cublasSetStream, per-launch error checks).
//

#include "fattn-dsa.cuh"

#include <algorithm>
#include <cstdlib>
#include <cstring>

// The gathered path is built on batched cuBLAS calls that have no HIP/MUSA mapping
// in ggml's vendor headers, so it is CUDA-only. Other backends report the
// op as unsupported and the entry point is a no-op that returns false.
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_flash_attn_ext_dsa_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
}

bool ggml_cuda_flash_attn_ext_dsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    return false;
}

#else

// V is frequently a view into the K rows (MLA-style: V = leading DV elements of each
// K row). In that case the gather for K already produced V, so it can be skipped.
static inline bool dsa_v_is_k_view(const ggml_tensor * K, const ggml_tensor * V) {
    if (!V || !V->data) {
        return false;
    }
    const char * k_data = (const char *) K->data;
    const char * v_data = (const char *) V->data;
    const size_t k_row_size = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row_size = ggml_row_size(V->type, V->ne[0]);
    // the reuse path indexes the gathered K buffer with K's row stride, so an
    // aliasing V whose stride differs would silently read the wrong rows
    if (V->nb[1] != K->nb[1]) {
        return false;
    }
    return v_data >= k_data && v_data + v_row_size <= k_data + k_row_size;
}

// gather the mask entries that correspond to the selected keys: [n_topk, n_tokens]
static __global__ void k_dsa_prepare_mask(int nidx, const int * __restrict__ idx, const half * __restrict__ m_in,
        half * __restrict__ m_out, size_t stride_idx, size_t stride_m) {
    int row = blockIdx.x;
    int col = blockIdx.y*blockDim.x + threadIdx.x;
    idx += row*stride_idx;
    m_out[row*nidx + col] = m_in[row*stride_m + idx[col]];
}

// gather the selected K (or V) rows into a compact [nk, ncol, nrows] f16 buffer
static __global__ void k_dsa_prepare_one_batch_kv(int nk, int ncol, const int * idx, const char * k_in,
        half * k_out, size_t stride_k, size_t stride_idx) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int i = idx[row*stride_idx + col];
    const half * k_row = (const half *)(k_in + stride_k * i);
    k_out += (row*ncol + col)*nk;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        k_out[j] = k_row[j];
    }
}

// f32 -> f16 Q, de-permuted into [ne0, n_head, nrows]
static __global__ void k_dsa_prepare_one_batch_q(int ne0, int ne1, size_t nb1, size_t nb2,
        const float * q_in, half * q_out) {
    int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    int i1 = blockIdx.y;
    int i2 = blockIdx.z;
    q_out[i0 + (i2 + i1*ne1)*ne0] = __float2half(q_in[i0 + i1*nb1 + i2*nb2]);
}

static __global__ void k_dsa_copy_dst(int nelem, const half * kqv16, float * dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    dst[i] = __half2float(kqv16[i]);
}

template <int ncols_template, int block_size_template>
static __global__ void dsa_soft_max_f16(half * x, const half * mask, const int ncols_par, const int nrows_y, const float scale) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx / nrows_y; // broadcast the mask in the row dimension

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float data_dsa_soft_max_f32[];
    float * buf_iw = data_dsa_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = buf_iw + WARP_SIZE;

    float max_val = -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = scale*__half2float(x[ix]) + __half2float(mask[iy]);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        x[ix] = __float2half(vals[col] * inv_sum);
    }
}

#define DSA_SOFT_MAX_BLOCK_SIZE 1024

// The softmax stages an entire top_k row in shared memory, so top_k is bounded by the
// shared memory per block of the device that will run it. This is the single definition
// of that requirement: the launcher asserts on it and dsa_attn_layout_ok() rejects on it,
// so supports_op can never accept a shape the launcher would then abort on, and the two
// cannot drift apart.
static inline size_t dsa_soft_max_shmem(int64_t ncols_x) {
    return (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
}

static void dsa_soft_max_f16_cuda(half * x, const half * mask, const int ncols_x, const int nrows_x,
        const int nrows_y, const float scale, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < DSA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = dsa_soft_max_shmem(ncols_x);
    static_assert(DSA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    GGML_ASSERT(shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    switch (ncols_x) {
        case 32:
            dsa_soft_max_f16<32, 32><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 64:
            dsa_soft_max_f16<64, 64><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 128:
            dsa_soft_max_f16<128, 128><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 256:
            dsa_soft_max_f16<256, 256><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 512:
            dsa_soft_max_f16<512, 512><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 1024:
            dsa_soft_max_f16<1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 2048:
            dsa_soft_max_f16<2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        case 4096:
            dsa_soft_max_f16<4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
        default:
            dsa_soft_max_f16<0, 0><<<block_nums, block_dims, shmem, stream>>>(x, mask, ncols_x, nrows_y, scale);
            break;
    }
}

// Both matmuls feed fp16 A/B and store fp16 C. cublasHgemmStridedBatched also ACCUMULATES
// in fp16, which puts a rounding floor on the result that grows with the reduction length
// (~sqrt(k) for the PV gemm, where k = top_k). LLAMA_DSA_F32ACC=1 switches to
// cublasGemmStridedBatchedEx with CUBLAS_COMPUTE_32F: identical shapes, strides, batch
// count, transposes and stream, identical fp16 storage for A/B/C, but the dot products
// accumulate in fp32. Default is on; LLAMA_DSA_F32ACC=0 restores the original
// fp16-accumulate path bit-for-bit.
//
// Note this fixes accumulation only. The KQ and KQV buffers are still fp16, so a rounding
// floor from those stores remains either way.
static bool dsa_f32_acc_enabled() {
    static const bool enabled = []() {
        // default ON: measured 1.9-3.6x error reduction (rel_L2 flat ~3.1e-4 across
        // top_k, better than stock FA vs the same fp64 reference) at -0.8% median
        // throughput on the box. LLAMA_DSA_F32ACC=0 restores fp16 accumulate.
        const char * s = getenv("LLAMA_DSA_F32ACC");
        return s == nullptr || s[0] == '\0' || s[0] != '0';
    }();
    return enabled;
}

static inline cublasStatus_t dsa_gemm_strided_batched(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const half * A, int lda, long long int strideA,
        const half * B, int ldb, long long int strideB,
        half       * C, int ldc, long long int strideC,
        int batch_count) {
    if (dsa_f32_acc_enabled()) {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k,
                &alpha, A, CUDA_R_16F, lda, strideA,
                        B, CUDA_R_16F, ldb, strideB,
                &beta,  C, CUDA_R_16F, ldc, strideC,
                batch_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    const half alpha = 1.0f;
    const half beta  = 0.0f;
    return cublasHgemmStridedBatched(handle, transa, transb, m, n, k,
            &alpha, A, lda, strideA,
                    B, ldb, strideB,
            &beta,  C, ldc, strideC, batch_count);
}

// ik's guard logic, kept as-is: any miss means "not handled here" and must be safe.
static bool dsa_attn_layout_ok(int device, const ggml_tensor * dst) {
    if (!dst) {
        return false;
    }
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_DSA || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K       = dst->src[1];
    const ggml_tensor * V       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sink    = dst->src[4];
    const ggml_tensor * indexer = dst->src[5];

    if (sink) return false; // sinks are not supported by this path
    if (!Q || !K || !V || !mask || !indexer) return false;

    if (indexer->ne[0] % 256 != 0) return false; // no tail handling for top_k not a multiple of 256
    if (K->ne[1] < 4*indexer->ne[0]) return false; // gathering only pays off when the cache is much larger than top_k
    if (K->ne[2] > 1 || K->ne[3] > 1 || mask->ne[2] > 1 || mask->ne[3] > 1 || Q->ne[3] > 1) return false;
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 || Q->type != GGML_TYPE_F32) return false;
    if (K->ne[0] != Q->ne[0]) return false;

    // not in ik's original, added because this path indexes V and the top-k tensor directly:
    if (V->ne[2] > 1 || V->ne[3] > 1) return false;
    if (indexer->type != GGML_TYPE_I32) return false;
    if (indexer->ne[1] < Q->ne[1] || indexer->ne[2] > 1 || indexer->ne[3] > 1) return false;

    // the softmax cannot stage a top_k row larger than the device allows. Without this the
    // shape is ACCEPTED here and then hard-aborts inside dsa_soft_max_f16_cuda, which kills
    // the process instead of falling back to the dense path.
    if (device < 0 || device >= ggml_cuda_info().device_count) return false;
    if (!(dsa_soft_max_shmem(indexer->ne[0]) < ggml_cuda_info().devices[device].smpb)) return false;

    return true;
}

bool ggml_cuda_flash_attn_ext_dsa_supported(int device, const ggml_tensor * dst) {
    return dsa_attn_layout_ok(device, dst);
}

bool ggml_cuda_flash_attn_ext_dsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!dsa_attn_layout_ok(ctx.device, dst)) {
        return false;
    }

    constexpr int k_max_rows = 32;

    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K       = dst->src[1];
    const ggml_tensor * V       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * indexer = dst->src[5];

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int  max_rows   = std::min<int>(Q->ne[1], k_max_rows);
    const bool is_k_view  = dsa_v_is_k_view(K, V);

    // the mask is relatively small, so it is gathered once for the whole calculation
    const int64_t mask_size   = indexer->ne[0]*Q->ne[1];
    const int64_t k_cache_size = indexer->ne[0]*K->ne[0]*max_rows;
    const int64_t v_cache_size = indexer->ne[0]*V->ne[0]*max_rows;
    const int64_t q_size      = Q->ne[0]*Q->ne[2]*max_rows;
    const int64_t kq_size     = indexer->ne[0]*Q->ne[2]*max_rows;
    const int64_t kqv_size    = V->ne[0]*Q->ne[2]*max_rows;

    ggml_cuda_pool_alloc<half> q16   (ctx.pool(), q_size);
    ggml_cuda_pool_alloc<half> kq16  (ctx.pool(), kq_size);
    ggml_cuda_pool_alloc<half> kqv16 (ctx.pool(), kqv_size);
    ggml_cuda_pool_alloc<half> mask16(ctx.pool(), mask_size);
    ggml_cuda_pool_alloc<half> k16   (ctx.pool(), k_cache_size);
    ggml_cuda_pool_alloc<half> v16   (ctx.pool());

    size_t v_offset = 0;
    if (is_k_view) {
        v_offset = (const half *)V->data - (const half *)K->data;
    } else {
        v16.alloc(v_cache_size);
    }

    cudaStream_t stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));

    const size_t stride_idx = indexer->nb[1]/sizeof(int);
    {
        dim3 grid(Q->ne[1], indexer->ne[0]/256, 1);
        k_dsa_prepare_mask<<<grid, 256, 0, stream>>>(indexer->ne[0], (const int *)indexer->data,
                (const half *)mask->data, mask16.get(), stride_idx, mask->nb[1]/sizeof(half));
        CUDA_CHECK(cudaGetLastError());
    }

    const int nstep = (Q->ne[1] + max_rows - 1)/max_rows;

    for (int istep = 0; istep < nstep; ++istep) {
        const int first = istep*max_rows;
        const int last  = std::min<int>(first + max_rows, Q->ne[1]);
        const int nrows = last - first;
        {
            dim3 grid(indexer->ne[0], nrows, 1);
            k_dsa_prepare_one_batch_kv<<<grid, 256, 0, stream>>>(K->ne[0], indexer->ne[0],
                    (const int *)indexer->data + stride_idx*first,
                    (const char *)K->data, k16.get(), K->nb[1], stride_idx);
            CUDA_CHECK(cudaGetLastError());
            if (!is_k_view) {
                k_dsa_prepare_one_batch_kv<<<grid, 256, 0, stream>>>(V->ne[0], indexer->ne[0],
                        (const int *)indexer->data + stride_idx*first,
                        (const char *)V->data, v16.get(), V->nb[1], stride_idx);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        {
            const int nblock = (Q->ne[0] + 255)/256;
            dim3 grid(nblock, nrows, Q->ne[2]);
            k_dsa_prepare_one_batch_q<<<grid, 256, 0, stream>>>(Q->ne[0], Q->ne[2],
                    Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float),
                    (const float *)((const char *)Q->data + first*Q->nb[1]), q16.get());
            CUDA_CHECK(cudaGetLastError());
        }

        // KQ = K_gathered^T * Q, one gemm per query row in the chunk
        CUBLAS_CHECK(dsa_gemm_strided_batched(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    indexer->ne[0], Q->ne[2], Q->ne[0],
                    k16.get(), K->ne[0], K->ne[0]*indexer->ne[0],
                    q16.get(), Q->ne[0], Q->ne[0]*Q->ne[2],
                    kq16.get(), indexer->ne[0], indexer->ne[0]*Q->ne[2], nrows));

        dsa_soft_max_f16_cuda(kq16.get(), mask16.get() + first*indexer->ne[0], indexer->ne[0], Q->ne[2]*nrows,
                Q->ne[2], scale, stream);
        CUDA_CHECK(cudaGetLastError());

        if (is_k_view) {
            CUBLAS_CHECK(dsa_gemm_strided_batched(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        V->ne[0], Q->ne[2], indexer->ne[0],
                        k16.get() + v_offset, K->ne[0], K->ne[0]*indexer->ne[0],
                        kq16.get(), indexer->ne[0], indexer->ne[0]*Q->ne[2],
                        kqv16.get(), V->ne[0], V->ne[0]*Q->ne[2], nrows));
        } else {
            CUBLAS_CHECK(dsa_gemm_strided_batched(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        V->ne[0], Q->ne[2], indexer->ne[0],
                        v16.get(), V->ne[0], V->ne[0]*indexer->ne[0],
                        kq16.get(), indexer->ne[0], indexer->ne[0]*Q->ne[2],
                        kqv16.get(), V->ne[0], V->ne[0]*Q->ne[2], nrows));
        }

        {
            const int nelem  = V->ne[0]*Q->ne[2]*nrows;
            const int nblock = (nelem + 255)/256;
            k_dsa_copy_dst<<<nblock, 256, 0, stream>>>(nelem, kqv16.get(),
                    (float *)((char *)dst->data + dst->nb[2]*first));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return true;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
