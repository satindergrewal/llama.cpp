#include "sinkhorn-norm.cuh"

// Fused Sinkhorn-Knopp normalization for DeepSeek-V4 mHC hyper-connections.
//
// The input is a stack of small square [n, n] matrices (one per token). The
// reference implementation decomposes this into ~137 ggml ops per layer
// (softmax + repeated sum_rows / div / transpose), which on small n (== 4)
// is pure kernel-launch + memory-round-trip overhead. This kernel does the
// entire per-slice normalization in registers with a single launch.

#define SINKHORN_NORM_MAX_N 8

template <int MAXN>
static __global__ void sinkhorn_norm_f32(
        const float * __restrict__ x,
        float       * __restrict__ y,
        const int     n,
        const int     n_iters,
        const float   eps,
        const int64_t n_slices,
        const int64_t slice_stride) {

    const int64_t s = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slices) {
        return;
    }

    const float * xs = x + s * slice_stride;
    float       * ys = y + s * slice_stride;

    float m[MAXN * MAXN];
    const int nn = n * n;

    for (int i = 0; i < nn; ++i) {
        m[i] = xs[i];
    }

    // softmax over ne0 (a) for each b (with max subtraction, matches ggml_soft_max)
    for (int b = 0; b < n; ++b) {
        float mx = -INFINITY;
        for (int a = 0; a < n; ++a) {
            mx = fmaxf(mx, m[b*n + a]);
        }
        float sum = 0.0f;
        for (int a = 0; a < n; ++a) {
            const float e = expf(m[b*n + a] - mx);
            m[b*n + a] = e;
            sum += e;
        }
        for (int a = 0; a < n; ++a) {
            m[b*n + a] /= sum;
        }
    }

    for (int i = 0; i < nn; ++i) {
        m[i] += eps;
    }

    // norm_cols: divide each element by (sum over b of its row a) + eps
    // norm_rows: divide each element by (sum over a of its column b) + eps
    #define HC_NORM_COLS() do {                          \
        for (int a = 0; a < n; ++a) {                    \
            float r = 0.0f;                              \
            for (int b = 0; b < n; ++b) r += m[b*n + a]; \
            r += eps;                                    \
            for (int b = 0; b < n; ++b) m[b*n + a] /= r; \
        }                                                \
    } while (0)

    #define HC_NORM_ROWS() do {                          \
        for (int b = 0; b < n; ++b) {                    \
            float c = 0.0f;                              \
            for (int a = 0; a < n; ++a) c += m[b*n + a]; \
            c += eps;                                    \
            for (int a = 0; a < n; ++a) m[b*n + a] /= c; \
        }                                                \
    } while (0)

    HC_NORM_COLS();
    for (int it = 1; it < n_iters; ++it) {
        HC_NORM_ROWS();
        HC_NORM_COLS();
    }

    #undef HC_NORM_COLS
    #undef HC_NORM_ROWS

    for (int i = 0; i < nn; ++i) {
        ys[i] = m[i];
    }
}

// Warp-cooperative variant: one lane per matrix element, NN = N*N lanes per
// square [N, N] slice (N a power of two, NN <= physical warp size). Global
// loads/stores are fully coalesced and every reduction is a segmented butterfly
// __shfl_xor over the NN-lane group:
//   a = lane % N  -> bits [0 .. log2(N))      (reduced with offsets 1..N/2)
//   b = lane / N  -> bits [log2(N) .. 2log2(N)) (reduced with offsets N..N*N/2)
template <int N>
static __global__ void sinkhorn_norm_warp_f32(
        const float * __restrict__ x,
        float       * __restrict__ y,
        const int     n_iters,
        const float   eps,
        const int64_t n_slices) {

    constexpr int NN = N * N;

    const int64_t g     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t token = g / NN;
    if (token >= n_slices) {
        return;
    }

    const int e = threadIdx.x % NN; // element within slice (== lane % NN)

    float v = x[token * NN + e];    // coalesced: consecutive lanes -> consecutive addrs

    // softmax over a (fixed b): max then sum via butterfly over the a-bits
    float mx = v;
#pragma unroll
    for (int off = 1; off < N; off <<= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, NN));
    }
    float ex  = expf(v - mx);
    float sum = ex;
#pragma unroll
    for (int off = 1; off < N; off <<= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off, NN);
    }
    v = ex / sum + eps;

    // norm_cols: divide by (sum over b of row a) + eps -> reduce over b-bits
    {
        float r = v;
#pragma unroll
        for (int off = N; off < NN; off <<= 1) {
            r += __shfl_xor_sync(0xffffffff, r, off, NN);
        }
        v /= (r + eps);
    }

    for (int it = 1; it < n_iters; ++it) {
        // norm_rows: divide by (sum over a of column b) + eps -> reduce over a-bits
        {
            float c = v;
#pragma unroll
            for (int off = 1; off < N; off <<= 1) {
                c += __shfl_xor_sync(0xffffffff, c, off, NN);
            }
            v /= (c + eps);
        }
        // norm_cols
        {
            float r = v;
#pragma unroll
            for (int off = N; off < NN; off <<= 1) {
                r += __shfl_xor_sync(0xffffffff, r, off, NN);
            }
            v /= (r + eps);
        }
    }

    y[token * NN + e] = v;
}

static bool sinkhorn_norm_is_pow2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

void ggml_cuda_op_sinkhorn_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float       * dst_d  = (float *)       dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_TENSOR_UNARY_OP_LOCALS;

    const int32_t n_iters = ggml_get_op_params_i32(dst, 0);
    const float   eps     = ggml_get_op_params_f32(dst, 1);

    const int n = (int) ne00;
    GGML_ASSERT(ne01 == ne00);      // square [n, n] slices
    GGML_ASSERT(n <= SINKHORN_NORM_MAX_N);
    GGML_ASSERT(n_iters >= 1);

    const int64_t n_slices     = ne02 * ne03;
    const int64_t slice_stride = (int64_t) n * n;

    const int block_size = 256;

    // Prefer the warp-cooperative (coalesced, butterfly-reduction) kernel when the
    // slice packs cleanly into a warp: N a power of two and N*N <= warp size.
    // Otherwise fall back to the generic one-thread-per-slice kernel.
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;

    if (sinkhorn_norm_is_pow2(n) && n*n <= warp_size) {
        const int64_t total_threads = n_slices * (int64_t) (n*n);
        const int64_t num_blocks    = (total_threads + block_size - 1) / block_size;
        switch (n) {
            case 2: sinkhorn_norm_warp_f32<2><<<num_blocks, block_size, 0, stream>>>(src0_d, dst_d, n_iters, eps, n_slices); break;
            case 4: sinkhorn_norm_warp_f32<4><<<num_blocks, block_size, 0, stream>>>(src0_d, dst_d, n_iters, eps, n_slices); break;
            case 8: sinkhorn_norm_warp_f32<8><<<num_blocks, block_size, 0, stream>>>(src0_d, dst_d, n_iters, eps, n_slices); break;
            default: GGML_ABORT("sinkhorn_norm: unsupported warp N=%d", n);
        }
        return;
    }

    const int64_t num_blocks = (n_slices + block_size - 1) / block_size;
    sinkhorn_norm_f32<SINKHORN_NORM_MAX_N><<<num_blocks, block_size, 0, stream>>>(
        src0_d, dst_d, n, n_iters, eps, n_slices, slice_stride);
}
