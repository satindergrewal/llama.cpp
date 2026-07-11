#include "dsv4-hc.cuh"

// CUDA kernels for the DeepSeek-V4 hyper-connection ops (from PR
// ggml-org/llama.cpp#23122, CPU reference by Chris Chuter). The decomposed
// graph builds these from ~36 full-tensor mul/add/concat nodes per layer;
// profiling shows V4 decode graphs at ~18K nodes/token with these chains as
// the largest contributor. One launch per op instead.
//
// Both kernels mirror the CPU reference loops exactly (same accumulation
// order). Strides are byte strides, taken from the ggml tensors, so permuted
// views work unchanged.

// out[d, t] = sum_h x[d, h, t] * w[h, t]
static __global__ void dsv4_hc_weighted_sum_f32(
        const char * __restrict__ x,
        const char * __restrict__ w,
        char       * __restrict__ dst,
        const int     n_embd,
        const int     n_hc,
        const int64_t n_tokens,
        const int64_t nb_x0, const int64_t nb_x1, const int64_t nb_x2,
        const int64_t nb_w0, const int64_t nb_w1,
        const int64_t nb_d0, const int64_t nb_d1) {

    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n_embd * n_tokens) {
        return;
    }

    const int64_t d = i % n_embd;
    const int64_t t = i / n_embd;

    float acc = 0.0f;
    for (int h = 0; h < n_hc; ++h) {
        const float xv = *(const float *) (x + d*nb_x0 + h*nb_x1 + t*nb_x2);
        const float wv = *(const float *) (w + h*nb_w0 + t*nb_w1);
        acc += xv * wv;
    }

    *(float *) (dst + d*nb_d0 + t*nb_d1) = acc;
}

// out[d, dst_hc, t] = blk[d, t] * post[dst_hc, t]
//                     + sum_src res[d, src, t] * comb[dst_hc, src, t]
// comb is read transposed (ne0 as dst_hc), matching the CPU reference.
static __global__ void dsv4_hc_expand_f32(
        const char * __restrict__ blk,
        const char * __restrict__ res,
        const char * __restrict__ post,
        const char * __restrict__ comb,
        char       * __restrict__ dst,
        const int     n_embd,
        const int     n_hc,
        const int64_t n_tokens,
        const int64_t nb_b0, const int64_t nb_b1,
        const int64_t nb_r0, const int64_t nb_r1, const int64_t nb_r2,
        const int64_t nb_p0, const int64_t nb_p1,
        const int64_t nb_c0, const int64_t nb_c1, const int64_t nb_c2,
        const int64_t nb_d0, const int64_t nb_d1, const int64_t nb_d2) {

    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n_embd * n_hc * n_tokens) {
        return;
    }

    const int64_t d      = i % n_embd;
    const int64_t tmp    = i / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t      = tmp / n_hc;

    const float bv = *(const float *) (blk  + d*nb_b0 + t*nb_b1);
    const float pv = *(const float *) (post + dst_hc*nb_p0 + t*nb_p1);

    float acc = bv * pv;
    for (int src = 0; src < n_hc; ++src) {
        const float cv = *(const float *) (comb + dst_hc*nb_c0 + src*nb_c1 + t*nb_c2);
        const float rv = *(const float *) (res  + d*nb_r0 + src*nb_r1 + t*nb_r2);
        acc += cv * rv;
    }

    *(float *) (dst + d*nb_d0 + dst_hc*nb_d1 + t*nb_d2) = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];

    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(w->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int     n_embd   = (int) dst->ne[0];
    const int     n_hc     = (int) x->ne[1];
    const int64_t n_tokens = dst->ne[1];

    const int64_t n_elem = (int64_t) n_embd * n_tokens;
    const int nth  = 256;
    const int64_t n_blk = (n_elem + nth - 1) / nth;

    dsv4_hc_weighted_sum_f32<<<n_blk, nth, 0, ctx.stream()>>>(
        (const char *) x->data, (const char *) w->data, (char *) dst->data,
        n_embd, n_hc, n_tokens,
        x->nb[0], x->nb[1], x->nb[2],
        w->nb[0], w->nb[1],
        dst->nb[0], dst->nb[1]);
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * blk  = dst->src[0];
    const ggml_tensor * res  = dst->src[1];
    const ggml_tensor * post = dst->src[2];
    const ggml_tensor * comb = dst->src[3];

    GGML_ASSERT(blk->type  == GGML_TYPE_F32);
    GGML_ASSERT(res->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int     n_embd   = (int) dst->ne[0];
    const int     n_hc     = (int) dst->ne[1];
    const int64_t n_tokens = dst->ne[2];

    const int64_t n_elem = (int64_t) n_embd * n_hc * n_tokens;
    const int nth  = 256;
    const int64_t n_blk = (n_elem + nth - 1) / nth;

    dsv4_hc_expand_f32<<<n_blk, nth, 0, ctx.stream()>>>(
        (const char *) blk->data, (const char *) res->data,
        (const char *) post->data, (const char *) comb->data, (char *) dst->data,
        n_embd, n_hc, n_tokens,
        blk->nb[0], blk->nb[1],
        res->nb[0], res->nb[1], res->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}
