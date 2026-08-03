// tests/test-paged-banded.cpp
//
// Equivalence witness for ggml_paged_attn_banded (3b part 2b): the paged banded path must
// match ggml_flash_attn_ext_banded (the arc-1/2/3-gated reference) on identical logical
// content. Single sequence, N tokens spanning TWO blocks (block iteration exercised), GQA
// heads, all tokens written this call (empty cache at start, write_slots = identity).
//
// Cases:
//   A: rel bias active, paged window=0 (implicit causal) vs FA analytic window=N (pure causal)
//   B: rel bias active, window=8 cutoff inside the context on both sides
//   C: rel=null on the paged side vs FA with a ZERO rel tensor, window=8
//
// The paged kernel rounds K/V through the F16 cache; the FA reference computes with F32 K/V
// here, so equality is toleranced (F16 rounding floor), not bit-exact.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float val_q(int tok, int h, int d)  { return 0.25f * sinf(0.7f*tok + 1.3f*h + 0.11f*d); }
static float val_k(int tok, int h, int d)  { return 0.25f * cosf(0.3f*tok + 0.9f*h + 0.07f*d); }
static float val_v(int tok, int h, int d)  { return 0.25f * sinf(0.5f*tok + 0.4f*h + 0.13f*d + 1.0f); }
static float val_r(int e, int h, int tok)  { return 0.50f * cosf(0.8f*e + 0.6f*h + 0.21f*tok); }

// head_dim is a PARAMETER, not a constant: the CUDA mma prefill instantiates 64 and 128
// separately, and 128 is what every real model uses (qwen3, GLM, llama). Testing only 64
// meant the shipping instantiation had no equivalence coverage at all.
static int run_cases(const int D, ggml_backend_t backend) {
    const int H    = 4;   // query heads
    const int HKV  = 2;   // kv heads (GQA 2:1)
    const int N    = 24;  // tokens, spans two blocks
    const int E    = 8;   // rel_extent
    const int BS   = 16;  // block_size
    const int NB   = 2;   // blocks
    const float scale = 1.0f / sqrtf((float) D);

    int n_fail = 0;

    for (int cse = 0; cse < 3; ++cse) {
        const bool     with_rel     = cse != 2;
        const int64_t  paged_window = (cse == 0) ? 0 : 8;
        const int64_t  fa_window    = (cse == 0) ? N : 8;

        ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
        ggml_context * ctx = ggml_init(ip);

        // ---- paged side ----
        ggml_tensor * q_p   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,  N);
        ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
        ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
        // cache layout per the CPU kernel: [D, block_size, 2*HKV (K then V), n_blocks] F16
        ggml_tensor * cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, BS, 2*HKV, NB);
        ggml_tensor * btab  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NB, 1);
        ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_tensor * clens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * boffs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * blens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor * rel_p = with_rel ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, H, N) : nullptr;

        ggml_tensor * out_p = ggml_paged_attn_banded(ctx, q_p, k_new, v_new, cache, cache,
                btab, slots, clens, boffs, blens, rel_p, scale, BS, NB,
                with_rel ? E : 1, paged_window);
        // rel==null requires rel_extent>0 for the ctor? plain path ignores it; pass 1
        ggml_set_name(out_p, "out_paged");

        // ---- FA side (reference) ----
        ggml_tensor * q_f   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, H, 1);
        ggml_tensor * k_f   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, HKV, 1);
        ggml_tensor * v_f   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, N, HKV, 1);
        ggml_tensor * rel_f = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, E, H, N, 1); // zero when !with_rel
        ggml_tensor * out_f = ggml_flash_attn_ext_banded(ctx, q_f, k_f, v_f, /*mask*/ nullptr,
                rel_f, scale, E, fa_window);
        ggml_flash_attn_ext_set_prec(out_f, GGML_PREC_F32);
        ggml_set_name(out_f, "out_fa");

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf);

        // ---- fills (same logical values in each side's layout) ----
        std::vector<float> tmp;
        tmp.resize((size_t) N*H*D);
        for (int t = 0; t < N; ++t) for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d)
            tmp[(size_t) t*H*D + h*D + d] = val_q(t, h, d);
        ggml_backend_tensor_set(q_p, tmp.data(), 0, tmp.size()*sizeof(float));
        for (int h = 0; h < H; ++h) for (int t = 0; t < N; ++t) for (int d = 0; d < D; ++d)
            tmp[(size_t) h*N*D + t*D + d] = val_q(t, h, d);
        ggml_backend_tensor_set(q_f, tmp.data(), 0, tmp.size()*sizeof(float));

        tmp.resize((size_t) N*HKV*D);
        for (int t = 0; t < N; ++t) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d)
            tmp[(size_t) t*HKV*D + h*D + d] = val_k(t, h, d);
        ggml_backend_tensor_set(k_new, tmp.data(), 0, tmp.size()*sizeof(float));
        for (int h = 0; h < HKV; ++h) for (int t = 0; t < N; ++t) for (int d = 0; d < D; ++d)
            tmp[(size_t) h*N*D + t*D + d] = val_k(t, h, d);
        ggml_backend_tensor_set(k_f, tmp.data(), 0, tmp.size()*sizeof(float));

        for (int t = 0; t < N; ++t) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d)
            tmp[(size_t) t*HKV*D + h*D + d] = val_v(t, h, d);
        ggml_backend_tensor_set(v_new, tmp.data(), 0, tmp.size()*sizeof(float));
        for (int h = 0; h < HKV; ++h) for (int t = 0; t < N; ++t) for (int d = 0; d < D; ++d)
            tmp[(size_t) h*N*D + t*D + d] = val_v(t, h, d);
        ggml_backend_tensor_set(v_f, tmp.data(), 0, tmp.size()*sizeof(float));

        // FA reference computes K/V through the F16 rounding the paged cache applies, so the
        // two sides share the same quantization floor -> compare tight
        {
            std::vector<float>        raw((size_t) N*HKV*D);
            std::vector<ggml_fp16_t>  h16((size_t) N*HKV*D);
            std::vector<float>        rounded((size_t) N*HKV*D);
            for (int h = 0; h < HKV; ++h) for (int t = 0; t < N; ++t) for (int d = 0; d < D; ++d)
                raw[(size_t) h*N*D + t*D + d] = val_k(t, h, d);
            ggml_fp32_to_fp16_row(raw.data(), h16.data(), raw.size());
            ggml_fp16_to_fp32_row(h16.data(), rounded.data(), raw.size());
            ggml_backend_tensor_set(k_f, rounded.data(), 0, rounded.size()*sizeof(float));
            for (int h = 0; h < HKV; ++h) for (int t = 0; t < N; ++t) for (int d = 0; d < D; ++d)
                raw[(size_t) h*N*D + t*D + d] = val_v(t, h, d);
            ggml_fp32_to_fp16_row(raw.data(), h16.data(), raw.size());
            ggml_fp16_to_fp32_row(h16.data(), rounded.data(), raw.size());
            ggml_backend_tensor_set(v_f, rounded.data(), 0, rounded.size()*sizeof(float));
        }

        tmp.assign((size_t) E*H*N, 0.0f);
        if (with_rel) {
            for (int t = 0; t < N; ++t) for (int h = 0; h < H; ++h) for (int e = 0; e < E; ++e)
                tmp[(size_t) t*H*E + h*E + e] = val_r(e, h, t);
            ggml_backend_tensor_set(rel_p, tmp.data(), 0, tmp.size()*sizeof(float));
        }
        // FA rel layout [E, H, n_q]: same linear order as the paged [E, H, N]
        ggml_backend_tensor_set(rel_f, tmp.data(), 0, tmp.size()*sizeof(float));

        std::vector<int32_t> i32(N);
        for (int t = 0; t < N; ++t) i32[t] = t;               // identity write slots
        ggml_backend_tensor_set(slots, i32.data(), 0, N*sizeof(int32_t));
        i32 = {0, 1};
        ggml_backend_tensor_set(btab, i32.data(), 0, 2*sizeof(int32_t));
        i32 = {N};
        ggml_backend_tensor_set(clens, i32.data(), 0, sizeof(int32_t));
        i32 = {0};
        ggml_backend_tensor_set(boffs, i32.data(), 0, sizeof(int32_t));
        i32 = {N};
        ggml_backend_tensor_set(blens, i32.data(), 0, sizeof(int32_t));

        std::vector<uint8_t> zeros(ggml_nbytes(cache), 0);
        ggml_backend_tensor_set(cache, zeros.data(), 0, zeros.size());

        // ---- compute both graphs ----
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_p);
        ggml_build_forward_expand(gf, out_f);
        ggml_backend_graph_compute(backend, gf);

        // ---- compare: paged out [D,H,N] linear == FA out [D,H,N,1] linear ----
        std::vector<float> op(ggml_nelements(out_p)), of(ggml_nelements(out_f));
        ggml_backend_tensor_get(out_p, op.data(), 0, ggml_nbytes(out_p));
        ggml_backend_tensor_get(out_f, of.data(), 0, ggml_nbytes(out_f));
        GGML_ASSERT(op.size() == of.size());

        double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
        for (size_t i = 0; i < op.size(); ++i) {
            const double d = (double) op[i] - of[i];
            max_abs = fabs(d) > max_abs ? fabs(d) : max_abs;
            sum_sq += d*d;
            ref_sq += (double) of[i]*of[i];
        }
        const double nmse = ref_sq > 0 ? sum_sq/ref_sq : sum_sq;
        // shared F16 K/V floor; the paged side divides by (exp_sum + 1e-6) -> not bit-exact
        const bool ok = max_abs < 2e-3 && nmse < 1e-6;

        printf("case %c: with_rel=%d paged_window=%lld fa_window=%lld max_abs=%.3e nmse=%.3e %s\n",
               'A' + cse, with_rel ? 1 : 0, (long long) paged_window, (long long) fa_window,
               max_abs, nmse, ok ? "PASS" : "FAIL");
        n_fail += ok ? 0 : 1;

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    return n_fail;
}

int main() {
    // best available backend: CUDA on the box, CPU on the Mac build (both paths get covered
    // by running the binary on each machine)
    ggml_backend_t backend = ggml_backend_init_best();
    GGML_ASSERT(backend);
    printf("backend: %s\n", ggml_backend_name(backend));

    int n_fail = 0;
    const int dims[] = { 64, 128 };
    for (int i = 0; i < 2; ++i) {
        printf("== head_dim %d ==\n", dims[i]);
        n_fail += run_cases(dims[i], backend);
    }

    ggml_backend_free(backend);

    printf(n_fail == 0 ? "test-paged-banded: ALL PASSED\n" : "test-paged-banded: %d FAILED\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
