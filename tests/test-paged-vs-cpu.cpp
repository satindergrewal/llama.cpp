// test-paged-vs-cpu — verify GGML_OP_PAGED_ATTN on the best available backend against the
// CPU implementation of the SAME op.
//
// Why this exists (measured 2026-08-04): test-paged-banded compares paged against
// FLASH_ATTN_EXT_BANDED, and that reference is CUDA-only and static_asserts
// D == 64 || D == 128 (fattn-banded.cu:80). That single dependency blocked two separate
// items at once:
//   - head_dim 96/192 mma instantiations could not be verified (nothing to compare to), so
//     an otherwise-ready 96 kernel was reverted rather than shipped unverified;
//   - a Metal paged port could not be gated at all, since the reference does not exist there.
// Comparing a backend against the CPU implementation of the same op has neither
// restriction: any head_dim, any backend. One harness, both blockers.
//
// The CPU path is the definition of correct here, so this is a port-verification tool, not
// a numerical-accuracy claim about the op itself — that is what test-paged-banded is for on
// the head_dims it can reach.

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

// Run the paged op once on `backend` and return the output, so the caller can run the same
// thing twice and diff it.
static std::vector<float> run_paged(ggml_backend_t backend, int D, bool with_rel, int64_t window) {
    const int H   = 4;    // query heads
    const int HKV = 2;    // kv heads (GQA 2:1)
    const int N   = 24;   // tokens, spans two blocks
    const int E   = 8;    // rel_extent
    const int BS  = 16;   // block_size
    const int NB  = 2;    // blocks
    const float scale = 1.0f / sqrtf((float) D);

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q_p   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,   N);
    ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
    ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
    ggml_tensor * cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, BS, 2*HKV, NB);
    ggml_tensor * btab  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NB, 1);
    ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * clens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * boffs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * blens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * rel_p = with_rel ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, H, N) : nullptr;

    ggml_tensor * out_p = ggml_paged_attn_banded(ctx, q_p, k_new, v_new, cache, cache,
            btab, slots, clens, boffs, blens, rel_p, scale, BS, NB, with_rel ? E : 1, window);
    ggml_set_name(out_p, "out_paged");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> tmp((size_t) N*H*D);
    for (int t = 0; t < N; ++t) for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d)
        tmp[(size_t) t*H*D + h*D + d] = val_q(t, h, d);
    ggml_backend_tensor_set(q_p, tmp.data(), 0, tmp.size()*sizeof(float));

    tmp.resize((size_t) N*HKV*D);
    for (int t = 0; t < N; ++t) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d)
        tmp[(size_t) t*HKV*D + h*D + d] = val_k(t, h, d);
    ggml_backend_tensor_set(k_new, tmp.data(), 0, tmp.size()*sizeof(float));
    for (int t = 0; t < N; ++t) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d)
        tmp[(size_t) t*HKV*D + h*D + d] = val_v(t, h, d);
    ggml_backend_tensor_set(v_new, tmp.data(), 0, tmp.size()*sizeof(float));

    if (with_rel) {
        std::vector<float> r((size_t) E*H*N, 0.0f);
        for (int t = 0; t < N; ++t) for (int h = 0; h < H; ++h) for (int e = 0; e < E; ++e)
            r[(size_t) t*H*E + h*E + e] = val_r(e, h, t);
        ggml_backend_tensor_set(rel_p, r.data(), 0, r.size()*sizeof(float));
    }

    std::vector<int32_t> i32(N);
    for (int t = 0; t < N; ++t) i32[t] = t;               // identity write slots
    ggml_backend_tensor_set(slots, i32.data(), 0, N*sizeof(int32_t));
    i32 = {0, 1};  ggml_backend_tensor_set(btab,  i32.data(), 0, 2*sizeof(int32_t));
    i32 = {N};     ggml_backend_tensor_set(clens, i32.data(), 0, sizeof(int32_t));
    i32 = {0};     ggml_backend_tensor_set(boffs, i32.data(), 0, sizeof(int32_t));
    i32 = {N};     ggml_backend_tensor_set(blens, i32.data(), 0, sizeof(int32_t));

    std::vector<uint8_t> zeros(ggml_nbytes(cache), 0);
    ggml_backend_tensor_set(cache, zeros.data(), 0, zeros.size());

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_p);
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(ggml_nelements(out_p));
    ggml_backend_tensor_get(out_p, out.data(), 0, ggml_nbytes(out_p));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

int main() {
    ggml_backend_t backend = ggml_backend_init_best();
    GGML_ASSERT(backend);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);

    const bool same = strcmp(ggml_backend_name(backend), ggml_backend_name(cpu)) == 0;
    printf("backend: %s   reference: %s%s\n", ggml_backend_name(backend), ggml_backend_name(cpu),
           same ? "   (identical -- this run only proves the harness, not a port)" : "");

    // head_dims the mma prefill instantiates or might: 64 and 128 ship today; 96/192 are the
    // ones test-paged-banded cannot reach, which is the whole reason this file exists.
    const int dims[] = { 64, 96, 128, 192 };
    int n_fail = 0;

    for (int di = 0; di < 4; ++di) {
        const int D = dims[di];
        for (int cse = 0; cse < 3; ++cse) {
            const bool    with_rel = cse != 2;
            const int64_t window   = (cse == 0) ? 0 : 8;

            const std::vector<float> a = run_paged(backend, D, with_rel, window);
            const std::vector<float> b = run_paged(cpu,     D, with_rel, window);
            GGML_ASSERT(a.size() == b.size());

            double max_abs = 0.0, sum_sq = 0.0, ref_sq = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = (double) a[i] - b[i];
                max_abs = fabs(d) > max_abs ? fabs(d) : max_abs;
                sum_sq += d*d;
                ref_sq += (double) b[i]*b[i];
            }
            const double nmse = ref_sq > 0 ? sum_sq/ref_sq : sum_sq;
            const bool ok = max_abs < 2e-3 && nmse < 1e-6;

            printf("D=%3d case %c: with_rel=%d window=%lld max_abs=%.3e nmse=%.3e %s\n",
                   D, 'A' + cse, with_rel ? 1 : 0, (long long) window, max_abs, nmse,
                   ok ? "PASS" : "FAIL");
            n_fail += ok ? 0 : 1;
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(backend);

    printf(n_fail == 0 ? "test-paged-vs-cpu: ALL PASSED\n" : "test-paged-vs-cpu: %d FAILED\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
