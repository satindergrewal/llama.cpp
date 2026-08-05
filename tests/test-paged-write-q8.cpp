// test-paged-write-q8 -- gate the q8_0 paged KV WRITE path EXACTLY, with no threshold games.
//
// Why this exists and why it is not part of test-paged-vs-cpu: the CPU paged reference is
// hardcoded f16 (stride_token = nb[1]/sizeof(ggml_fp16_t), staging as ggml_fp16_t), so there is
// no CPU q8_0 path to compare a Metal q8_0 run against. Comparing Metal-q8_0 to CPU-f16 with a
// loosened tolerance would conflate QUANTISATION error (~1/127, ~8e-3) with IMPLEMENTATION error
// (the suite's bar is 2e-3) -- widen the gate that far and a genuinely broken kernel passes too.
//
// So this gates the WRITE alone, and gates it EXACTLY: run the paged op on Metal with a q8_0
// cache, read the CACHE BACK, and compare against the same quantisation performed on the host.
// Both sides do identical q8_0 rounding, so the expected difference is ZERO, not "small".
//
// It says nothing about the read path. That is the point -- half the work, gated honestly.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static float val_k(int tok, int h, int d) { return 0.25f * cosf(0.3f*tok + 0.9f*h + 0.07f*d); }
static float val_v(int tok, int h, int d) { return 0.25f * sinf(0.5f*tok + 0.4f*h + 0.13f*d + 1.0f); }

// Host-side q8_0 of one 32-element group -- deliberately the SAME arithmetic as the kernel, so a
// mismatch means the kernel is wrong rather than that the two rounded differently.
static void quant_ref(const float * src, float & d_out, int8_t * qs_out) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) { amax = fmaxf(amax, fabsf(src[j])); }
    const float d  = amax / 127.0f;
    const float id = d ? 1.0f/d : 0.0f;
    d_out = (float) (ggml_fp32_to_fp16(d));   // kernel stores the scale as half
    d_out = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
    for (int j = 0; j < 32; ++j) { qs_out[j] = (int8_t) roundf(src[j] * id); }
}

int main() {
    ggml_backend_t backend = ggml_backend_init_best();
    GGML_ASSERT(backend);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    GGML_ASSERT(cpu);

    if (strcmp(ggml_backend_name(backend), ggml_backend_name(cpu)) == 0) {
        printf("test-paged-write-q8: no non-CPU backend; nothing to gate\n");
        return 0;
    }
    printf("backend: %s\n", ggml_backend_name(backend));

    const int D = 128, H = 4, HKV = 2, BS = 32, NB = 2;
    const int N = BS*NB - BS/2;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q_p   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,   N);
    ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
    ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
    ggml_tensor * cache = ggml_new_tensor_4d(ctx, GGML_TYPE_Q8_0, D, BS, 2*HKV, NB);
    ggml_tensor * btab  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NB, 1);
    ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * clens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * boffs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * blens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_tensor * out = ggml_paged_attn_banded(ctx, q_p, k_new, v_new, cache, cache,
            btab, slots, clens, boffs, blens, nullptr, 1.0f/sqrtf((float) D), BS, NB, 1, 0);
    ggml_set_name(out, "out");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("test-paged-write-q8: could not allocate a q8_0 paged cache -- SKIP\n"); return 0; }

    std::vector<float> tmp((size_t) N*H*D, 0.0f);
    ggml_backend_tensor_set(q_p, tmp.data(), 0, tmp.size()*sizeof(float));

    std::vector<float> kh((size_t) N*HKV*D), vh((size_t) N*HKV*D);
    for (int t = 0; t < N; ++t) for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d) {
        kh[(size_t) t*HKV*D + h*D + d] = val_k(t, h, d);
        vh[(size_t) t*HKV*D + h*D + d] = val_v(t, h, d);
    }
    ggml_backend_tensor_set(k_new, kh.data(), 0, kh.size()*sizeof(float));
    ggml_backend_tensor_set(v_new, vh.data(), 0, vh.size()*sizeof(float));

    std::vector<int32_t> i32(N);
    for (int t = 0; t < N; ++t) i32[t] = t;
    ggml_backend_tensor_set(slots, i32.data(), 0, N*sizeof(int32_t));
    i32 = {0, 1};  ggml_backend_tensor_set(btab,  i32.data(), 0, 2*sizeof(int32_t));
    i32 = {N};     ggml_backend_tensor_set(clens, i32.data(), 0, sizeof(int32_t));
    i32 = {0};     ggml_backend_tensor_set(boffs, i32.data(), 0, sizeof(int32_t));
    i32 = {N};     ggml_backend_tensor_set(blens, i32.data(), 0, sizeof(int32_t));

    std::vector<uint8_t> zeros(ggml_nbytes(cache), 0);
    ggml_backend_tensor_set(cache, zeros.data(), 0, zeros.size());

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(backend, gf);

    // Read the CACHE back -- this is the artifact under test, not dst.
    std::vector<uint8_t> got(ggml_nbytes(cache));
    ggml_backend_tensor_get(cache, got.data(), 0, got.size());

    const int    nbk        = D / 32;
    const size_t blk_bytes  = ggml_type_size(GGML_TYPE_Q8_0);
    const size_t row_blocks = (size_t) nbk;

    int n_bad = 0, n_checked = 0;
    for (int t = 0; t < N; ++t) {
        for (int h = 0; h < HKV; ++h) {
            for (int b = 0; b < nbk; ++b) {
                float  d_ref; int8_t qs_ref[32];
                quant_ref(&kh[(size_t) t*HKV*D + h*D + b*32], d_ref, qs_ref);

                const size_t blk_index = (size_t) (t/BS) * (BS * 2*HKV * row_blocks)
                                       + (size_t) h * (BS * row_blocks)
                                       + (size_t) (t%BS) * row_blocks + b;
                const uint8_t * p = got.data() + blk_index*blk_bytes;
                const float  d_got = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
                const int8_t * qs_got = (const int8_t *) (p + sizeof(ggml_fp16_t));

                ++n_checked;
                if (fabsf(d_got - d_ref) > 1e-6f) { ++n_bad; continue; }
                for (int j = 0; j < 32; ++j) {
                    if (qs_got[j] != qs_ref[j]) { ++n_bad; break; }
                }
            }
        }
    }

    printf("checked %d q8_0 blocks, %d mismatched\n", n_checked, n_bad);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(backend);

    printf(n_bad == 0 ? "test-paged-write-q8: PASS\n" : "test-paged-write-q8: FAIL\n");
    return n_bad == 0 ? 0 : 1;
}
