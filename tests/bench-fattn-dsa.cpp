//
// Wall-clock benchmark: GGML_OP_FLASH_ATTN_EXT_DSA (gather top-k) vs the mask-shaped
// formulation it is meant to replace, at real GLM-5.2 MLA shapes.
//
// The two things being compared, as llm_graph_context::build_attn() actually builds them:
//
//   gather    ggml_flash_attn_ext_dsa(q, k, v, kq_mask, top_k, scale)
//             one node; reads top_k K/V rows.
//
//   mask      ggml_fill(kq_mask, -INF) -> ggml_set_rows(zeros at top_k idx) -> ggml_add(kq_mask)
//             -> ggml_flash_attn_ext(q, k, v, mask_top_k, scale, 0, 0) @ GGML_PREC_F32
//             four nodes; the FA reads all n_kv rows. The three mask nodes are work that
//             only exists because of the mask formulation, so they are part of its cost.
//
//   construct just the three mask nodes, so the construction cost can be broken out of
//             the mask total instead of being asserted.
//
// The mask-construction chain is copied verbatim from src/llama-graph.cpp (the
// `// prepare new kq mask - starts filled with -INFINITY` block), including the
// view shapes, so this measures the real graph and not an idealised one.
//
// Measurement notes:
//   * inputs (q_cur, k, kq_mask, top_k) are identical for all three paths and live in
//     their own backend buffer; only the per-path graph scratch is allocated by the
//     gallocr, so ggml_gallocr_get_buffer_size() is a clean per-path compute-buffer number.
//   * ggml_backend_graph_compute() synchronises the backend before returning; an explicit
//     ggml_backend_synchronize() is issued as well, so no async launch time is hidden.
//   * warmup is at least 8 iterations AND at least 300 ms, then the iteration count is
//     chosen so the timed region runs >= 1.5 s with a floor of 20 iterations.
//   * median is the headline; min/max/p25/p75 are reported so clock drift is visible
//     (the GPU is not in persistence mode and clocks are not locked).
//   * device free-memory is sampled via ggml_backend_dev_memory() at four points so the
//     kernel's own pool scratch (which does not appear in the compute buffer) is visible.
//     For that number to mean anything, run ONE case+path per process (--case/--path).
//   * q/K/mask/top_k contents are generated from a fixed seed with an index-keyed PRNG, so
//     two separate processes produce bit-identical inputs. Each run prints a checksum of
//     the output, which cross-validates gather-vs-mask numerically at the benchmark shapes.
//
// top-k index distribution: uniformly scattered distinct positions over the causally
// allowed range. That is the worst case for any block-level sparsity exploitation and the
// best case for nothing; a real indexer's picks cluster. See the report.
//
// Deliberately NOT wired into tests/CMakeLists.txt (CUDA-only op, no CPU fallback).
// Build and run by hand:
//
//   cd <repo-root>
//   g++ -O2 -std=c++17 -Iggml/include tests/bench-fattn-dsa.cpp \
//       -Lbuild/bin -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o build/bin/bench-fattn-dsa
//   CUDA_VISIBLE_DEVICES=0 ./build/bin/bench-fattn-dsa --list
//   CUDA_VISIBLE_DEVICES=0 ./build/bin/bench-fattn-dsa --case 0 --path gather
//

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------- prng
// index-keyed so two processes generate identical data without sharing state

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27))*0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// uniform in [-1, 1)
static inline float rnd_sym(uint64_t stream, uint64_t i) {
    const uint64_t h = splitmix64(stream*0x1000000000000003ull + i);
    return (float) ((double) (h >> 11)/(double) (1ull << 53))*2.0f - 1.0f;
}

// ---------------------------------------------------------------------------- config

enum path_kind {
    PATH_GATHER = 0,
    PATH_MASK,
    PATH_CONSTRUCT,
    PATH_NULL,      // 1-node graph on a 1-element tensor: measures the launch+sync floor
    PATH_COUNT,
};

static const char * path_name(path_kind p) {
    switch (p) {
        case PATH_GATHER:    return "gather";
        case PATH_MASK:      return "mask";
        case PATH_CONSTRUCT: return "construct";
        case PATH_NULL:      return "null";
        default:             return "?";
    }
}

struct bench_case {
    const char * tag;
    int64_t DK;         // K/Q head size
    int64_t DV;         // V head size
    int64_t n_head;     // Q->ne[2] after permute
    int64_t n_kv;
    int64_t n_tokens;
    int64_t top_k;
    bool    v_is_k_view;
};

// GLM-5.2 MLA: DK 576, DV 512, 64 heads, 1 KV head, n_indexer_top_k 2048, V a view into K.
// The head count is glm-dsa.attention.head_count = 64, read from the served GGUF. An earlier
// revision of this file used 128, which is wrong and inflates the gathered path: n_head is
// the n dimension of both batched GEMMs, so a larger value makes them less skinny and the
// batched cuBLAS calls more efficient than they really are. Use --nhead to reproduce the
// old numbers.
#define GLM_ROW(nkv, ntok, tk) { "glm", 576, 512, 64, (nkv), (ntok), (tk), true }

static const bench_case g_cases[] = {
    // headline: top_k = 2048, decode
    GLM_ROW(  8192,   1, 2048),
    GLM_ROW( 16384,   1, 2048),
    GLM_ROW( 32768,   1, 2048),
    GLM_ROW( 65536,   1, 2048),
    GLM_ROW(131072,   1, 2048),
    // headline: top_k = 2048, prefill / ubatch
    GLM_ROW(  8192, 512, 2048),
    GLM_ROW( 16384, 512, 2048),
    GLM_ROW( 32768, 512, 2048),
    GLM_ROW( 65536, 512, 2048),
    GLM_ROW(131072, 512, 2048),
    // control: top_k = 256, so n_kv/top_k rises 8x at fixed n_kv
    GLM_ROW(  8192,   1,  256),
    GLM_ROW( 32768,   1,  256),
    GLM_ROW(131072,   1,  256),
    GLM_ROW(  8192, 512,  256),
    GLM_ROW( 32768, 512,  256),
    GLM_ROW(131072, 512,  256),
    // guard boundary: n_kv/top_k == 4 is the smallest ratio the kernel accepts, so this is
    // where the crossover lives if there is one. Held at ratio 4 across three absolute
    // sizes to separate "speedup depends on the ratio" from "speedup depends on top_k".
    GLM_ROW( 32768,   1, 8192),
    GLM_ROW( 65536,   1,16384),
    GLM_ROW( 32768, 512, 8192),
    GLM_ROW( 65536, 512,16384),
    // ratio 8 at a large top_k, to complete the ratio-vs-top_k separation
    GLM_ROW( 65536, 512, 8192),
    GLM_ROW( 65536,   1, 8192),
};

static const int g_n_cases = (int) (sizeof(g_cases)/sizeof(g_cases[0]));

// ---------------------------------------------------------------------------- stats

struct timing {
    double med = 0, mn = 0, mx = 0, p25 = 0, p75 = 0;
    int    n   = 0;
};

static timing summarise(std::vector<double> v) {
    timing t;
    if (v.empty()) {
        return t;
    }
    std::sort(v.begin(), v.end());
    t.n   = (int) v.size();
    t.mn  = v.front();
    t.mx  = v.back();
    t.med = v.size() % 2 ? v[v.size()/2] : 0.5*(v[v.size()/2 - 1] + v[v.size()/2]);
    t.p25 = v[(size_t) (0.25*(v.size() - 1))];
    t.p75 = v[(size_t) (0.75*(v.size() - 1))];
    return t;
}

// ---------------------------------------------------------------------------- run

struct run_result {
    bool    supported          = false;
    const char * refusal       = nullptr;
    timing  t;
    size_t  compute_buf        = 0;  // gallocr graph scratch, exact
    size_t  input_buf          = 0;  // shared inputs, identical across paths
    int64_t dev_after_inputs   = 0;  // free bytes, device
    int64_t dev_after_reserve  = 0;
    int64_t dev_after_run      = 0;
    int64_t dev_base           = 0;
    double  out_absmean        = 0.0;
    int64_t out_nonfinite      = -1;
    int64_t n_nodes            = 0;
};

// mirror of dsa_attn_layout_ok() in ggml/src/ggml-cuda/fattn-dsa.cu; used only to name the
// guard that refused a shape, so a refusal is reported instead of silently skipped
static const char * guard_refusal(const ggml_tensor * dst) {
    const ggml_tensor * Q   = dst->src[0];
    const ggml_tensor * K   = dst->src[1];
    const ggml_tensor * V   = dst->src[2];
    const ggml_tensor * M   = dst->src[3];
    const ggml_tensor * IDX = dst->src[5];

    if (dst->type != GGML_TYPE_F32)                 return "dst not F32";
    if (!Q || !K || !V || !M || !IDX)               return "missing src";
    if (IDX->ne[0] % 256 != 0)                      return "top_k % 256 != 0";
    if (K->ne[1] < 4*IDX->ne[0])                    return "n_kv < 4*top_k";
    if (K->ne[2] > 1 || K->ne[3] > 1)               return "K ne2/ne3 > 1";
    if (M->ne[2] > 1 || M->ne[3] > 1)               return "mask ne2/ne3 > 1";
    if (Q->ne[3] > 1)                               return "Q ne3 > 1";
    if (K->type != GGML_TYPE_F16)                   return "K not F16";
    if (V->type != GGML_TYPE_F16)                   return "V not F16";
    if (M->type != GGML_TYPE_F16)                   return "mask not F16";
    if (Q->type != GGML_TYPE_F32)                   return "Q not F32";
    if (K->ne[0] != Q->ne[0])                       return "DK mismatch";
    if (V->ne[2] > 1 || V->ne[3] > 1)               return "V ne2/ne3 > 1";
    if (IDX->type != GGML_TYPE_I32)                 return "top_k not I32";
    if (IDX->ne[1] < Q->ne[1])                      return "top_k rows < n_tokens";
    if (IDX->ne[2] > 1 || IDX->ne[3] > 1)           return "top_k ne2/ne3 > 1";
    return "unknown (backend refused a shape all mirrored guards accept)";
}

static int64_t dev_free(ggml_backend_dev_t dev) {
    size_t f = 0, t = 0;
    ggml_backend_dev_memory(dev, &f, &t);
    return (int64_t) f;
}

static run_result run_one(ggml_backend_t backend, const bench_case & c, path_kind path,
                          uint64_t seed, int min_iters, double target_s, bool verbose) {
    run_result R;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    R.dev_base = dev_free(dev);

    const float scale = 1.0f/std::sqrt((float) c.DK);

    // ---- inputs (identical for every path) -------------------------------------

    ggml_init_params ip_in = { /*.mem_size =*/ 16ull*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx_in = ggml_init(ip_in);
    GGML_ASSERT(ctx_in);

    // pre-permute layout, exactly as llm_graph_context hands it to build_attn:
    //   q_cur [DK, n_head, n_tokens, 1] contiguous, permuted to [DK, n_tokens, n_head, 1]
    ggml_tensor * q_cur   = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F32, c.DK, c.n_head, c.n_tokens, 1);
    ggml_tensor * k       = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, c.DK, c.n_kv, 1, 1);
    ggml_tensor * kq_mask = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, c.n_kv, c.n_tokens, 1, 1);
    ggml_tensor * top_k   = ggml_new_tensor_4d(ctx_in, GGML_TYPE_I32, c.top_k, c.n_tokens, 1, 1);
    ggml_tensor * v_sep   = c.v_is_k_view ? nullptr
                          : ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, c.DV, c.n_kv, 1, 1);

    ggml_set_name(q_cur, "q_cur");
    ggml_set_name(k, "k");
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_name(top_k, "top_k");

    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    if (!buf_in) {
        printf("FATAL: could not allocate input tensors\n");
        ggml_free(ctx_in);
        return R;
    }
    R.input_buf        = ggml_backend_buffer_get_size(buf_in);
    R.dev_after_inputs = dev_free(dev);

    // ---- host data -------------------------------------------------------------

    {
        std::vector<float> q_host((size_t) c.DK*c.n_head*c.n_tokens);
        for (size_t i = 0; i < q_host.size(); ++i) {
            q_host[i] = rnd_sym(1, i);
        }
        ggml_backend_tensor_set(q_cur, q_host.data(), 0, ggml_nbytes(q_cur));
    }
    {
        std::vector<ggml_fp16_t> k_host((size_t) c.DK*c.n_kv);
        for (size_t i = 0; i < k_host.size(); ++i) {
            k_host[i] = ggml_fp32_to_fp16(rnd_sym(2, i));
        }
        ggml_backend_tensor_set(k, k_host.data(), 0, ggml_nbytes(k));
    }
    if (v_sep) {
        std::vector<ggml_fp16_t> v_host((size_t) c.DV*c.n_kv);
        for (size_t i = 0; i < v_host.size(); ++i) {
            v_host[i] = ggml_fp32_to_fp16(rnd_sym(3, i));
        }
        ggml_backend_tensor_set(v_sep, v_host.data(), 0, ggml_nbytes(v_sep));
    }

    // causal base mask: token t may attend to kv positions [0, n_kv - n_tokens + t]
    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);
    {
        std::vector<ggml_fp16_t> m_host((size_t) c.n_kv*c.n_tokens);
        for (int64_t t = 0; t < c.n_tokens; ++t) {
            const int64_t lim = c.n_kv - c.n_tokens + t; // inclusive
            ggml_fp16_t * row = m_host.data() + t*c.n_kv;
            for (int64_t j = 0; j <= lim; ++j) {
                row[j] = f16_zero;
            }
            for (int64_t j = lim + 1; j < c.n_kv; ++j) {
                row[j] = f16_ninf;
            }
        }
        ggml_backend_tensor_set(kq_mask, m_host.data(), 0, ggml_nbytes(kq_mask));
    }

    // top-k: distinct, uniformly scattered over the causally allowed range, unsorted
    {
        std::vector<int32_t> perm(c.n_kv);
        std::iota(perm.begin(), perm.end(), 0);
        std::vector<int32_t> idx_host((size_t) c.top_k*c.n_tokens);
        uint64_t ctr = 0;
        for (int64_t t = 0; t < c.n_tokens; ++t) {
            const int64_t lim = c.n_kv - c.n_tokens + t + 1; // exclusive
            GGML_ASSERT(lim >= c.top_k);
            for (int64_t cc = 0; cc < c.top_k; ++cc) {
                const uint64_t h = splitmix64(4*0x1000000000000003ull + (ctr++));
                const int64_t  r = cc + (int64_t) (h % (uint64_t) (lim - cc));
                std::swap(perm[cc], perm[r]);
                idx_host[t*c.top_k + cc] = perm[cc];
            }
        }
        ggml_backend_tensor_set(top_k, idx_host.data(), 0, ggml_nbytes(top_k));
    }

    // ---- graph -----------------------------------------------------------------

    ggml_init_params ip_g = { /*.mem_size =*/ 16ull*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx_g = ggml_init(ip_g);
    GGML_ASSERT(ctx_g);

    ggml_tensor * q = ggml_permute(ctx_g, q_cur, 0, 2, 1, 3);              // [DK, n_tokens, n_head, 1]
    ggml_tensor * v = c.v_is_k_view
        ? ggml_view_4d(ctx_g, k, c.DV, c.n_kv, 1, 1, k->nb[1], k->nb[2], k->nb[3], 0)
        : v_sep;

    ggml_tensor * out = nullptr;

    if (path == PATH_NULL) {
        // smallest possible unit of work: one graph launch + one backend sync. Every other
        // path pays this exact overhead once per timed iteration, so it is the floor that
        // must be subtracted to get the marginal cost the op would have inside a real
        // model graph (where one CUDA graph capture covers ~1000 nodes).
        ggml_tensor * tiny = ggml_new_tensor_1d(ctx_g, GGML_TYPE_F32, 1);
        out = ggml_fill(ctx_g, tiny, 0.0f);
        ggml_set_name(out, "null");
    } else if (path == PATH_GATHER) {
        out = ggml_flash_attn_ext_dsa(ctx_g, q, k, v, kq_mask, top_k, scale);
        ggml_set_name(out, "kqv_dsa");

        if (!ggml_backend_dev_supports_op(dev, out)) {
            R.supported = false;
            R.refusal   = guard_refusal(out);
            ggml_free(ctx_g);
            ggml_backend_buffer_free(buf_in);
            ggml_free(ctx_in);
            return R;
        }
    } else {
        // --- verbatim from src/llama-graph.cpp, llm_graph_context::build_attn(k_dsa) ---

        // prepare new kq mask - starts filled with -INFINITY
        ggml_tensor * kq_mask_all = ggml_fill(ctx_g, kq_mask, -INFINITY);

        // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
        kq_mask_all = ggml_view_4d(ctx_g, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
                                   kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

        // [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
        ggml_tensor * top_k_3d = ggml_view_4d(ctx_g, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
                                              top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

        ggml_tensor * zeros = ggml_new_tensor_4d(ctx_g, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
        zeros = ggml_fill(ctx_g, zeros, 0.0f);

        ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx_g, kq_mask_all, zeros, top_k_3d);

        // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
        kq_mask_top_k = ggml_view_4d(ctx_g, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
                                     kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

        kq_mask_top_k = ggml_add(ctx_g, kq_mask_top_k, kq_mask);
        ggml_set_name(kq_mask_top_k, "kq_mask_top_k");

        if (path == PATH_CONSTRUCT) {
            out = kq_mask_top_k;
        } else {
            // build_attn_mha() with cparams.flash_attn, kq_b == nullptr, sinks == nullptr
            out = ggml_flash_attn_ext(ctx_g, q, k, v, kq_mask_top_k, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            ggml_set_name(out, "kqv_mask");

            if (!ggml_backend_dev_supports_op(dev, out)) {
                R.supported = false;
                R.refusal   = "backend refused stock GGML_OP_FLASH_ATTN_EXT at this shape";
                ggml_free(ctx_g);
                ggml_backend_buffer_free(buf_in);
                ggml_free(ctx_in);
                return R;
            }
        }
    }

    R.supported = true;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, out);
    R.n_nodes = ggml_graph_n_nodes(gf);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_reserve(galloc, gf)) {
        printf("FATAL: gallocr reserve failed\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx_g);
        ggml_backend_buffer_free(buf_in);
        ggml_free(ctx_in);
        R.supported = false;
        R.refusal   = "gallocr reserve failed";
        return R;
    }
    R.compute_buf       = ggml_gallocr_get_buffer_size(galloc, 0);
    R.dev_after_reserve = dev_free(dev);

    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        printf("FATAL: gallocr alloc_graph failed\n");
        R.supported = false;
        R.refusal   = "gallocr alloc failed";
        ggml_gallocr_free(galloc);
        ggml_free(ctx_g);
        ggml_backend_buffer_free(buf_in);
        ggml_free(ctx_in);
        return R;
    }

    // ---- warmup ----------------------------------------------------------------
    // >= 8 iterations and >= 300 ms: brings clocks up and forces cuBLAS handle creation,
    // kernel module load, CUDA-graph capture and the ggml CUDA pool to their steady state

    auto now = [] { return std::chrono::steady_clock::now(); };
    auto ms  = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    {
        const auto t0 = now();
        int it = 0;
        while (it < 8 || ms(t0, now()) < 300.0) {
            const ggml_status st = ggml_backend_graph_compute(backend, gf);
            if (st != GGML_STATUS_SUCCESS) {
                printf("FATAL: graph compute returned %d\n", (int) st);
                R.supported = false;
                R.refusal   = "graph compute failed";
                ggml_gallocr_free(galloc);
                ggml_free(ctx_g);
                ggml_backend_buffer_free(buf_in);
                ggml_free(ctx_in);
                return R;
            }
            ggml_backend_synchronize(backend);
            ++it;
            if (it > 50000) {
                break;
            }
        }
        if (verbose) {
            printf("  warmup: %d iters in %.1f ms\n", it, ms(t0, now()));
        }
    }

    // one timed probe to size the real loop
    double t_probe = 0.0;
    {
        const auto t0 = now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        t_probe = ms(t0, now());
    }

    int iters = (int) std::ceil(1000.0*target_s/std::max(t_probe, 1e-3));
    iters = std::max(iters, min_iters);
    iters = std::min(iters, 2000);

    // ---- timed -----------------------------------------------------------------

    std::vector<double> samples;
    samples.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = now();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        const auto t1 = now();
        samples.push_back(ms(t0, t1));
    }
    R.t = summarise(samples);

    R.dev_after_run = dev_free(dev);

    // ---- output checksum -------------------------------------------------------
    // identical inputs across processes => gather and mask must agree here

    {
        const int64_t n = ggml_nelements(out);
        if (out->type == GGML_TYPE_F32) {
            std::vector<float> h(n);
            ggml_backend_tensor_get(out, h.data(), 0, ggml_nbytes(out));
            double s = 0.0;
            int64_t bad = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (std::isfinite(h[i])) {
                    s += std::fabs((double) h[i]);
                } else {
                    ++bad;
                }
            }
            R.out_absmean   = s/(double) n;
            R.out_nonfinite = bad;
        } else if (out->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h(n);
            ggml_backend_tensor_get(out, h.data(), 0, ggml_nbytes(out));
            double s = 0.0;
            int64_t bad = 0;
            for (int64_t i = 0; i < n; ++i) {
                const float x = ggml_fp16_to_fp32(h[i]);
                // -INF is the expected value for most of a constructed mask; count only NaN
                if (std::isnan(x)) {
                    ++bad;
                } else if (std::isfinite(x)) {
                    s += std::fabs((double) x);
                }
            }
            R.out_absmean   = s/(double) n;
            R.out_nonfinite = bad;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx_g);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);

    return R;
}

// ---------------------------------------------------------------------------- main

static void print_result(const bench_case & c, path_kind path, const run_result & R) {
    if (!R.supported) {
        printf("RESULT tag=%s path=%s n_kv=%lld n_tokens=%lld top_k=%lld n_head=%lld REFUSED reason=\"%s\"\n",
                c.tag, path_name(path), (long long) c.n_kv, (long long) c.n_tokens, (long long) c.top_k,
                (long long) c.n_head, R.refusal ? R.refusal : "?");
        return;
    }

    // free-memory deltas: positive == bytes taken
    const long long d_inputs  = (long long) (R.dev_base          - R.dev_after_inputs);
    const long long d_compute = (long long) (R.dev_after_inputs  - R.dev_after_reserve);
    const long long d_pool    = (long long) (R.dev_after_reserve - R.dev_after_run);

    printf("RESULT tag=%s path=%s n_kv=%lld n_tokens=%lld top_k=%lld n_head=%lld nodes=%lld "
           "iters=%d med_ms=%.4f min_ms=%.4f p25_ms=%.4f p75_ms=%.4f max_ms=%.4f "
           "compute_buf=%llu input_buf=%llu dev_inputs=%lld dev_compute=%lld dev_pool=%lld "
           "absmean=%.9g nonfinite=%lld\n",
            c.tag, path_name(path), (long long) c.n_kv, (long long) c.n_tokens, (long long) c.top_k,
            (long long) c.n_head, (long long) R.n_nodes,
            R.t.n, R.t.med, R.t.mn, R.t.p25, R.t.p75, R.t.mx,
            (unsigned long long) R.compute_buf, (unsigned long long) R.input_buf,
            d_inputs, d_compute, d_pool,
            R.out_absmean, (long long) R.out_nonfinite);
}

int main(int argc, char ** argv) {
    int         only_case = -1;
    path_kind   only_path = PATH_COUNT;
    int         min_iters = 20;
    double      target_s  = 1.5;
    uint64_t    seed      = 1234;
    bool        list      = false;
    bool        verbose   = false;
    int64_t     ov_nkv    = 0;   // --nkv/--ntok/--topk override the selected case in place,
    int64_t     ov_ntok   = 0;   // so one-off shapes can be probed without editing the table
    int64_t     ov_topk   = 0;
    int64_t     ov_nhead  = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : ""; };
        if      (a == "--case")    only_case = atoi(next());
        else if (a == "--iters")   min_iters = atoi(next());
        else if (a == "--target")  target_s  = atof(next());
        else if (a == "--seed")    seed      = strtoull(next(), nullptr, 10);
        else if (a == "--list")    list      = true;
        else if (a == "-v")        verbose   = true;
        else if (a == "--nkv")     ov_nkv    = atoll(next());
        else if (a == "--ntok")    ov_ntok   = atoll(next());
        else if (a == "--topk")    ov_topk   = atoll(next());
        else if (a == "--nhead")   ov_nhead  = atoll(next());
        else if (a == "--path") {
            const std::string p = next();
            if      (p == "gather")    only_path = PATH_GATHER;
            else if (p == "mask")      only_path = PATH_MASK;
            else if (p == "construct") only_path = PATH_CONSTRUCT;
            else if (p == "null")      only_path = PATH_NULL;
            else { printf("unknown --path %s\n", p.c_str()); return 1; }
        } else {
            printf("unknown arg %s\n", a.c_str());
            return 1;
        }
    }

    if (list) {
        for (int i = 0; i < g_n_cases; ++i) {
            const bench_case & c = g_cases[i];
            printf("case %2d  n_kv=%-7lld n_tokens=%-4lld top_k=%-5lld DK=%lld DV=%lld n_head=%lld v_is_k_view=%d\n",
                    i, (long long) c.n_kv, (long long) c.n_tokens, (long long) c.top_k,
                    (long long) c.DK, (long long) c.DV, (long long) c.n_head, (int) c.v_is_k_view);
        }
        return 0;
    }

    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        printf("no GPU backend available\n");
        return 1;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    size_t f0 = 0, t0 = 0;
    ggml_backend_dev_memory(dev, &f0, &t0);
    printf("# backend: %s (%s)  free %.2f GiB / %.2f GiB\n",
            ggml_backend_name(backend), ggml_backend_dev_description(dev),
            f0/1073741824.0, t0/1073741824.0);

    int rc = 0;
    for (int i = 0; i < g_n_cases; ++i) {
        if (only_case >= 0 && i != only_case) {
            continue;
        }
        for (int p = 0; p < PATH_COUNT; ++p) {
            if (only_path != PATH_COUNT && (int) only_path != p) {
                continue;
            }
            bench_case c = g_cases[i];
            if (ov_nkv)  c.n_kv     = ov_nkv;
            if (ov_ntok) c.n_tokens = ov_ntok;
            if (ov_topk) c.top_k    = ov_topk;
            if (ov_nhead) c.n_head  = ov_nhead;

            const run_result R = run_one(backend, c, (path_kind) p, seed, min_iters, target_s, verbose);
            print_result(c, (path_kind) p, R);
            fflush(stdout);
            if (!R.supported && R.refusal && strstr(R.refusal, "failed")) {
                rc = 1;
            }
        }
    }

    ggml_backend_free(backend);
    return rc;
}
