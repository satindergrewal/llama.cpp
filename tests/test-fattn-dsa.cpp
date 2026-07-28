//
// Numerical validation of GGML_OP_FLASH_ATTN_EXT_DSA (CUDA, fattn-dsa.cu).
//
// The claim under test: gathering the indexer-selected top-k K/V rows and running
// attention over just those is EXACTLY equivalent to running the stock
// GGML_OP_FLASH_ATTN_EXT over the full cache with a mask that is the "real" mask
// at the selected positions and -INF everywhere else. Softmax over the same
// surviving set, so the two must agree to within fp16 accumulation error.
//
// Three paths are computed and cross-compared:
//   A  ggml_flash_attn_ext_dsa(q, k, v, mask_dsa, topk_idx, scale)      (the new op)
//   B  ggml_flash_attn_ext(q, k, v, mask_ref, scale, 0, 0)              (stock, -INF mask)
//   R  an exact fp64 host reference over the same selected set          (ground truth)
//
// R is computed from the same fp16 K/V bits that were uploaded, so it isolates
// kernel error from input quantisation. Comparing A-vs-R against B-vs-R attributes
// any A-vs-B disagreement to one path or the other instead of just reporting a delta.
//
// Discriminating power is checked explicitly: a control reference computed over a
// DIFFERENT random top-k set is also compared, so a near-zero A-vs-B number can be
// trusted to mean "same math" and not "test is insensitive".
//
// Deliberately NOT wired into tests/CMakeLists.txt: the op is CUDA-only and has no
// CPU implementation, so it cannot be a portable ctest. Build and run by hand:
//
//   cd <repo-root>
//   g++ -O2 -std=c++17 -Iggml/include tests/test-fattn-dsa.cpp \
//       -Lbuild/bin -lggml -lggml-base -Wl,-rpath,$PWD/build/bin -o build/bin/test-fattn-dsa
//   CUDA_VISIBLE_DEVICES=0 ./build/bin/test-fattn-dsa
//

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------- utils

struct diff_stats {
    double max_abs      = 0.0;  // max |a - b|
    double max_rel      = 0.0;  // max |a - b| / max(|b|, 1e-3*max|b|)
    double max_abs_norm = 0.0;  // max |a - b| / max|b|          (tensor-scale normalised)
    double rel_l2       = 0.0;  // ||a - b||_2 / ||b||_2
    int64_t n_bad       = 0;    // NaN / Inf in a
};

static diff_stats compare(const std::vector<double> & a, const std::vector<double> & b) {
    diff_stats s;
    double bmax = 0.0;
    for (double x : b) {
        bmax = std::max(bmax, std::fabs(x));
    }
    const double floor_v = std::max(1e-30, 1e-3*bmax);

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) {
            s.n_bad++;
            continue;
        }
        const double d = std::fabs(a[i] - b[i]);
        s.max_abs = std::max(s.max_abs, d);
        s.max_rel = std::max(s.max_rel, d/std::max(std::fabs(b[i]), floor_v));
        num += d*d;
        den += b[i]*b[i];
    }
    s.max_abs_norm = bmax > 0.0 ? s.max_abs/bmax : 0.0;
    s.rel_l2       = den  > 0.0 ? std::sqrt(num/den) : 0.0;
    return s;
}

static void print_stats(const char * label, const diff_stats & s) {
    printf("    %-26s max_abs %10.3e   max_rel %10.3e   max_abs/scale %10.3e   rel_L2 %10.3e   bad %lld\n",
            label, s.max_abs, s.max_rel, s.max_abs_norm, s.rel_l2, (long long) s.n_bad);
}

static std::vector<double> to_f64(const std::vector<float> & v) {
    return std::vector<double>(v.begin(), v.end());
}

// ---------------------------------------------------------------------------- config

struct config {
    const char * name;
    int64_t DK;         // K head size  (== Q head size)
    int64_t DV;         // V head size
    int64_t n_head;     // Q->ne[2]; K/V heads are forced to 1 by the op's guards
    int64_t n_kv;       // K->ne[1]
    int64_t n_tokens;   // Q->ne[1]
    int64_t top_k;      // indexer->ne[0]
    bool    v_is_k_view;// V is a sub-view of the K rows (MLA)
    bool    poison_mask;// fill NON-selected mask entries with junk instead of 0
    bool    mask_holes; // additionally set -INF on a fraction of the SELECTED entries
    bool    expect_unsupported; // the op's guards must reject this shape
    bool    duplicate_idx;      // repeat every index twice: gather double-counts, the
                                // -INF-mask formulation cannot. A-vs-B divergence here
                                // is expected and is a property of the formulation.
};

// mirror of dsa_attn_layout_ok() in ggml/src/ggml-cuda/fattn-dsa.cu, for reporting
// exactly which guard rejected a case instead of just "unsupported".
static void audit_guards(const ggml_tensor * dst) {
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K       = dst->src[1];
    const ggml_tensor * V       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sink    = dst->src[4];
    const ggml_tensor * indexer = dst->src[5];

    auto chk = [](const char * what, bool ok) {
        printf("      guard %-46s %s\n", what, ok ? "ok" : "FAILED");
    };

    chk("dst->type == F32",                       dst->type == GGML_TYPE_F32);
    chk("no sinks",                               sink == nullptr);
    chk("Q,K,V,mask,indexer all present",         Q && K && V && mask && indexer);
    chk("indexer->ne[0] % 256 == 0",              indexer && indexer->ne[0] % 256 == 0);
    chk("K->ne[1] >= 4*indexer->ne[0]",           K && indexer && K->ne[1] >= 4*indexer->ne[0]);
    chk("K/mask ne2,ne3 == 1 && Q->ne[3] == 1",   K && mask && Q &&
                                                  K->ne[2] == 1 && K->ne[3] == 1 &&
                                                  mask->ne[2] == 1 && mask->ne[3] == 1 && Q->ne[3] == 1);
    chk("K,V,mask F16 && Q F32",                  K && V && mask && Q &&
                                                  K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16 &&
                                                  mask->type == GGML_TYPE_F16 && Q->type == GGML_TYPE_F32);
    chk("K->ne[0] == Q->ne[0]",                   K && Q && K->ne[0] == Q->ne[0]);
    chk("V ne2,ne3 == 1",                         V && V->ne[2] == 1 && V->ne[3] == 1);
    chk("indexer->type == I32",                   indexer && indexer->type == GGML_TYPE_I32);
    chk("indexer ne1 >= Q->ne[1], ne2/ne3 == 1",  indexer && Q && indexer->ne[1] >= Q->ne[1] &&
                                                  indexer->ne[2] == 1 && indexer->ne[3] == 1);
}

// ---------------------------------------------------------------------------- the case

static bool run_case(ggml_backend_t backend, const config & cfg, uint32_t seed) {
    printf("\n=== %s\n", cfg.name);
    printf("    DK=%lld DV=%lld n_head=%lld n_kv=%lld n_tokens=%lld top_k=%lld v_is_k_view=%d poison_mask=%d mask_holes=%d\n",
            (long long) cfg.DK, (long long) cfg.DV, (long long) cfg.n_head, (long long) cfg.n_kv,
            (long long) cfg.n_tokens, (long long) cfg.top_k, (int) cfg.v_is_k_view,
            (int) cfg.poison_mask, (int) cfg.mask_holes);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::uniform_real_distribution<float> junk(-8.0f, 8.0f);

    ggml_init_params ip = { /*.mem_size =*/ 64ull*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        printf("    FAIL: ggml_init\n");
        return false;
    }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, cfg.DK, cfg.n_tokens, cfg.n_head, 1);
    ggml_set_name(q, "q");

    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.DK, cfg.n_kv, 1, 1);
    ggml_set_name(k, "k");

    ggml_tensor * v = nullptr;
    if (cfg.v_is_k_view) {
        // MLA layout: V is the leading DV elements of each K row, same buffer, same row stride
        v = ggml_view_4d(ctx, k, cfg.DV, cfg.n_kv, 1, 1, k->nb[1], k->nb[2], k->nb[3], 0);
    } else {
        v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.DV, cfg.n_kv, 1, 1);
    }
    ggml_set_name(v, "v");

    ggml_tensor * mask_dsa = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.n_kv, cfg.n_tokens, 1, 1);
    ggml_tensor * mask_ref = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, cfg.n_kv, cfg.n_tokens, 1, 1);
    ggml_tensor * idx      = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, cfg.top_k, cfg.n_tokens, 1, 1);
    ggml_set_name(mask_dsa, "mask_dsa");
    ggml_set_name(mask_ref, "mask_ref");
    ggml_set_name(idx,      "topk_idx");

    const float scale = 1.0f/std::sqrt((float) cfg.DK);

    ggml_tensor * out_dsa = ggml_flash_attn_ext_dsa(ctx, q, k, v, mask_dsa, idx, scale);
    ggml_set_name(out_dsa, "out_dsa");

    ggml_tensor * out_ref = ggml_flash_attn_ext(ctx, q, k, v, mask_ref, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out_ref, GGML_PREC_F32);
    ggml_set_name(out_ref, "out_ref");

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool sup_dsa = ggml_backend_dev_supports_op(dev, out_dsa);
    const bool sup_ref = ggml_backend_dev_supports_op(dev, out_ref);
    printf("    supports_op: DSA=%s  stock FLASH_ATTN_EXT=%s\n", sup_dsa ? "yes" : "NO", sup_ref ? "yes" : "NO");

    if (!sup_dsa) {
        audit_guards(out_dsa);
    }
    if (cfg.expect_unsupported) {
        printf("    verdict: guards must reject this shape -> %s\n", sup_dsa ? "FAIL (accepted!)" : "PASS (rejected)");
        ggml_free(ctx);
        return !sup_dsa;
    }
    if (!sup_dsa || !sup_ref) {
        printf("    SKIPPED (backend reports the op unsupported for this shape)\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("    FAIL: ggml_backend_alloc_ctx_tensors\n");
        ggml_free(ctx);
        return false;
    }

    // ---- host data -------------------------------------------------------------

    const int64_t nq = cfg.DK*cfg.n_tokens*cfg.n_head;
    std::vector<float> q_host(nq);
    for (auto & x : q_host) x = uni(rng);

    std::vector<ggml_fp16_t> k_host(cfg.DK*cfg.n_kv);
    for (auto & x : k_host) x = ggml_fp32_to_fp16(uni(rng));

    std::vector<ggml_fp16_t> v_host;
    if (!cfg.v_is_k_view) {
        v_host.resize(cfg.DV*cfg.n_kv);
        for (auto & x : v_host) x = ggml_fp32_to_fp16(uni(rng));
    }

    // v_val(row, j) reads whichever tensor actually backs V
    auto v_val = [&](int64_t row, int64_t j) -> double {
        return cfg.v_is_k_view
            ? (double) ggml_fp16_to_fp32(k_host[row*cfg.DK + j])
            : (double) ggml_fp16_to_fp32(v_host[row*cfg.DV + j]);
    };

    // top-k: distinct indices per query row, deliberately in RANDOM order (softmax is
    // permutation invariant, so a correct gather must not care)
    std::vector<int32_t> idx_host(cfg.top_k*cfg.n_tokens);
    std::vector<int32_t> ctrl_idx(cfg.top_k*cfg.n_tokens); // control set: different keys
    {
        std::vector<int32_t> perm(cfg.n_kv);
        std::iota(perm.begin(), perm.end(), 0);
        for (int64_t t = 0; t < cfg.n_tokens; ++t) {
            std::shuffle(perm.begin(), perm.end(), rng);
            std::copy(perm.begin(), perm.begin() + cfg.top_k, idx_host.begin() + t*cfg.top_k);
            if (cfg.duplicate_idx) {
                // repeat the first quarter of the selection over the last quarter: those
                // keys now appear twice in the gathered list and once in the -INF mask
                for (int64_t c = 0; c < cfg.top_k/4; ++c) {
                    idx_host[t*cfg.top_k + 3*cfg.top_k/4 + c] = idx_host[t*cfg.top_k + c];
                }
            }
            std::shuffle(perm.begin(), perm.end(), rng);
            std::copy(perm.begin(), perm.begin() + cfg.top_k, ctrl_idx.begin() + t*cfg.top_k);
        }
    }

    const ggml_fp16_t f16_zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t f16_ninf = ggml_fp32_to_fp16(-INFINITY);

    std::vector<ggml_fp16_t> mask_dsa_host(cfg.n_kv*cfg.n_tokens, f16_zero);
    std::vector<ggml_fp16_t> mask_ref_host(cfg.n_kv*cfg.n_tokens, f16_ninf);

    if (cfg.poison_mask) {
        // entries the DSA path must never read; if the mask gather picks the wrong
        // column this poison shows up immediately
        for (auto & x : mask_dsa_host) x = ggml_fp32_to_fp16(junk(rng));
    }
    const int64_t n_holes = cfg.mask_holes ? cfg.top_k/4 : 0;
    for (int64_t t = 0; t < cfg.n_tokens; ++t) {
        for (int64_t c = 0; c < cfg.top_k; ++c) {
            const int32_t j = idx_host[t*cfg.top_k + c];
            // a selected position carries the "real" mask value: 0, or -INF for a hole
            const ggml_fp16_t mv = (c < n_holes) ? f16_ninf : f16_zero;
            mask_dsa_host[t*cfg.n_kv + j] = mv;
            mask_ref_host[t*cfg.n_kv + j] = mv;
        }
    }

    ggml_backend_tensor_set(q,        q_host.data(),        0, ggml_nbytes(q));
    ggml_backend_tensor_set(k,        k_host.data(),        0, ggml_nbytes(k));
    if (!cfg.v_is_k_view) {
        ggml_backend_tensor_set(v,    v_host.data(),        0, ggml_nbytes(v));
    }
    ggml_backend_tensor_set(mask_dsa, mask_dsa_host.data(), 0, ggml_nbytes(mask_dsa));
    ggml_backend_tensor_set(mask_ref, mask_ref_host.data(), 0, ggml_nbytes(mask_ref));
    ggml_backend_tensor_set(idx,      idx_host.data(),      0, ggml_nbytes(idx));

    // poison the destinations so an op that writes nothing cannot masquerade as a pass
    {
        std::vector<float> nanbuf(ggml_nelements(out_dsa), std::nanf(""));
        ggml_backend_tensor_set(out_dsa, nanbuf.data(), 0, ggml_nbytes(out_dsa));
        nanbuf.resize(ggml_nelements(out_ref), std::nanf(""));
        ggml_backend_tensor_set(out_ref, nanbuf.data(), 0, ggml_nbytes(out_ref));
    }

    // ---- run -------------------------------------------------------------------

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_dsa);
    ggml_build_forward_expand(gf, out_ref);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        printf("    FAIL: graph compute returned %d\n", (int) st);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_synchronize(backend);

    std::vector<float> a_f32(ggml_nelements(out_dsa));
    std::vector<float> b_f32(ggml_nelements(out_ref));
    ggml_backend_tensor_get(out_dsa, a_f32.data(), 0, ggml_nbytes(out_dsa));
    ggml_backend_tensor_get(out_ref, b_f32.data(), 0, ggml_nbytes(out_ref));

    // ---- fp64 host reference ---------------------------------------------------

    auto reference = [&](const std::vector<int32_t> & sel) {
        std::vector<double> out((size_t) cfg.DV*cfg.n_head*cfg.n_tokens, 0.0);
        std::vector<double> logit(cfg.top_k);
        for (int64_t t = 0; t < cfg.n_tokens; ++t) {
            for (int64_t h = 0; h < cfg.n_head; ++h) {
                const float * qp = q_host.data() + t*cfg.DK + h*cfg.DK*cfg.n_tokens;
                double lmax = -INFINITY;
                for (int64_t c = 0; c < cfg.top_k; ++c) {
                    const int32_t j = sel[t*cfg.top_k + c];
                    const ggml_fp16_t * kp = k_host.data() + (int64_t) j*cfg.DK;
                    double dot = 0.0;
                    for (int64_t d = 0; d < cfg.DK; ++d) {
                        dot += (double) qp[d]*(double) ggml_fp16_to_fp32(kp[d]);
                    }
                    const double m = (double) ggml_fp16_to_fp32(mask_dsa_host[t*cfg.n_kv + j]);
                    logit[c] = (double) scale*dot + m;
                    lmax = std::max(lmax, logit[c]);
                }
                double sum = 0.0;
                for (int64_t c = 0; c < cfg.top_k; ++c) {
                    logit[c] = std::isfinite(logit[c]) ? std::exp(logit[c] - lmax) : 0.0;
                    sum += logit[c];
                }
                const double inv = 1.0/sum;
                double * op = out.data() + h*cfg.DV + t*cfg.DV*cfg.n_head;
                for (int64_t c = 0; c < cfg.top_k; ++c) {
                    const double p = logit[c]*inv;
                    if (p == 0.0) continue;
                    const int32_t j = sel[t*cfg.top_k + c];
                    for (int64_t d = 0; d < cfg.DV; ++d) {
                        op[d] += p*v_val(j, d);
                    }
                }
            }
        }
        return out;
    };

    const std::vector<double> ref  = reference(idx_host);
    const std::vector<double> ctrl = reference(ctrl_idx);

    const std::vector<double> a = to_f64(a_f32);
    const std::vector<double> b = to_f64(b_f32);

    const diff_stats ab   = compare(a, b);
    const diff_stats ar   = compare(a, ref);
    const diff_stats br   = compare(b, ref);
    const diff_stats ctl  = compare(ctrl, ref);

    printf("    -- results ---------------------------------------------------------------\n");
    print_stats("A(DSA)   vs B(stock FA)", ab);
    print_stats("A(DSA)   vs R(fp64 ref)", ar);
    print_stats("B(stock) vs R(fp64 ref)", br);
    print_stats("control (wrong top-k)  ", ctl);

    // discriminating power: the wrong-index control must be orders of magnitude worse
    const bool discriminates = ctl.rel_l2 > 100.0*std::max(ab.rel_l2, 1e-12);
    // A must not be materially worse than the stock kernel against the same ground truth
    const bool a_ok = ar.rel_l2 <= std::max(5.0*br.rel_l2, 5e-3) && ar.n_bad == 0 && ab.n_bad == 0;
    const bool ab_ok = ab.rel_l2 <= 5e-3;

    bool ok;
    if (cfg.duplicate_idx) {
        // A must still match the gather semantics (R), and it must visibly DIVERGE from
        // the -INF-mask formulation, which physically cannot double-count a key.
        ok = a_ok && ab.rel_l2 > 100.0*ar.rel_l2;
        printf("    verdict: duplicate-index case. A-vs-R %.3e (%s, gather semantics held) | "
               "A-vs-B %.3e = EXPECTED divergence (%s)\n",
                ar.rel_l2, a_ok ? "PASS" : "FAIL",
                ab.rel_l2, ok ? "seen as predicted" : "NOT SEEN");
    } else {
        ok = ab_ok && a_ok && discriminates;
        printf("    verdict: A-vs-B rel_L2 %.3e (%s) | A-vs-R %.3e vs B-vs-R %.3e (%s) | control rel_L2 %.3e (%s)\n",
                ab.rel_l2, ab_ok ? "PASS" : "FAIL",
                ar.rel_l2, br.rel_l2, a_ok ? "PASS" : "FAIL",
                ctl.rel_l2, discriminates ? "test is sensitive" : "TEST NOT SENSITIVE");
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

// ---------------------------------------------------------------------------- main

int main(int argc, char ** argv) {
    const uint32_t seed0 = argc > 1 ? (uint32_t) strtoul(argv[1], nullptr, 10) : 1234u;

    ggml_backend_load_all();

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        printf("no GPU backend available\n");
        return 1;
    }
    printf("backend: %s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_description(ggml_backend_get_device(backend)));

    const config cfgs[] = {
        // name                                        DK   DV  nh   n_kv  ntok  topk  view  poison holes  unsup  dup
        { "d128 separate V, 8 tok, clean mask",        128, 128,  8, 2048,    8,  256, false, false, false, false, false },
        { "d128 separate V, 8 tok, poisoned mask",     128, 128,  8, 2048,    8,  256, false, true,  false, false, false },
        { "d128 separate V, 8 tok, poison + holes",    128, 128,  8, 2048,    8,  256, false, true,  true,  false, false },
        { "d128 separate V, 40 tok (2 chunks)",        128, 128,  8, 2048,   40,  256, false, true,  false, false, false },
        { "d128 separate V, 65 tok (3 chunks)",        128, 128,  4, 2048,   65,  256, false, true,  true,  false, false },
        { "d128 top_k=512, n_kv=4096",                 128, 128,  8, 4096,   16,  512, false, true,  false, false, false },
        { "MLA d576/512, V=view(K), 8 tok",            576, 512,  8, 2048,    8,  256, true,  true,  false, false, false },
        { "MLA d576/512, V=view(K), 40 tok, holes",    576, 512,  8, 2048,   40,  256, true,  true,  true,  false, false },
        { "MLA d576/512, V=view(K), 128 heads",        576, 512,128, 2048,    4,  256, true,  false, false, false, false },
        // guards must reject these, not compute garbage
        { "GUARD top_k=192 (not %256)",                128, 128,  8, 2048,    8,  192, false, false, false, true,  false },
        { "GUARD n_kv=512 < 4*top_k",                  128, 128,  8,  512,    8,  256, false, false, false, true,  false },
        // formulation caveat, not a kernel bug: gather double-counts a repeated index
        { "DUP indices (expected divergence)",         128, 128,  8, 2048,    8,  256, false, false, false, false, true  },
    };

    int n_pass = 0;
    int n_fail = 0;
    uint32_t seed = seed0;
    printf("base seed: %u\n", seed0);
    for (const config & c : cfgs) {
        if (run_case(backend, c, seed++)) {
            n_pass++;
        } else {
            n_fail++;
        }
    }

    printf("\n===========================================================\n");
    printf("passed %d / %d\n", n_pass, n_pass + n_fail);

    ggml_backend_free(backend);
    return n_fail == 0 ? 0 : 1;
}
