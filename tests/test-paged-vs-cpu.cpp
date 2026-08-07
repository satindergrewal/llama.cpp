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
// ⚠ sink_mode GOES LAST. It was added as the 5th parameter, in front of the existing kv_type, and
// the two existing calls that pass GGML_TYPE_Q8_0 / GGML_TYPE_F16 silently became sink_mode=8 and
// sink_mode=1 with kv_type defaulted. Both compiled. The f16 incremental arms then failed at
// max_abs=1.7e-01 and I spent a bisect chasing the library before finding it here. Adding a
// defaulted parameter in the MIDDLE of a signature is a silent argument shift.
static std::vector<float> run_paged(ggml_backend_t backend, int D, bool with_rel, int64_t window,
                                    ggml_type kv_type = GGML_TYPE_F16,
                                    int sink_mode = 0,     // 0 none, 1 finite, 2 -inf control
                                    int causal = 1,
                                    // ⚠ LAST. THIRD TIME THIS SESSION. read_only_check was inserted
                                    // between sink_mode and causal, so run_paged(..., 0, 0) -- meant
                                    // as sink_mode=0, causal=0 -- silently became read_only=false with
                                    // causal DEFAULTED TO 1, and the non-causal arm quietly tested
                                    // causal attention. It compiled. It ran. It "passed" its CPU
                                    // comparison, because both sides were causal.
                                    bool read_only_check = false) {
    // ⚠ H/HKV ARE ENV-OVERRIDABLE so the harness can be REPLAYED at the server's real geometry.
    // The ARGDUMP showed the server running head_dim=256, n_heads=16, n_heads_kv=4 (GQA 4:1) with
    // n_blocks=1 -- while this file had GQA 2:1 hardcoded. Every "PASS" it has printed was at a GQA
    // ratio the failing configuration does not use, which is one more instance of the coverage
    // being narrower than the claim it licensed.
    const int H   = getenv("DS4P_TEST_H")   ? atoi(getenv("DS4P_TEST_H"))   : 4;    // query heads
    const int HKV = getenv("DS4P_TEST_HKV") ? atoi(getenv("DS4P_TEST_HKV")) : 2;    // kv heads
    const int E   = 8;    // rel_extent
    // block_size / block count are env-overridable so a gate can exercise a path that has a
    // block-size PRECONDITION (the lane-per-key loop needs bs >= 32). Hard-coded at BS=16,
    // DS4P_METAL_LPK=1 would have printed "ALL PASSED" for a path that never ran -- the
    // presence marker caught exactly that, which is why this knob exists.
    const int BS  = getenv("DS4P_TEST_BS") ? atoi(getenv("DS4P_TEST_BS")) : 16;   // block_size
    const int NB  = getenv("DS4P_TEST_NB") ? atoi(getenv("DS4P_TEST_NB")) : 2;    // blocks
    // Ends MID-block on purpose: a token count that lands exactly on a block boundary never
    // tests the partial-tile tail, where the lane-per-key masking lives.
    // ⚠ N IS OVERRIDABLE so block COUNT can be varied independently of token COUNT. Sweeping NB
    // alone changes both (N = BS*NB - BS/2), so "NB=1 fails, NB=2 passes" is two factors moving
    // together and cannot name a cause -- exactly the arms-differ-in-one-thing rule this lane has
    // already paid for once.
    const int N   = getenv("DS4P_TEST_N") ? atoi(getenv("DS4P_TEST_N")) : BS*NB - BS/2;
    const float scale = 1.0f / sqrtf((float) D);

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q_p   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,   N);
    ggml_tensor * k_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
    ggml_tensor * v_new = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, N);
    ggml_tensor * cache = ggml_new_tensor_4d(ctx, kv_type, D, BS, 2*HKV, NB);
    ggml_tensor * btab  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NB, 1);
    ggml_tensor * slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_tensor * clens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * boffs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * blens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * rel_p = with_rel ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, H, N) : nullptr;

    // ★ SINKS ARM. sink_mode 2 is the NEGATIVE CONTROL and is the one worth trusting: a sink of
    // -inf contributes exp(-inf - M) = 0 to the denominator, so the result MUST equal the no-sinks
    // answer bit for bit. That property holds without me having to trust any reference derivation --
    // if the plumbing is wrong, -inf will not reproduce it; if only my sink MATH is wrong, -inf still
    // passes while the finite arm diverges. Two failures, two signals.
    ggml_tensor * sinks_p = sink_mode ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
    if (sinks_p) { ggml_set_name(sinks_p, "sinks"); }

    ggml_tensor * out_p = ggml_paged_attn_banded(ctx, q_p, k_new, v_new, cache, cache,
            btab, slots, clens, boffs, blens, rel_p, scale, BS, NB, NB, sinks_p, with_rel ? E : 1, window, causal);
    ggml_set_name(out_p, "out_paged");

    // ★ READ-ONLY ARM. A SECOND op over the SAME cache with K and V null: it must skip the write and
    // attend exactly what the first call stored, so its output has to equal out_p. Expanded into the
    // graph after out_p so the write has happened; the two ops share the cache tensor, which is the
    // whole point -- gemma4-assistant's NextN head attends KV the main graph wrote.
    //
    // ⚠ Without this arm, "read-only compiles and nothing regressed" would mean only that the write
    // still happens when K/V are PRESENT. It would say nothing about the path actually taken.
    ggml_tensor * out_ro = nullptr;
    if (read_only_check) {
        out_ro = ggml_paged_attn_banded(ctx, q_p, nullptr, nullptr, cache, cache,
                btab, slots, clens, boffs, blens, rel_p, scale, BS, NB, NB, sinks_p, with_rel ? E : 1, window, causal);
        ggml_set_name(out_ro, "out_paged_ro");
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    if (sinks_p) {
        std::vector<float> sk(H);
        for (int h = 0; h < H; ++h) {
            sk[h] = sink_mode == 2 ? -INFINITY : (0.25f + 0.10f*h);
        }
        ggml_backend_tensor_set(sinks_p, sk.data(), 0, ggml_nbytes(sinks_p));
    }

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
    // ⚠ WAS `i32 = {0, 1}` with a hardcoded 2-entry write -- at DS4P_TEST_NB=1 that wrote past the
    // end of btab and aborted in ggml_backend_tensor_set. Identity table sized from NB, like the
    // slots above; the block count is a knob, so nothing may assume its value.
    i32.assign(NB, 0);
    for (int b = 0; b < NB; ++b) { i32[b] = b; }
    ggml_backend_tensor_set(btab,  i32.data(), 0, (size_t) NB*sizeof(int32_t));
    i32 = {N};     ggml_backend_tensor_set(clens, i32.data(), 0, sizeof(int32_t));
    i32 = {0};     ggml_backend_tensor_set(boffs, i32.data(), 0, sizeof(int32_t));
    i32 = {N};     ggml_backend_tensor_set(blens, i32.data(), 0, sizeof(int32_t));

    std::vector<uint8_t> zeros(ggml_nbytes(cache), 0);
    ggml_backend_tensor_set(cache, zeros.data(), 0, zeros.size());

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_p);
    if (out_ro) { ggml_build_forward_expand(gf, out_ro); }
    ggml_backend_graph_compute(backend, gf);

    std::vector<float> out(ggml_nelements(out_p));
    ggml_backend_tensor_get(out_p, out.data(), 0, ggml_nbytes(out_p));
    if (out_ro) {
        // return the READ-ONLY result so the caller compares it against the ordinary one
        ggml_backend_tensor_get(out_ro, out.data(), 0, ggml_nbytes(out_ro));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// ★ INCREMENTAL WRITES -- the largest difference between this harness and a real server, and the
// one the poison probe pointed at.
//
// run_paged() above writes all N tokens in ONE op call. A server writes the prompt, then ONE token
// per decode step, into a cache the previous step already partly filled. Measured on Ornith-9B:
// paged f16 is bit-identical with the pool poisoned, paged q8_0 is NOT -- so on the quantised path
// something reads a region the write never covered, and an unwritten q8_0 block dequantises to a
// clean 0.0 rather than to NaN, which is why f16 never showed it.
//
// This runs the SAME work as two sequential graphs over one cache: tokens [0,N1) then [N1,N).
// Attention is causal, so the concatenation MUST equal the single-call result exactly -- same
// arithmetic, same order, only the write schedule differs. Any divergence is the incremental path.
static std::vector<float> run_paged_split(ggml_backend_t backend, int D, bool with_rel, int64_t window,
                                          ggml_type kv_type, int n1, int n_dec = 1,
                                          // ⚠ LAST. I moved sink_mode to the end of run_paged one hour
                                          // ago for exactly this reason and then inserted causal in
                                          // front of two non-defaulted parameters here. The compiler
                                          // caught it this time; in run_paged it did not, because
                                          // there the shifted argument still type-checked.
                                          int causal = 1) {
    const int H   = getenv("DS4P_TEST_H")   ? atoi(getenv("DS4P_TEST_H"))   : 4;
    const int HKV = getenv("DS4P_TEST_HKV") ? atoi(getenv("DS4P_TEST_HKV")) : 2;
    const int E   = 8;
    const int BS  = getenv("DS4P_TEST_BS") ? atoi(getenv("DS4P_TEST_BS")) : 16;
    const int NB  = getenv("DS4P_TEST_NB") ? atoi(getenv("DS4P_TEST_NB")) : 2;
    const int N   = getenv("DS4P_TEST_N") ? atoi(getenv("DS4P_TEST_N")) : BS*NB - BS/2;
    const float scale = 1.0f / sqrtf((float) D);
    // n_dec sequential single-token DECODE calls after the prefill. n_dec >= 2 is the shape the
    // server fails on and this harness could not previously express: a token written by one decode
    // dispatch and read back by the NEXT one. With n_dec == 1 the decode reads only what it just
    // wrote plus prefill, which passes.
    const int   n_parts = 1 + n_dec;
    const int   n_pre   = N - n_dec;   // prefill token count
    GGML_ASSERT(n_pre >= 1 && n1 <= N);

    ggml_init_params ip = { ggml_tensor_overhead()*32*(size_t)(n_parts+2) + ggml_graph_overhead()*(size_t)n_parts, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * cache = ggml_new_tensor_4d(ctx, kv_type, D, BS, 2*HKV, NB);
    ggml_tensor * btab  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NB, 1);

    struct part {
        int n, off;
        ggml_tensor *q, *k, *v, *slots, *clens, *boffs, *blens, *rel, *out;
    };
    std::vector<part> P;
    P.push_back({ n_pre, 0, nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr });
    for (int d = 0; d < n_dec; ++d) {
        P.push_back({ 1, n_pre + d, nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr });
    }

    for (int s = 0; s < n_parts; ++s) {
        part & p = P[s];
        p.q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H,   p.n);
        p.k     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, p.n);
        p.v     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, HKV, p.n);
        p.slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, p.n);
        p.clens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        p.boffs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        p.blens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        p.rel   = with_rel ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, H, p.n) : nullptr;
        p.out   = ggml_paged_attn_banded(ctx, p.q, p.k, p.v, cache, cache,
                      btab, p.slots, p.clens, p.boffs, p.blens, p.rel,
                      scale, BS, NB, NB, nullptr, with_rel ? E : 1, window, causal);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<uint8_t> zeros(ggml_nbytes(cache), 0);
    ggml_backend_tensor_set(cache, zeros.data(), 0, zeros.size());
    std::vector<int32_t> bt(NB);
    for (int b = 0; b < NB; ++b) { bt[b] = b; }
    ggml_backend_tensor_set(btab, bt.data(), 0, (size_t) NB*sizeof(int32_t));

    std::vector<float> out((size_t) N*H*D);

    for (int s = 0; s < n_parts; ++s) {
        part & p = P[s];
        std::vector<float> tq((size_t) p.n*H*D), tk((size_t) p.n*HKV*D), tv((size_t) p.n*HKV*D);
        for (int t = 0; t < p.n; ++t) {
            const int gt = p.off + t;   // GLOBAL token index: the values must not depend on the split
            for (int h = 0; h < H;   ++h) for (int d = 0; d < D; ++d) tq[(size_t) t*H*D   + h*D + d] = val_q(gt, h, d);
            for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d) tk[(size_t) t*HKV*D + h*D + d] = val_k(gt, h, d);
            for (int h = 0; h < HKV; ++h) for (int d = 0; d < D; ++d) tv[(size_t) t*HKV*D + h*D + d] = val_v(gt, h, d);
        }
        ggml_backend_tensor_set(p.q, tq.data(), 0, tq.size()*sizeof(float));
        ggml_backend_tensor_set(p.k, tk.data(), 0, tk.size()*sizeof(float));
        ggml_backend_tensor_set(p.v, tv.data(), 0, tv.size()*sizeof(float));

        if (with_rel) {
            std::vector<float> r((size_t) E*H*p.n, 0.0f);
            for (int t = 0; t < p.n; ++t) for (int h = 0; h < H; ++h) for (int e = 0; e < E; ++e)
                r[(size_t) t*H*E + h*E + e] = val_r(e, h, p.off + t);
            ggml_backend_tensor_set(p.rel, r.data(), 0, r.size()*sizeof(float));
        }

        std::vector<int32_t> sl(p.n);
        for (int t = 0; t < p.n; ++t) sl[t] = p.off + t;      // identity slots, same as run_paged
        ggml_backend_tensor_set(p.slots, sl.data(), 0, p.n*sizeof(int32_t));
        int32_t v_clen = p.off + p.n;   // cumulative context AFTER this write
        int32_t v_zero = 0, v_len = p.n;
        ggml_backend_tensor_set(p.clens, &v_clen, 0, sizeof(int32_t));
        ggml_backend_tensor_set(p.boffs, &v_zero, 0, sizeof(int32_t));
        ggml_backend_tensor_set(p.blens, &v_len,  0, sizeof(int32_t));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, p.out);
        ggml_backend_graph_compute(backend, gf);

        ggml_backend_tensor_get(p.out, out.data() + (size_t) p.off*H*D, 0, ggml_nbytes(p.out));
    }

    // ★ CACHE-CONTENT CHECK, COVERAGE DERIVED FROM THE READ'S ADDRESS RANGE.
    //
    // My earlier write gate reported "384 blocks, 0 mismatched" and it was TRUE -- it validated a
    // region NARROWER than what the attend pass consumes, which is exactly how a coverage bug
    // survives a passing gate. So this walks EVERY (token, head) row the read can address under the
    // current geometry, reads the cache back, dequantises on the host and compares against the
    // values the write was handed. No chosen sample, no chosen block.
    //
    // It splits the remaining search in one run: rows wrong in the cache means the WRITE is
    // corrupting (and names which rows); rows right means the write is fine and the READ addresses
    // something else. Those two produce an IDENTICAL output signature, which is what made the
    // retracted decode-dequant claim look plausible.
    if (getenv("DS4P_TEST_CACHECHK")) {
        const size_t row_bytes = ggml_row_size(kv_type, D);
        std::vector<uint8_t> raw(row_bytes);
        std::vector<float>   got(D);
        // q8_0 error is bounded by dq/2 = amax/254, ~1e-3 at these input magnitudes.
        const double tol = (kv_type == GGML_TYPE_Q8_0) ? 3e-3 : 1e-3;
        int bad_k = 0, bad_v = 0, first_bad = -1;
        double worst = 0.0;

        for (int t = 0; t < N; ++t) {
            const int blk = t / BS, tib = t % BS;
            for (int h = 0; h < HKV; ++h) {
                for (int which = 0; which < 2; ++which) {   // 0 = K, 1 = V
                    const size_t off = (size_t) blk*cache->nb[3]
                                     + (size_t) (which ? HKV + h : h)*cache->nb[2]
                                     + (size_t) tib*cache->nb[1];
                    ggml_backend_tensor_get(cache, raw.data(), off, row_bytes);
                    if (kv_type == GGML_TYPE_Q8_0) {
                        const int nbk = D / (int) ggml_blck_size(GGML_TYPE_Q8_0);
                        for (int b = 0; b < nbk; ++b) {
                            const uint8_t * bp = raw.data() + (size_t) b*ggml_type_size(GGML_TYPE_Q8_0);
                            const float d = ggml_fp16_to_fp32(*(const ggml_fp16_t *) bp);
                            const int8_t * qs = (const int8_t *) (bp + sizeof(ggml_fp16_t));
                            for (int j = 0; j < 32; ++j) { got[b*32 + j] = d * (float) qs[j]; }
                        }
                    } else {
                        const ggml_fp16_t * hp = (const ggml_fp16_t *) raw.data();
                        for (int d = 0; d < D; ++d) { got[d] = ggml_fp16_to_fp32(hp[d]); }
                    }
                    for (int d = 0; d < D; ++d) {
                        const double want = which ? val_v(t, h, d) : val_k(t, h, d);
                        const double err  = fabs(want - got[d]);
                        if (err > worst) { worst = err; }
                        if (err > tol) {
                            if (which) { ++bad_v; } else { ++bad_k; }
                            if (first_bad < 0) { first_bad = t; }
                            break;   // one strike per row
                        }
                    }
                }
            }
        }
        printf("   [cachechk %-5s D=%3d N=%d NB=%d] bad_K_rows=%d bad_V_rows=%d of %d each"
               " | first_bad_token=%d worst=%.3e\n",
               ggml_type_name(kv_type), D, N, NB, bad_k, bad_v, N*HKV, first_bad, worst);
    }

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
    // 256 added: Ornith-9B has head_dim=256 and the capability predicate was refusing it on a
    // hardcoded {64,128} allow-list. NPT = (D+31)/32 with float qv[8] means D<=256 is the
    // kernel's real ceiling -- so 256 must be TESTED before the predicate is widened to admit it.
    const int dims[] = { 64, 96, 128, 192, 256, 512 };   // 512: gemma4 n_embd_head
    int n_fail = 0;

    // ★ MIRRORS paged_layer_supported()'s JOINT (block_size, head_dim) CONTRACT. The scalar path
    // stages 2 * block_size * head_dim halves, so head_dim alone never bounded it: bs=64/D=512
    // wants 131,072 B of threadgroup memory and used to return silently wrong numbers (2.34e-02,
    // f16 and q8_0 alike). The dispatch now asserts, so without this skip the whole run aborts at
    // BS=64 and the OTHER dims never get tested -- a harness that dies on a case the real path
    // refuses is measuring nothing. ⚠ SKIPS ARE PRINTED: a silently-shrunk sweep reads as
    // "everything passed", which is the exact failure this file already has scars from.
    const int tst_bs = getenv("DS4P_TEST_BS") ? atoi(getenv("DS4P_TEST_BS")) : 16;
    const auto tile_fits = [&](int D) {
        if ((int64_t) tst_bs * D <= 8192) { return true; }
        printf("SKIP D=%3d at block_size=%d: block_size*head_dim=%d > 8192, the paged predicate "
               "refuses this combination (staged tile would not fit)\n", D, tst_bs, tst_bs*D);
        return false;
    };

    // ⚠ WAS `di < 4` -- a HARDCODED bound beside a sized array. Adding 256 to dims[] silently did
    // nothing and the run still printed ALL PASSED, i.e. a green verdict for a case never executed.
    // Same gate-plumbing-lie class this lane keeps paying for. Derive the bound from the array.
    for (size_t di = 0; di < sizeof(dims)/sizeof(dims[0]); ++di) {
        const int D = dims[di];
        if (!tile_fits(D)) { continue; }
        for (int cse = 0; cse < 3; ++cse) {
            const bool    with_rel = cse != 2;
            const int64_t window   = (cse == 0) ? 0 : 8;

            const std::vector<float> a = run_paged(backend, D, with_rel, window);
            const std::vector<float> b = run_paged(cpu,     D, with_rel, window);

            // ★ SINKS. Finite sink against the CPU op (which now implements them), plus the -inf
            // control against the NO-SINKS answer on the same backend.
            if (cse == 0) {
                const std::vector<float> as = run_paged(backend, D, with_rel, window, GGML_TYPE_F16, 1);
                const std::vector<float> bs = run_paged(cpu,     D, with_rel, window, GGML_TYPE_F16, 1);
                double m = 0.0;
                for (size_t i = 0; i < as.size() && i < bs.size(); ++i) m = std::max(m, (double) fabs(as[i]-bs[i]));
                printf("D=%3d sinks finite : max_abs=%.3e %s\n", D, m, m < 2e-3 ? "PASS" : "FAIL");
                n_fail += m < 2e-3 ? 0 : 1;

                // ★ NON-CAUSAL ARM. Metal against the CPU op, both with causal=0. Its control is
                // the ALL PASSED above: the same code path with causal=1 reproduces every existing
                // answer, so a non-causal divergence cannot be the branch merely existing.
                const std::vector<float> anc = run_paged(backend, D, with_rel, window, GGML_TYPE_F16, 0, 0);
                const std::vector<float> bnc = run_paged(cpu,     D, with_rel, window, GGML_TYPE_F16, 0, 0);
                double mn = 0.0;
                for (size_t i = 0; i < anc.size() && i < bnc.size(); ++i) mn = std::max(mn, (double) fabs(anc[i]-bnc[i]));
                printf("D=%3d non-causal   : max_abs=%.3e %s\n", D, mn, mn < 2e-3 ? "PASS" : "FAIL");
                n_fail += mn < 2e-3 ? 0 : 1;

                // ⚠ AND IT MUST ACTUALLY DIFFER FROM THE CAUSAL ANSWER. If non-causal produced the
                // same numbers, the flag would be doing nothing and the arm above would pass while
                // testing nothing -- the exact shape of a gate that cannot fail.
                double dc = 0.0;
                for (size_t i = 0; i < anc.size() && i < a.size(); ++i) dc = std::max(dc, (double) fabs(anc[i]-a[i]));
                printf("D=%3d non-causal differs from causal: max_abs=%.3e %s\n",
                       D, dc, dc > 1e-3 ? "PASS" : "FAIL (flag is a no-op)");
                n_fail += dc > 1e-3 ? 0 : 1;

                // ★ READ-ONLY: same pool, second op with K/V null, must equal the ordinary answer.
                // ⚠ OFF BY DEFAULT UNTIL READ-ONLY IS IMPLEMENTED. The op aborts on null K/V today.
                // The arm is kept, not deleted: it is the check that caught the segfault the moment a
                // read-only call actually ran, after "compiles and does not regress" had said nothing.
                if (!getenv("DS4P_TEST_READONLY")) { goto skip_ro; }
                {
                const std::vector<float> aro = run_paged(backend, D, with_rel, window, GGML_TYPE_F16, 0, true);
                double mr = 0.0;
                for (size_t i = 0; i < aro.size() && i < a.size(); ++i) mr = std::max(mr, (double) fabs(aro[i]-a[i]));
                printf("D=%3d read-only    : max_abs=%.3e %s   (must equal the write-then-read answer)\n",
                       D, mr, mr < 1e-6 ? "PASS" : "FAIL");
                n_fail += mr < 1e-6 ? 0 : 1;
                }
                skip_ro:;

                const std::vector<float> ai = run_paged(backend, D, with_rel, window, GGML_TYPE_F16, 2);
                double c = 0.0;
                for (size_t i = 0; i < ai.size() && i < a.size(); ++i) c = std::max(c, (double) fabs(ai[i]-a[i]));
                printf("D=%3d sinks -inf   : max_abs=%.3e %s   (control: must equal no-sinks)\n",
                       D, c, c < 1e-6 ? "PASS" : "FAIL");
                n_fail += c < 1e-6 ? 0 : 1;
            }

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

    // ★ FULL-PATH q8_0 ARM. Both sides now quantise with IDENTICAL arithmetic (the CPU reference's
    // quant_row mirrors kernel_paged_attn_write_q8_0), so this keeps the SAME 2e-3 bar as every
    // f16 case. No loosened tolerance -- a gate widened to swallow ~8e-3 of quantisation error
    // would also pass a genuinely broken kernel.
    // ⚠ SWEEPS THE SAME DIMS AS THE f16 ARM. This was D=128 ONLY, and on that single dimension I
    // called the q8_0 read path "proven on both backends" -- while the model the end-to-end gate
    // actually runs (Ornith-9B) has head_dim 256, which the arm never touched. Same shape as the
    // `di < 4` loop bound that printed ALL PASSED for a case that never ran: the claim was wider
    // than the coverage. dims[] is shared with the f16 arm so the two cannot drift apart.
    if (!same) {
        for (size_t di = 0; di < sizeof(dims)/sizeof(dims[0]); ++di)
        for (int cse = 0; cse < 3; ++cse) {
            const int     D        = dims[di];
            if (!tile_fits(D)) { continue; }
            const bool    with_rel = cse != 2;
            const int64_t window   = (cse == 0) ? 0 : 8;

            const std::vector<float> a = run_paged(backend, D, with_rel, window, GGML_TYPE_Q8_0);
            const std::vector<float> b = run_paged(cpu,     D, with_rel, window, GGML_TYPE_Q8_0);
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

            // Which SIDE is bad? nmse=nan with a finite max_abs means NaNs, not zeros -- and a
            // verdict that cannot say which side failed sends you diagnosing the wrong backend.
            int nan_a = 0, nan_b = 0, nz_a = 0, nz_b = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                if (std::isnan(a[i])) ++nan_a; else if (a[i] != 0.0f) ++nz_a;
                if (std::isnan(b[i])) ++nan_b; else if (b[i] != 0.0f) ++nz_b;
            }
            printf("   [diag] metal: nan=%d nonzero=%d | cpu: nan=%d nonzero=%d | n=%zu\n",
                   nan_a, nz_a, nan_b, nz_b, a.size());

            printf("q8_0 D=%3d case %c: with_rel=%d window=%lld max_abs=%.3e nmse=%.3e %s\n",
                   D, 'A' + cse, with_rel ? 1 : 0, (long long) window, max_abs, nmse, ok ? "PASS" : "FAIL");
            n_fail += ok ? 0 : 1;

            // ★ DIVERGENCE BEFORE EQUALITY. The agreement above is metal-q8 vs cpu-q8; if BOTH
            // sides silently skipped quantisation it would pass just as cleanly. So require the
            // q8_0 result to DIFFER from the f16 result by roughly a quantisation step. A gate
            // that only checks equality cannot tell "both correct" from "both bypassed".
            const std::vector<float> f16ref = run_paged(backend, D, with_rel, window, GGML_TYPE_F16);
            double qdiff = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double d = fabs((double) a[i] - f16ref[i]);
                qdiff = d > qdiff ? d : qdiff;
            }
            // Floor: below this the q8 store did nothing. Ceiling: above it the error is not
            // quantisation noise but a bug wearing its clothes.
            const bool diverged = qdiff > 1e-5 && qdiff < 5e-2;
            printf("   [divergence] q8_0 vs f16 max_abs=%.3e %s (quantisation actually applied)\n",
                   qdiff, diverged ? "PASS" : "FAIL");
            n_fail += diverged ? 0 : 1;
        }
    }


    // ★ INCREMENTAL-WRITE ARM. Chases the e2e failure the single-call arms cannot see: paged q8_0
    // produces garbage in a real server while passing every op-level case here. f16 runs the SAME
    // split as a control -- if f16 also diverges the fault is the split harness, not quantisation.
    // ⚠ Placed BEFORE ggml_backend_free. An earlier arm went in after it and died on
    // GGML_ASSERT(device); test arms live on the same side of the lifecycle boundary as the backend.
    {
        const ggml_type kts[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
        for (size_t ki = 0; ki < sizeof(kts)/sizeof(kts[0]); ++ki)
        for (size_t di = 0; di < sizeof(dims)/sizeof(dims[0]); ++di) {
            const int D  = dims[di];
            if (!tile_fits(D)) { continue; }
            const int BS = getenv("DS4P_TEST_BS") ? atoi(getenv("DS4P_TEST_BS")) : 16;
            const int NB = getenv("DS4P_TEST_NB") ? atoi(getenv("DS4P_TEST_NB")) : 2;
            const int N  = BS*NB - BS/2;

            // ★ TWO SPLITS, AND THE SECOND ONE IS THE POINT.
            //   N/2 -> second call is a multi-token PREFILL dispatch
            //   N-1 -> second call is n_tokens == 1, i.e. the DECODE dispatch
            // Until this arm existed, test-paged-vs-cpu had NEVER ONCE dispatched n_tokens=1. Every
            // "q8_0 PASS at all six head_dims and both block sizes" this file has ever printed was
            // true of the PREFILL path alone -- and the decode branch reads the cache directly as
            // half at six sites with no dequant, so a quantised cache was garbage there and no
            // amount of widening head_dim or block_size could ever have shown it. The gate covered
            // one of the kernel's two branches and its verdict named neither.
            // n_dec = number of TRAILING single-token decode dispatches.
            //   1 -> the decode reads only prefill-written tokens plus the one it just wrote
            //   2 -> the second decode reads a token a PREVIOUS DECODE DISPATCH wrote  <-- the
            //        shape the server actually fails on, and the one n_dec=1 cannot express
            //   3 -> confirms it is not specific to the first repeat
            const int decs[] = { 1, 2, 3 };
            for (size_t si = 0; si < sizeof(decs)/sizeof(decs[0]); ++si) {
            const int n_dec = decs[si];
            const int n1    = N - n_dec;

            const std::vector<float> whole = run_paged      (backend, D, true, 0, kts[ki]);
            const std::vector<float> split = run_paged_split(backend, D, true, 0, kts[ki], n1, n_dec);

            double max_abs = 0.0;
            for (size_t i = 0; i < whole.size() && i < split.size(); ++i) {
                const double d = fabs((double) whole[i] - split[i]);
                max_abs = d > max_abs ? d : max_abs;
            }
            // Same arithmetic in the same order -- only the WRITE SCHEDULE differs, so this is an
            // exactness check, not a tolerance. Anything above f32 noise is the incremental path.
            const bool ok = max_abs < 1e-5;
            printf("incremental %-5s D=%3d prefill=%2d + %d decode(s)/%d: max_abs=%.3e %s\n",
                   ggml_type_name(kts[ki]), D, n1, n_dec, N, max_abs, ok ? "PASS" : "FAIL");
            n_fail += ok ? 0 : 1;
            }
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(backend);

    printf(n_fail == 0 ? "test-paged-vs-cpu: ALL PASSED\n" : "test-paged-vs-cpu: %d FAILED\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
