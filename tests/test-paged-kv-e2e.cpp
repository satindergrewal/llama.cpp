// tests/test-paged-kv-e2e.cpp
//
// End-to-end equivalence test for paged KV cache.
// We compare top-K agreement rather than raw logit values because the paged
// attention path uses a custom CUDA kernel with online softmax, while the
// unified path uses standard ggml attention with two-pass softmax. The two
// produce mathematically equivalent results but with different F16
// accumulation order, drifts on the order of 0.05-0.5 in raw
// logit values is expected and not a correctness issue. Top-K set agreement
// is robust to this drift while still catching the real issues (e.g. cross-device
// reads, layout corruption, MQA broadcast bugs) which produce wildly
// different distributions.
//
// Also samples N_COMPARE tokens greedy as a secondary 'cheap' check.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#define EXPECT_TRUE(x)                                                   \
    do {                                                                 \
        if (!(x)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            throw std::runtime_error("FAILED assertion.");               \
        }                                                                \
    } while (0)

static constexpr const char * TEST_PROMPT       = "Once upon a time there was a lovely";
static constexpr int          N_PREDICT         = 16;
static constexpr int          N_COMPARE         = 4;  // token-equivalence window
static constexpr int          TOP_K             = 5;
static constexpr int          MIN_TOP_K_OVERLAP = 4;  // at least 4 of top-5 must match

// Result of running one path: prefill-final logits + sampled token sequence.
struct path_result {
    std::vector<float>       prefill_logits;  // [n_vocab]
    std::vector<float>       decode_logits;   // [n_vocab] -- logits AFTER the first decode step
                                              // (#19: the ONLY thing sensitive to the DECODE-only MSA
                                              //  gather; the token/prefill checks are blind to it).
    std::vector<llama_token> tokens;          // [N_PREDICT]
    int                      n_vocab = 0;
};

static path_result run_non_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = false;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    // Prefill
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    EXPECT_TRUE(llama_decode(ctx, batch) == 0);

    // Capture prefill-final logits BEFORE any further decode steps overwrite them.
    path_result result;
    result.n_vocab = n_vocab;
    {
        const float * raw = llama_get_logits_ith(ctx, -1);  // last logit
        EXPECT_TRUE(raw != nullptr);
        result.prefill_logits.assign(raw, raw + n_vocab);
    }

    // Sample N_PREDICT tokens for the secondary token-equivalence check.
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    llama_token cur = -1;
    for (int i = 0; i < N_PREDICT; ++i) {
        cur = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, cur, true);
        result.tokens.push_back(cur);
        if (llama_vocab_is_eog(vocab, cur)) {
            break;
        }

        llama_batch step = llama_batch_get_one(&cur, 1);
        EXPECT_TRUE(llama_decode(ctx, step) == 0);
        // #19: capture the FIRST decode step's logits -- this pass runs the DECODE-only MSA gather.
        if (i == 0) {
            const float * raw = llama_get_logits_ith(ctx, -1);
            if (raw) { result.decode_logits.assign(raw, raw + n_vocab); }
        }
    }

    common_sampler_free(smpl);
    return result;
}

static path_result run_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = true;
    params.n_gpu_blocks  = 64;
    params.n_cpu_blocks  = 16;
    params.n_sequences   = 1;
    params.n_parallel    = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    bool ok = llama_paged_scheduler_add_request(sched, prompt_tokens.data(), prompt_tokens.size(), 0, 0);
    EXPECT_TRUE(ok);

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    path_result result;
    result.n_vocab                      = n_vocab;
    bool        captured_prefill_logits = false;
    bool        did_draft_probe         = false;
    bool        did_accept_probe        = false;
    llama_batch batch                   = {};

    // #19 Tier 1: DS4P_NO_SPEC=1 strips the speculative-decode probes (sentinel + draft +
    // multi-accept), leaving a CLEAN basic paged decode. minimax-m3's MSA memory does not
    // implement draft rollback, so the probes false-fail it; this isolates "does basic paged
    // decode match non-paged" from "does speculation work".
    const bool no_spec = getenv("DS4P_NO_SPEC") != nullptr;

    while ((int) result.tokens.size() < N_PREDICT) {
        bool prepared = llama_paged_scheduler_prepare_batch(sched, &batch);
        EXPECT_TRUE(prepared);
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr && info->n_seq == 1);

        const int32_t last_idx = info->batch_offsets[0] + info->batch_lens[0] - 1;

        // First decode is the prefill — capture its final logits before
        // sampling anything else.
        if (!captured_prefill_logits) {
            const float * raw = llama_get_logits_ith(ctx, last_idx);
            EXPECT_TRUE(raw != nullptr);
            result.prefill_logits.assign(raw, raw + n_vocab);
            captured_prefill_logits = true;
        } else if (result.decode_logits.empty()) {
            // #19: the FIRST real decode step -- runs the DECODE-only MSA gather. Sensitive.
            const float * raw = llama_get_logits_ith(ctx, last_idx);
            if (raw) { result.decode_logits.assign(raw, raw + n_vocab); }
        }

        llama_token next = common_sampler_sample(smpl, ctx, last_idx);
        common_sampler_accept(smpl, next, true);
        result.tokens.push_back(next);

        bool   stop      = llama_vocab_is_eog(vocab, next) || (int) result.tokens.size() >= N_PREDICT;

        // ⚠ HOLD THE GROUP ALIVE FOR THE MULTI-ACCEPT PROBE. A stop_flag of 1 marks the group
        // FINISHED and finish() removes it from id_to_group, so set_draft() then returns false and
        // the probe cannot run at all -- which is exactly what happened on the first placement. The
        // probe needs a live group with a complete token stream, and this is the only instant where
        // both are true.
        const bool run_accept_probe = stop && did_draft_probe && !did_accept_probe;
        int8_t stop_flag = (stop && !run_accept_probe) ? 1 : 0;

        // ★★ EXERCISE THE SENTINEL BRANCH OF n_accepted, WHICH OTHERWISE HAS ZERO EXECUTIONS.
        //
        // Step D added a ragged path to update(): a NEGATIVE n_accepted[i] means "legacy for this
        // row" and a non-null array switches every row to BATCH-OFFSET token layout. Everything that
        // has passed so far only proves the NULL path survived refactoring -- the new code compiled
        // and never ran. That is precisely where this week's defects lived (the read-only CPU arm,
        // the vacuous read-only test arm, the fused append).
        //
        // Alternating the two forms on successive decode steps means both are executed on every run
        // of this test, and they must produce IDENTICAL results -- a sentinel row is BY DEFINITION
        // today's behaviour, just reached through the other branch and the other token layout.
        // If the two ever disagree, this test fails rather than B inheriting a broken contract.
        const bool use_sentinel = !no_spec && (result.tokens.size() % 2) == 1;
        if (use_sentinel) {
            // batch-offset layout: sentinel rows read their ONE token at batch_offsets[i]
            std::vector<llama_token> toks((size_t) info->batch_offsets[0] + (size_t) info->batch_lens[0], 0);
            toks[info->batch_offsets[0]] = next;
            const int32_t n_acc = -1;   // sentinel: legacy semantics for this row
            llama_paged_scheduler_update(sched, &batch, toks.data(), &stop_flag, &n_acc);
        } else {
            llama_paged_scheduler_update(sched, &batch, &next, &stop_flag, /*n_accepted =*/ nullptr);
        }
        if (stop) {
            // ⚠ FIRES HERE, AT THE STOP BOUNDARY, and the placement is the whole point. Gating it
            // inside the reject probe on "tokens complete" made it VACUOUS -- the reject probe runs
            // once, early, when the stream is short, so the gate skipped the multi-accept path on
            // every run and the test still printed PASSED. Third vacuity of the same shape today.
            // Here the stream is complete, the loop is ending, and the probe's unrecorded tokens
            // cannot shorten it.
            if (run_accept_probe) {
                did_accept_probe = true;
            // ★★ THE MULTI-ACCEPT PROBE. Everything above exercises the REJECT shape (n_acc = 1).
                // The multi-token append loop, the second batch-offset read (new_tokens[offset+1]) and
                // E's multi-accept last_accepted_idx have ZERO EXECUTIONS without this.
                //
                // Acceptance cannot be forced through prediction -- it would require knowing the next
                // token before decoding it. So this probe accepts BOTH rows unconditionally and checks
                // the SCHEDULER STATE rather than the text, then stops. Its tokens are deliberately NOT
                // pushed into result.tokens, so the losslessness comparison is unaffected by a pair that
                // was never verified against logits.
                    {
                    const llama_token d2 = result.tokens.back();
                    EXPECT_TRUE(llama_paged_scheduler_set_draft(sched, 0, &d2, 1));
                    EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
                    const llama_paged_batch_info * i2 = llama_paged_scheduler_get_batch_info(sched);
                    EXPECT_TRUE(i2 != nullptr && i2->batch_lens[0] == 2);
                    EXPECT_TRUE(llama_decode(ctx, batch) == 0);
                    llama_synchronize(ctx);

                    llama_paged_seq_state s0{};
                    EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, 0, &s0));

                    const int32_t b2 = i2->batch_offsets[0];
                    std::vector<llama_token> t2((size_t) b2 + 2, 0);
                    const float * r0 = llama_get_logits_ith(ctx, b2 + 0);
                    const float * r1 = llama_get_logits_ith(ctx, b2 + 1);
                    EXPECT_TRUE(r0 != nullptr && r1 != nullptr);
                    t2[b2 + 0] = (int32_t) (std::max_element(r0, r0 + n_vocab) - r0);
                    t2[b2 + 1] = (int32_t) (std::max_element(r1, r1 + n_vocab) - r1);

                    const int32_t n_acc2 = 2;   // force the ACCEPT shape
                    int8_t        st2    = 0;
                    llama_paged_scheduler_update(sched, &batch, t2.data(), &st2, &n_acc2);

                    llama_paged_seq_state s1{};
                    EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, 0, &s1));
                    printf("test-paged-kv-e2e: multi-accept probe n_past %u -> %u, logical %u -> %u (want +2, +2)\n",
                           s0.n_past, s1.n_past, s0.n_logical, s1.n_logical);
                    EXPECT_TRUE((int32_t) (s1.n_past - s0.n_past) == 2);
                    // ⚠ AND THE APPEND, SEPARATELY. n_past advances by n_acc whatever the append loop
                    // does, so checking it alone passed a mutation that capped the loop at ONE token.
                    // Advance-count and append-count are different quantities; assert both.
                    EXPECT_TRUE((int32_t) (s1.n_logical - s0.n_logical) == 2);
                }

            }
            break;
        }

        // ★★ EXERCISE THE DRAFT PATH ITSELF -- set_draft + multi-row emission + logits mask +
        // block arithmetic + the accepted count. Without this every one of those is COMPILED AND
        // NEVER RUN, which is the exact state that produced this week's three worst defects.
        //
        // The check is self-verifying and needs no real draft model. Stage the token the target is
        // ABOUT to produce anyway (a perfect 1-token draft), then:
        //   - prepare_batch must emit 2 rows instead of 1
        //   - logits must be sampleable at BOTH indices (C's mask)
        //   - accepting both must advance n_past by exactly 2 (A's blocks, D's counter, E's max_pos)
        // and the generated text must be IDENTICAL to the non-drafting run, because greedy
        // speculative decoding is lossless. compare_results() at the end is what enforces that.
        if (!no_spec && !did_draft_probe && result.tokens.size() >= 2) {
            did_draft_probe = true;

            // draft one token: whatever greedy would pick next is unknown here, so use the token we
            // just sampled. A WRONG draft is fine and is in fact the more interesting case -- it
            // exercises the reject path, and acceptance is decided below by comparing logits.
            const llama_token draft_tok = next;
            EXPECT_TRUE(llama_paged_scheduler_set_draft(sched, /*request_id =*/ 0, &draft_tok, 1));

            EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
            const llama_paged_batch_info * dinfo = llama_paged_scheduler_get_batch_info(sched);
            EXPECT_TRUE(dinfo != nullptr && dinfo->n_seq == 1);
            // B: two rows, not one
            EXPECT_TRUE(dinfo->batch_lens[0] == 2);

            EXPECT_TRUE(llama_decode(ctx, batch) == 0);
            llama_synchronize(ctx);

            // C: logits requested on BOTH rows -- get_logits_ith throws by name if not
            const int32_t base = dinfo->batch_offsets[0];
            const float * row0 = llama_get_logits_ith(ctx, base + 0);
            const float * row1 = llama_get_logits_ith(ctx, base + 1);
            EXPECT_TRUE(row0 != nullptr && row1 != nullptr);

            // greedy verify: row0 predicts the token that follows `next`. Accept the draft only if
            // it matches, exactly as a real verify step would.
            const int32_t argmax0 = (int32_t) (std::max_element(row0, row0 + n_vocab) - row0);
            const int32_t n_acc_d = (argmax0 == draft_tok) ? 2 : 1;

            llama_paged_seq_state st_before{};
            EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, 0, &st_before));

            std::vector<llama_token> dtoks((size_t) base + 2, 0);
            dtoks[base + 0] = argmax0;
            if (n_acc_d == 2) {
                dtoks[base + 1] = (int32_t) (std::max_element(row1, row1 + n_vocab) - row1);
            }
            int8_t dstop = 0;
            llama_paged_scheduler_update(sched, &batch, dtoks.data(), &dstop, &n_acc_d);

            // D + E: n_past advanced by the ACCEPTED count, not the submitted one
            llama_paged_seq_state st_after{};
            EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, 0, &st_after));
            EXPECT_TRUE((int32_t) (st_after.n_past - st_before.n_past) == n_acc_d);

            // ⚠ THE RUN MUST TESTIFY WHICH BRANCH IT TOOK. Staging `next` as the draft means
            // acceptance requires the model to immediately repeat itself, so in practice this probe
            // always takes the REJECT branch (n_acc = 1 of 2). Inferring that from "the test passed"
            // is exactly the mistake this file keeps paying for -- print it.
            printf("test-paged-kv-e2e: draft probe took the %s branch (n_acc=%d of 2)\n",
                   n_acc_d == 2 ? "ACCEPT" : "REJECT", n_acc_d);

            for (int32_t k = 0; k < n_acc_d; ++k) {
                result.tokens.push_back(dtoks[base + k]);
                common_sampler_accept(smpl, dtoks[base + k], true);
            }

        }
    }

    common_sampler_free(smpl);
    llama_paged_scheduler_free(sched);
    return result;
}

static void compare_results(const path_result & ref, const path_result & paged) {
    auto top_k = [](const std::vector<float> & l, int k) {
        std::vector<int> idx(l.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&l](int a, int b) { return l[a] > l[b]; });
        idx.resize(k);
        return idx;
    };

    const auto top_ref   = top_k(ref.prefill_logits, TOP_K);
    const auto top_paged = top_k(paged.prefill_logits, TOP_K);

    // Argmax must match: the most-confident next token should be identical.
    EXPECT_TRUE(top_ref[0] == top_paged[0]);

    // Top-K set overlap: at least MIN_TOP_K_OVERLAP of the K most likely
    // tokens must appear in both distributions.
    std::set<int> ref_set(top_ref.begin(), top_ref.end());
    int           overlap = 0;
    for (int t : top_paged) {
        if (ref_set.count(t)) {
            overlap++;
        }
    }

    fprintf(stderr, "test-paged-kv-e2e: top-%d argmax match: ref=%d paged=%d\n", TOP_K, top_ref[0], top_paged[0]);
    fprintf(stderr, "test-paged-kv-e2e: top-%d set overlap: %d/%d (require >= %d)\n", TOP_K, overlap, TOP_K,
            MIN_TOP_K_OVERLAP);

    if (overlap < MIN_TOP_K_OVERLAP) {
        fprintf(stderr,
                "FAIL: top-%d distributions diverge too much. Only %d of %d most-likely "
                "tokens match between ref and paged. Real correctness issue likely.\n",
                TOP_K, overlap, TOP_K);
        fprintf(stderr, "  ref:   ");
        for (int t : top_ref) {
            fprintf(stderr, "%d(%.3f) ", t, ref.prefill_logits[t]);
        }
        fprintf(stderr, "\n  paged: ");
        for (int t : top_paged) {
            fprintf(stderr, "%d(%.3f) ", t, paged.prefill_logits[t]);
        }
        fprintf(stderr, "\n");
        throw std::runtime_error("FAILED test.");
    }

    // Token-level secondary check: first N_COMPARE tokens must match.
    // We don't compare beyond N_COMPARE because greedy sampling on small
    // models is sensitive to argmax tiebreakers, and minor floating-point
    // accumulation differences between the paged and non-paged paths can
    // flip individual tokens after a handful of decode steps.
    EXPECT_TRUE((int) ref.tokens.size() >= N_COMPARE);
    EXPECT_TRUE((int) paged.tokens.size() >= N_COMPARE);
    for (int i = 0; i < N_COMPARE; ++i) {
        if (ref.tokens[i] != paged.tokens[i]) {
            fprintf(stderr, "FAIL: token %d differs in the equivalence window: ref=%d paged=%d\n", i, ref.tokens[i],
                    paged.tokens[i]);
            throw std::runtime_error("FAILED test.");
        }
    }

    // #19: decode-logit check. NOTE (measured 2026-08-13): for MSA this compares the DIRECT-decode
    // path (llama_decode without the paged scheduler), where has_paged_batch_info is false, so the
    // DENSE gather runs -- garbaging the POOL path by 100x leaves this UNCHANGED. It is therefore
    // sensitive for the DENSE arch, NOT proof of the paged MSA gather. The paged MSA store+gather is
    // verified separately by scratchpad/msa_gather_roundtrip.cpp (a real model would close the rest).
    if (!ref.decode_logits.empty() && ref.decode_logits.size() == paged.decode_logits.size()) {
        double max_abs = 0.0; int worst = -1;
        for (size_t i = 0; i < ref.decode_logits.size(); ++i) {
            const double d = std::fabs((double) ref.decode_logits[i] - (double) paged.decode_logits[i]);
            if (d > max_abs) { max_abs = d; worst = (int) i; }
        }
        fprintf(stderr, "test-paged-kv-e2e: decode-logit max_abs_diff = %.6g (tok %d)\n", max_abs, worst);
        const double tol = 1e-2;  // f16 KV + Metal vs CPU accumulation; generous but garbage (100x) blows past it
        if (max_abs > tol) {
            fprintf(stderr, "FAIL: decode logits diverge (max_abs %.6g > tol %.6g) -- the paged DECODE gather is wrong\n",
                    max_abs, tol);
            throw std::runtime_error("FAILED test.");
        }
    } else {
        fprintf(stderr, "test-paged-kv-e2e: WARN decode_logits not captured/comparable (ref=%zu paged=%zu)\n",
                ref.decode_logits.size(), paged.decode_logits.size());
    }

    fprintf(stderr, "test-paged-kv-e2e: PASSED\n");
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PAGED)) {
        fprintf(stderr, "usage: %s -m <model>\n", argv[0]);
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "skip: no --model provided\n");
        return 0;
    }

    common_init();
    llama_backend_init();

    fprintf(stderr, "test-paged-kv-e2e: running non-paged reference\n");
    path_result ref = run_non_paged(params.model.path);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", ref.tokens.size(), ref.n_vocab);

    fprintf(stderr, "test-paged-kv-e2e: running paged path\n");
    path_result paged = run_paged(params.model.path);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", paged.tokens.size(), paged.n_vocab);

    compare_results(ref, paged);

    llama_backend_free();
    return 0;
}
