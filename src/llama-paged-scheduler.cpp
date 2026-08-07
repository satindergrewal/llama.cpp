#include "ggml.h"
#include "llama-context.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-impl.h"
#include "llama-paged-scheduler-impl.h"

struct llama_paged_scheduler {
    llama_paged_scheduler_impl impl;

    // ★ Last verdict from step(). prepare_batch collapses DEADLOCK into `false`, which the caller
    // cannot tell apart from the ordinary "nothing admitted this tick". Measured consequence: the
    // scheduler logged "Scheduler deadlock detected" 4,170,155 times, once per tick, with correct
    // counts and an actionable hint -- and the server's `if (!success ...) return;` discarded every
    // one of them. A correct diagnosis with no channel to the caller is not a diagnosis.
    //
    // Additive on purpose: changing prepare_batch's return type would break every existing caller
    // for a fact they can now simply ask for.
    bool last_deadlock = false;

    llama_paged_scheduler(uint32_t n_ctx, uint32_t block_sz, uint32_t n_batch, llama_kv_cache_paged * kv_manager) :
        impl(n_ctx, block_sz, n_batch, kv_manager) {}
};

LLAMA_API struct llama_paged_scheduler * llama_paged_scheduler_init(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    // Get the paged kv cache
    bool   is_hybrid = false;
    auto * paged_kv  = dynamic_cast<llama_kv_cache_paged *>(ctx->get_memory());
    if (!paged_kv) {
        // Hybrid archs: the pool lives inside the hybrid wrapper, not as the whole memory.
        // BOTH wrappers must be tried. Resolving only the ISWA type meant a hybrid WITHOUT SWA
        // fell through to the error below and was told SWA was unsupported -- on Ornith, which
        // reports n_swa = 0. Checked ISWA first only because it is the narrower type.
        if (auto * hyb_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(ctx->get_memory())) {
            paged_kv = hyb_iswa->get_mem_attn_paged();
            if (paged_kv) {
                LLAMA_LOG_INFO("%s: using the hybrid-iswa wrapper's paged attention pool\n", __func__);
                is_hybrid = true;
            }
        } else if (auto * hyb = dynamic_cast<llama_memory_hybrid *>(ctx->get_memory())) {
            paged_kv = hyb->get_mem_attn_paged();
            if (paged_kv) {
                LLAMA_LOG_INFO("%s: using the hybrid wrapper's paged attention pool\n", __func__);
                is_hybrid = true;
            }
        } else if (auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(ctx->get_memory())) {
            // PURE SWA (gemma-3, llama-4 class): not a hybrid at all, so neither branch above
            // matched and the scheduler used to report "no paged cache" for a model whose pool
            // was live. Third wrapper, same resolution.
            paged_kv = iswa->get_mem_attn_paged();
            if (paged_kv) {
                LLAMA_LOG_INFO("%s: using the iswa (pure-SWA) cache's paged attention pool\n", __func__);
                is_hybrid = true;
            }
        }
    }
    if (!paged_kv) {
        // Report WHICH precondition failed. The old text blamed SWA unconditionally, which
        // sent a debugging session down an SWA path on a model with n_swa = 0. Name what was
        // actually found instead of guessing at the cause.
        const char * kind =
            dynamic_cast<llama_memory_hybrid_iswa *>(ctx->get_memory()) ? "hybrid-iswa wrapper, but it holds no paged pool" :
            dynamic_cast<llama_memory_hybrid      *>(ctx->get_memory()) ? "hybrid wrapper, but it holds no paged pool"      :
                                                                         "a non-paged memory type";
        LLAMA_LOG_ERROR(
            "%s: context does not have a paged KV cache: found %s. "
            "Pass --kv-paged (-kvp); if you did, this model's memory was built without a paged "
            "attention pool (hybrid bring-up requires DS4P_PAGED_HYBRID=1; SWA archs such as "
            "gemma3/llama4 are not wired yet).\n",
            __func__, kind);
        return nullptr;
    }

    // Extract params
    const uint32_t n_ctx    = ctx->n_ctx();
    const uint32_t block_sz = ctx->block_size();
    const uint32_t n_batch  = ctx->n_batch();
    GGML_ASSERT(n_batch == ctx->n_ubatch() && "kv_paged requires n_batch == n_ubatch.");

    try {
        auto * sched = new llama_paged_scheduler(n_ctx, block_sz, n_batch, paged_kv);
        sched->impl.set_hybrid(is_hybrid);
        return sched;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: Error when creating llama_paged_scheduler: %s\n", __func__, e.what());
        return nullptr;
    }
}

LLAMA_API int32_t llama_paged_debug_seq_kv_checksum(const struct llama_paged_scheduler * sched,
                                                    int32_t  request_id,
                                                    int32_t  n_tokens,
                                                    double * out_sums,
                                                    int32_t  max_layers) {
    if (!sched || !out_sums) {
        return -1;
    }
    const llama_sequence_group * group = sched->impl.get_group_from_id(request_id);
    if (!group) {
        return -1;
    }
    return sched->impl.kv_cache()->debug_seq_kv_checksum(*group, n_tokens, out_sums, max_layers);
}

LLAMA_API void llama_paged_scheduler_free(struct llama_paged_scheduler * sched) {
    if (sched) {
        delete sched;
    }
}

LLAMA_API bool llama_paged_scheduler_prepare_batch(struct llama_paged_scheduler * sched, struct llama_batch * batch) {
    if (!sched || !batch) {
        return false;
    }

    llama_scheduler_status status;
    try {
        status = sched->impl.step(*batch);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }

    sched->last_deadlock = (status == llama_scheduler_status::DEADLOCK);

    if (status == llama_scheduler_status::DEADLOCK) {
        LLAMA_LOG_ERROR("%s: Deadlock detected.\n", __func__);
        return false;
    }

    return true;
}

LLAMA_API bool llama_paged_scheduler_add_request(struct llama_paged_scheduler * sched,
                                                 const llama_token *            tokens,
                                                 int32_t                        n_tokens,
                                                 int32_t                        request_id,
                                                 int32_t                        n_warm) {
    if (!sched || !tokens) {
        return false;
    }

    llama_sequence_group group;
    group.request_id = request_id;
    group.n_prompt   = n_tokens;
    group.n_decoded  = 0;
    group.n_past     = 0;
    for (int i = 0; i < n_tokens; ++i) {
        group.logical_seq.push_back(tokens[i]);
    }
    group.t_arrival_time = ggml_time_us();  // int64_t milliseconds

    return sched->impl.queue_request(group, n_warm > 0 ? (uint32_t) n_warm : 0u);
}

LLAMA_API bool llama_paged_scheduler_fork_request(struct llama_paged_scheduler * sched,
                                                  const llama_token *            tokens,
                                                  int32_t                        n_tokens,
                                                  int32_t                        request_id,
                                                  int32_t                        parent_request_id) {
    if (!sched || !tokens) {
        return false;
    }

    llama_sequence_group group;
    group.request_id = request_id;
    group.n_prompt   = n_tokens;
    group.n_decoded  = 0;
    group.n_past     = 0;
    for (int i = 0; i < n_tokens; ++i) {
        group.logical_seq.push_back(tokens[i]);
    }
    group.t_arrival_time = ggml_time_us();

    return sched->impl.queue_forked_request(group, parent_request_id);
}

LLAMA_API void llama_paged_scheduler_update(struct llama_paged_scheduler * sched,
                                            struct llama_batch *           batch,
                                            const llama_token *            tokens,
                                            const int8_t *                 stop_flags,
                                            const int32_t *                n_accepted) {
    if (!sched || !batch || !tokens || !stop_flags) {
        return;
    }

    const auto * info = sched->impl.get_curr_batch_info();
    GGML_ASSERT(info != nullptr && "no batch info was set.");

    // ★ STEP D of paged speculation. n_accepted is OPTIONAL: null keeps today's exact behaviour --
    // one accepted token per sequence, `tokens` indexed [i] -- so the two out-of-server callers
    // (tests/test-paged-kv-e2e.cpp, examples/paged/paged.cpp) compile and behave unchanged.
    //
    // When present, `tokens` is laid out at the BATCH offsets rather than one-per-sequence, because
    // sequences can submit different counts and a [i * n_sub] layout would read the wrong sequence's
    // tokens the moment two drafts differ in length.
    const size_t n_tok = n_accepted
        ? (size_t) info->batch_offsets[info->n_seq - 1] + (size_t) info->batch_lens[info->n_seq - 1]
        : (size_t) info->n_seq;
    std::vector<llama_token> tokens_vec(tokens, tokens + n_tok);
    sched->impl.update(*batch, tokens_vec, stop_flags, n_accepted);
}

LLAMA_API bool llama_paged_scheduler_set_draft(struct llama_paged_scheduler * sched,
                                              int32_t                        request_id,
                                              const llama_token *            draft,
                                              int32_t                        n_draft) {
    if (!sched) {
        return false;
    }
    return sched->impl.set_draft(request_id, draft, n_draft);
}

LLAMA_API void llama_paged_scheduler_set_on_finish(struct llama_paged_scheduler * sched,
                                                   llama_paged_on_finish_cb       cb,
                                                   void *                         user_data) {
    sched->impl.set_on_finish(cb, user_data);
}

LLAMA_API bool llama_paged_scheduler_get_seq_state(struct llama_paged_scheduler * sched,
                                                   int32_t                        request_id,
                                                   struct llama_paged_seq_state * out_state) {
    if (!sched || !out_state) {
        return false;
    }

    llama_sequence_group * group = sched->impl.get_group_from_id(request_id);
    if (group == nullptr) {
        LLAMA_LOG_ERROR("%s: request_id=%d does not exist.", __func__, request_id);
        return false;
    }

    out_state->request_id       = group->request_id;
    out_state->n_prompt         = group->n_prompt;
    out_state->n_decoded        = group->n_decoded;
    out_state->n_past           = group->n_past;
    out_state->t_arrival_us     = group->t_arrival_time;
    out_state->t_first_token_us = group->t_first_token_us;
    out_state->n_logical        = (int32_t) group->logical_seq.size();
    return true;
}

// True when the LAST prepare_batch returned false because the scheduler declared a deadlock,
// as opposed to simply having nothing to admit. See the note on last_deadlock.
// Drain the ids the scheduler terminated for CAPACITY (not normal completions). Returns how many
// were written to `out`, and clears the internal list.
LLAMA_API int32_t llama_paged_scheduler_take_terminated(struct llama_paged_scheduler * sched,
                                                       int32_t * out, int32_t max_out) {
    if (!sched || !out || max_out <= 0) { return 0; }
    int32_t n = 0;
    for (int32_t id : sched->impl.terminated_ids) {
        if (n >= max_out) { break; }
        out[n++] = id;
    }
    sched->impl.terminated_ids.clear();
    return n;
}

LLAMA_API bool llama_paged_scheduler_last_was_deadlock(const struct llama_paged_scheduler * sched) {
    return sched != nullptr && sched->last_deadlock;
}

LLAMA_API const struct llama_paged_batch_info * llama_paged_scheduler_get_batch_info(
    const struct llama_paged_scheduler * sched) {
    if (!sched) {
        return nullptr;
    }
    return sched->impl.get_curr_batch_info();
}
