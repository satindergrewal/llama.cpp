#include "llama-paged-scheduler-impl.h"

#include "llama-impl.h"

#include <memory>

llama_paged_scheduler_impl::llama_paged_scheduler_impl(uint32_t               n_ctx,
                                                       uint32_t               block_sz,
                                                       int32_t                n_batch,
                                                       llama_kv_cache_paged * kv_manager,
                                                       uint32_t               n_seq_max_batch) :
    n_seq_max_ctx(n_ctx),
    block_size(block_sz),
    n_batch(n_batch),
    n_seq_max_batch(n_seq_max_batch),
    kv_cache_manager(kv_manager) {}

llama_paged_scheduler_impl::~llama_paged_scheduler_impl() {
    for (auto & h : held_prefixes) {
        if (h && kv_cache_manager && !h->block_table.empty()) {
            kv_cache_manager->free_blocks(*h);
        }
    }
    held_prefixes.clear();
}

bool llama_paged_scheduler_impl::check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const {
    if (n_candidates > 0) {
        return false;
    }
    if (n_swapped == 0 && n_waiting == 0) {
        return false;
    }
    // A live running group still holds blocks (the shared master prefix, or a
    // child that could not grow this tick). Waiters resume when a child leaves
    // and frees its unref tail. That is vLLM "waiting", not a deadlock -- the
    // server used to 500 everyone here.
    if (!running.empty()) {
        return false;
    }
    // Parked requests that would fit in the pool alone are waiting for a child
    // to leave, not unsatisfiable. Only report deadlock when every parked
    // request itself outgrew the entire pool.
    const uint32_t usable = kv_cache_manager->get_usable_gpu_blocks();
    auto fits = [&](const llama_sequence_group_ptr & g) {
        if (!g) {
            return false;
        }
        // Waiting admits from n_prompt; a forked child already holds n_past.
        const uint32_t tokens = g->n_past > 0 ? g->n_past + 1 : g->n_prompt + 1;
        const uint32_t blocks = (tokens + block_size - 1) / block_size;
        return blocks <= usable;
    };
    for (const auto & g : waiting) {
        if (fits(g)) {
            return false;
        }
    }
    for (const auto & g : swapped) {
        if (fits(g)) {
            return false;
        }
    }
    LLAMA_LOG_ERROR(
        "%s: Scheduler deadlock detected. "
        "%d sequence(s) are swapped out and %d are waiting, "
        "and every parked request outgrew the GPU block pool. "
        "Hint: increase n_gpu_blocks (currently %d) or reduce n_sequences.\n",
        __func__, n_swapped, n_waiting, kv_cache_manager->get_num_gpu_blocks());
    return true;
}

bool llama_paged_scheduler_impl::check_livelock(uint32_t n_swapped, uint32_t prev_n_swapped) {
    // swapped count is non-zero and not decreasing
    if (n_swapped > 0 && n_swapped >= prev_n_swapped) {
        n_livelock_steps++;
        if (n_livelock_steps >= max_livelock_steps) {
            LLAMA_LOG_ERROR(
                "%s: Livelock detected. Swapped count has been "
                "non-decreasing for %d steps (currently %d swapped). "
                "Increase n_gpu_blocks (currently %d) or reduce "
                "n_sequences.\n",
                __func__, n_livelock_steps, n_swapped, kv_cache_manager->get_num_gpu_blocks());
            return true;
        }
    } else {
        // Swapped count decreased — swap-ins are happening, reset counter
        n_livelock_steps = 0;
    }
    return false;
}

int32_t llama_paged_scheduler_impl::get_curr_decode_tokens() const {
    return running.size();
}

llama_scheduler_status llama_paged_scheduler_impl::step(llama_batch & batch) {
    // Free previous inference batches
    clear_batch(batch);

    llama_sequence_group_raw_list candidates;
    process_running_list(candidates);
    process_swapped_list(candidates);

    // Cap the live decode set at n_seq_max_batch. Running groups past the
    // cap stay in `running` with their blocks and take the next step. That
    // is the champion contract at -np 1: one sequence per decode, many
    // sequences in the pool. Do NOT promote waiters once the cap is full --
    // allocate() would take blocks they cannot use this step and can starve
    // the group that is actually in the batch.
    if (n_seq_max_batch > 0 && candidates.size() > (size_t) n_seq_max_batch) {
        candidates.resize((size_t) n_seq_max_batch);
    }

    const bool room_for_waiters =
        n_seq_max_batch == 0 || candidates.size() < (size_t) n_seq_max_batch;
    if (room_for_waiters) {
        const int32_t remaining = n_batch - (int32_t) candidates.size();
        process_waiting_list(candidates, remaining);
        if (n_seq_max_batch > 0 && candidates.size() > (size_t) n_seq_max_batch) {
            candidates.resize((size_t) n_seq_max_batch);
        }
    }

    const uint32_t n_running    = running.size();
    const uint32_t n_swapped    = swapped.size();
    const uint32_t n_waiting    = waiting.size();
    const uint32_t n_candidates = candidates.size();

    // ⚠ DEBUG, NOT INFO. This fires once per scheduler step INCLUDING every idle step, so an idle
    // server emitted 15 MB/s of "running=0, swapped=0, waiting=0, candidates=0" at -lv 4 -- 898 MB
    // in about a minute. Two costs, and the second is worse than the disk:
    //   1. it fills the disk (a day idle would be ~1.3 TB)
    //   2. it lands ONLY on the paged arm, so any paged-vs-static TIMING comparison is measuring
    //      15 MB/s of logging as if it were paging cost. arms-must-differ-in-ONE-thing.
    // Same family as the deadlock that logged itself 4.17M times into a bool.
    LLAMA_LOG_DEBUG("%s: Scheduler status: running=%d, swapped=%d, waiting=%d, candidates=%d\n", __func__, n_running,
                   n_swapped, n_waiting, n_candidates);

    const bool deadlock = check_deadlock(n_candidates, n_swapped, n_waiting);
    const bool livelock = check_livelock(n_swapped, prev_n_swapped);  // updates n_livelock_steps
    // a stall verdict is only terminal when NOTHING can make progress: with live
    // candidates, a stuck swapped request must not starve the runnable ones (the
    // true-starved P2-8 arm ticked DEADLOCK forever -- 5M detector lines -- while
    // runnable groups sat idle behind one unswappable victim)
    if ((deadlock || livelock) && n_candidates == 0) {
        return llama_scheduler_status::DEADLOCK;
    }

    prev_n_swapped = n_swapped;
    populate_batch_from(candidates, batch);
    kv_cache_manager->set_paged_batch_info(&curr_info);
    return llama_scheduler_status::OK;
}

bool llama_paged_scheduler_impl::queue_request(llama_sequence_group group, uint32_t n_warm) {
    // ★ ENTRY marker. Two requests register but only ONE runs the prefix-share scan, so the second
    // leaves before reaching it. Instrument the ENTRY before the branch -- the Gemma4 lesson.
    if (getenv("DS4P_DECODE_TRACE")) {
        LLAMA_LOG_WARN("DS4P-QREQ enter id=%d n_prompt=%u n_warm=%u kvmgr=%d logical=%zu\n",
                       group.request_id, group.n_prompt, n_warm,
                       kv_cache_manager != nullptr, group.logical_seq.size());
    }
    // Rejecting any requests that exceeds max context for a seq
    if (group.n_prompt >= n_seq_max_ctx) {
        if (kv_cache_manager != nullptr) {
            kv_cache_manager->discard_restored(group.request_id);
        }
        LLAMA_LOG_ERROR("%s: request %d exceeds max context (%d > %d).\n", __func__, group.request_id, group.n_prompt,
                        n_seq_max_ctx);
        return false;
    }

    // P1-5 WARM ADMIT. state_read has already written this sequence's KV into real blocks
    // and parked them in the cache; THIS is where they become the group's block_table --
    // the step that made the restore half impossible before, because the cache has no way
    // to reach a scheduler group that does not exist yet.
    //
    // The resulting group has exactly the shape a COW fork produces (inherited block_table
    // + n_past > 0 + full logical_seq), so it flows through the SAME prefill-from-n_past
    // path that P1-6's fork gate already proved correct. Nothing new to schedule: the disk
    // bank is just another source of a prefix.
    //
    // n_warm is the caller's cap -- only the span it has verified to be a prefix of THIS
    // request's prompt may be reused. Blocks beyond it are released inside
    // take_restored_blocks rather than handed over and silently attended to.
    if (kv_cache_manager != nullptr) {
        if (n_warm > 0 && group.block_table.empty()) {
            llama_block_ids restored;
            const uint32_t n_past = kv_cache_manager->take_restored_blocks(group.request_id, n_warm, restored);
            if (n_past > 0) {
                group.block_table = std::move(restored);
                group.n_past      = n_past;

                // "cached state", NOT "the KV bank": by this point the bytes are just bytes.
                // Whether they came from the RAM prompt cache or off disk is known only to
                // the server layer that fetched them, and a log line that names a source it
                // cannot see is how a RAM hit gets read as proof the disk path works.
                LLAMA_LOG_INFO("%s: request %d admitted WARM: %u of %u prompt tokens restored from "
                               "cached state (%zu blocks), %u left to prefill\n",
                               __func__, group.request_id, n_past, group.n_prompt,
                               group.block_table.size(), group.n_prompt - n_past);
            }
        } else if (group.block_table.empty()) {
            // ★ PREFIX SHARING across INDEPENDENT requests (the vLLM/SGLang gap).
            //
            // The scheduler already names three admission classes: WARM (prefix restored from a
            // disk bank), FORK (brings its own blocks), COLD (nothing). A request that could inherit
            // a LIVE sequence's prefix currently collapses into COLD and re-prefills from zero. This
            // is the missing fourth case, not a new concept -- the comment at the WARM path already
            // says "bank is just another source of a prefix".
            //
            // MVP is a linear scan, deliberately, not a radix trie: O(live_groups * prefix_len) is
            // nothing at these sequence counts, and it needs NO index to keep coherent under
            // eviction, preemption and block recycling -- all of which this scheduler does, and
            // where a stale trie entry pointing at a recycled block is a silent-corruption bug.
            // Verify-at-use beats invalidate-on-change here. Index it later, once it is measured.
            //
            // ⚠ THIS BRANCH IS ONLY FOR AN EMPTY TABLE. queue_forked_request already called
            // fork_blocks, restored logical_seq / n_prompt, then calls queue_request. The old
            // else was "not warm", so a fork child re-entered this scan, fork_blocks ran a
            // second time (refcount leak, no unshare) and clobbered n_prompt again. Measured
            // 2026-08-14 on the independent-share path: admitted SHARED then
            // GGML_ASSERT(remaining_prompt > 0). Same clobber. Do not scan a group that already
            // holds blocks.
            uint32_t               best_n   = 0;
            llama_sequence_group * best_src = nullptr;

            if (!group.logical_seq.empty()) {
                const uint32_t bs = kv_cache_manager->get_block_size();
                // ⚠ SCAN BOTH SETS. queue_request ends with set_waiting(), so a request in flight is
                // not necessarily in `running` when the NEXT one is admitted -- it is promoted on a
                // later step(). Scanning only `running` found an EMPTY LIST every time (measured:
                // "DS4P-SHARE scan: running=0"), so discovery never fired and the feature was dark
                // despite correct logic. Any group with computed KV is a valid prefix source
                // regardless of which queue currently holds it.
                std::vector<llama_sequence_group *> candidates;
                for (auto & g : running) { if (g) { candidates.push_back(g.get()); } }
                for (auto & g : waiting) { if (g) { candidates.push_back(g.get()); } }
                // Finished masters stay shareable: a child that arrives after
                // finish() must still hit. held_prefixes is the APC park.
                for (auto & g : held_prefixes) { if (g) { candidates.push_back(g.get()); } }
                for (auto * src_raw : candidates) {
                    auto & src_ptr = src_raw;
                    llama_sequence_group * src = src_ptr;
                    // ⚠ Do NOT skip on request_id. request_id IS the slot id and slot ids are
                    // REUSED, so two concurrent live requests can carry the same id (measured:
                    // both admissions logged id=1). Skipping on it made a live source look like
                    // "myself" and killed sharing outright. The incoming group is not yet in either
                    // queue, so identity by address is the correct and sufficient test.
                    if (src == nullptr || src == &group) { continue; }

                    // ⚠ A valid prefix source must actually HOLD BLOCKS. A group sitting in `waiting`
                    // has a logical_seq and may have n_past set, but no block_table until it is
                    // scheduled -- and fork_blocks returns 0 on an empty table, so discovery would
                    // report a 960-token match and then silently share nothing. Filter here so the
                    // scan cannot select a source it is impossible to inherit from.
                    // ★ Ask the CACHE for this sequence's blocks. group.block_table is cleared after
                    // the cache copies it, so a live prefilled sequence shows blocks=0 there while
                    // holding dozens of real blocks in sequence_blocks. Measured:
                    //     DS4P-CAND id=1 blocks=0 n_past=1169 logical=1170
                    const llama_block_ids * src_blocks =
                        kv_cache_manager->get_sequence_blocks(src->request_id);
                    if (getenv("DS4P_DECODE_TRACE")) {
                        LLAMA_LOG_WARN("DS4P-CAND id=%d cache_blocks=%zu n_past=%u logical=%zu\n",
                                       src->request_id, src_blocks ? src_blocks->size() : 0,
                                       src->n_past, src->logical_seq.size());
                    }
                    if (src_blocks == nullptr || src_blocks->empty() || src->n_past == 0) { continue; }

                    // ⚠ Cap at the source's n_past, NOT its logical_seq length. logical_seq is what
                    // the sequence WILL be; n_past is what it has actually computed into KV. Sharing
                    // blocks for tokens the source has not prefilled hands over KV that does not
                    // exist yet -- allocated but never written -- which surfaces as garbage output,
                    // not a crash.
                    const size_t lim = std::min({ src->logical_seq.size(),
                                                  group.logical_seq.size(),
                                                  (size_t) src->n_past });
                    uint32_t n = 0;
                    while (n < lim && src->logical_seq[n] == group.logical_seq[n]) { ++n; }

                    // ⚠ fork_blocks shares whole PHYSICAL blocks. A 100-token match at bs=16 may
                    // share only 96: the partial block still belongs to the source and must be
                    // COW-copied or re-prefilled, never aliased. Sharing a partially filled block is
                    // how two sequences end up writing the same cells -- the -np>1 defect.
                    n = bs ? (n / bs) * bs : 0;
                    if (n > best_n) { best_n = n; best_src = src; }
                }
            }

            if (getenv("DS4P_DECODE_TRACE")) {
                LLAMA_LOG_WARN("DS4P-SHARE scan: running=%zu best_n=%u src=%d my_prompt=%u\n",
                               running.size(), best_n,
                               best_src ? best_src->request_id : -1, group.n_prompt);
            }

            // Exact-prefix grow (second /completion on a named session with the
            // same prompt). Inheriting n_prompt tokens leaves remaining_prompt=0
            // and n_decoded=0; populate_batch_from then asserts (measured
            // 2026-08-16 Qwen 8k named session). Same contract as WARM
            // (toks.size()-1): leave at least one token so the last chunk can
            // emit logits. Whole-block share then leaves the last block
            // unshared -- safe to write, and remaining_prompt > 0.
            if (best_n > 0 && best_src != nullptr &&
                group.n_prompt > 0 && best_n >= group.n_prompt) {
                const uint32_t bs_leave = kv_cache_manager->get_block_size();
                const uint32_t leave    = group.n_prompt - 1;
                best_n = bs_leave ? (leave / bs_leave) * bs_leave : 0;
            }

            if (best_n > 0 && best_src != nullptr) {
                // fork_blocks reads src.block_table, which is empty on a live group. Give it a
                // source view whose table is the cache's authoritative copy.
                llama_sequence_group src_view = *best_src;
                if (const auto * bl = kv_cache_manager->get_sequence_blocks(best_src->request_id)) {
                    src_view.block_table = *bl;
                }
                // ⚠ SAME CONTRACT AS queue_forked_request. fork_blocks trims dst.logical_seq
                // to the inherited span and sets dst.n_prompt = n_inherited. That is the P1-6
                // contract -- do not change fork_blocks. Without putting the full prompt back,
                // remaining_prompt is 0 and populate_batch_from asserts (measured 2026-08-14:
                // "64 of 64, 0 left to prefill" then abort). The log used to print the
                // clobbered length; print the saved one.
                const std::vector<llama_token> full_seq    = group.logical_seq;
                const uint32_t                 full_prompt = group.n_prompt;
                const uint32_t shared = kv_cache_manager->fork_blocks(src_view, group, best_n);
                if (shared > 0) {
                    group.logical_seq = full_seq;
                    group.n_prompt    = full_prompt;
                    group.n_past      = shared;
                    LLAMA_LOG_INFO("%s: request %d admitted SHARED: %u of %u prompt tokens inherited "
                                   "from %s %d (%zu blocks), %u left to prefill\n",
                                   __func__, group.request_id, shared, group.n_prompt,
                                   best_src->request_id < 0 ? "held prefix" : "live request",
                                   best_src->request_id, group.block_table.size(),
                                   group.n_prompt - shared);
                }
            }

            // cold request: nothing may be left parked under this id or it pins pool
            // blocks nobody is ever going to claim
            kv_cache_manager->discard_restored(group.request_id);
        } else {
            // already inherited (P1-6 fork child). do not scan, do not fork_blocks again.
            kv_cache_manager->discard_restored(group.request_id);
        }
    }

    auto group_ptr = std::make_unique<llama_sequence_group>(std::move(group));

    id_to_group[group_ptr->request_id] = group_ptr.get();

    set_waiting(std::move(group_ptr));
    return true;
}

bool llama_paged_scheduler_impl::queue_forked_request(llama_sequence_group group, int32_t parent_request_id) {
    last_fork_used_blocks_ = false;

    // Hybrid with rewindable RS (DSV4, Qwen3.5) or no RS (pure SWA, dense wrappers):
    // take fork_blocks. Hybrid with non-rewindable RS (SSM): refuse loud.
    // Do NOT degrade to queue_request -- that is a silent cold, or an APC share
    // that pretends the fork happened (measured 2026-08-15 DSV4 Flash 8k e2e).
    if (!can_fork()) {
        LLAMA_LOG_ERROR("%s: request %d: fork unsupported -- hybrid arch has recurrent "
                        "state that cannot rewind; refusing (not queueing as a normal request)\n",
                        __func__, group.request_id);
        return false;
    }

    llama_sequence_group * parent_ptr = find_parent_group(parent_request_id);
    if (parent_ptr == nullptr) {
        LLAMA_LOG_ERROR("%s: parent request %d not found; queueing as a normal request\n",
                        __func__, parent_request_id);
        return queue_request(std::move(group));
    }

    // Live groups clear block_table after the cache copies it. Held
    // prefixes keep theirs. Give fork_blocks a view with a real table.
    llama_sequence_group parent_view = *parent_ptr;
    if (parent_view.block_table.empty() && kv_cache_manager) {
        if (const auto * bl = kv_cache_manager->get_sequence_blocks(parent_ptr->request_id)) {
            parent_view.block_table = *bl;
        }
    }
    const llama_sequence_group & parent = parent_view;

    // fork at the COMMON PREFIX: the parent's logical_seq grows with its own generated
    // tokens, so requiring a full prefix match would reject every live fork
    size_t n_shared = 0;
    const size_t n_cmp = std::min(parent.logical_seq.size(), group.logical_seq.size());
    while (n_shared < n_cmp && parent.logical_seq[n_shared] == group.logical_seq[n_shared]) {
        n_shared++;
    }
    // chunked prefill: KV blocks are allocated for the WHOLE prompt at admission but only
    // written up to n_past. A mid-prefill parent's token LCP can exceed its actual KV
    // progress, and sharing those allocated-but-unwritten blocks hands the child garbage
    // (measured: plausible-but-wrong children, 2-3 logit shifts, CPU and CUDA alike).
    // Inherit only what the parent has actually computed.
    n_shared = std::min(n_shared, (size_t) parent.n_past);
    if (n_shared == 0) {
        LLAMA_LOG_WARN("%s: request %d shares no prefix with %d; queueing normally\n",
                       __func__, group.request_id, parent_request_id);
        return queue_request(std::move(group));
    }

    const std::vector<llama_token> full_seq = group.logical_seq;

    const uint32_t n_inherited = kv_cache_manager->fork_blocks(parent, group, (uint32_t) n_shared);
    last_fork_used_blocks_ = n_inherited > 0 || !group.block_table.empty();

    // logical_seq must stay the FULL prompt; fork_blocks trimmed it to the inherited span
    group.logical_seq = full_seq;
    group.n_prompt    = (uint32_t) full_seq.size();
    group.n_past      = n_inherited;

    LLAMA_LOG_INFO("%s: request %d forked from %d: common prefix %zu, %u tokens inherited by reference "
                   "(no re-prefill) of %zu prompt tokens\n",
                   __func__, group.request_id, parent_request_id, n_shared, n_inherited, full_seq.size());

    return queue_request(std::move(group));
}

void llama_paged_scheduler_impl::insert_sorted_by_arrival_time(llama_sequence_group_ptr    new_group_ptr,
                                                               llama_sequence_group_list & list) {
    GGML_ASSERT(new_group_ptr && "New group cannot be sorted because it's nullptr.");
    auto it =
        std::lower_bound(list.begin(), list.end(), new_group_ptr->t_arrival_time,
                         [](const llama_sequence_group_ptr & group, int64_t time) {
                             GGML_ASSERT(group && "group cannot be checked for arrival time because it's nullptr.");
                             return group->t_arrival_time < time;
                         });
    list.insert(it, std::move(new_group_ptr));
}

void llama_paged_scheduler_impl::set_running(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::RUNNING &&
                "Request is already running.");
    group_ptr->status = llama_sequence_group_status::RUNNING;
    insert_sorted_by_arrival_time(std::move(group_ptr), running);
}

void llama_paged_scheduler_impl::set_swapped(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::SWAPPED &&
                "Request is already swapped.");
    group_ptr->status = llama_sequence_group_status::SWAPPED;
    group_ptr->pending_draft.clear();   // ⚠ parked now, resumed against a different cache state
    insert_sorted_by_arrival_time(std::move(group_ptr), swapped);
}

void llama_paged_scheduler_impl::set_waiting(llama_sequence_group_ptr group_ptr, bool prepend) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::WAITING &&
                "Request is already waiting.");
    group_ptr->status = llama_sequence_group_status::WAITING;
    if (prepend) {
        // vLLM PREEMPTED + prepend: resume this victim before later arrivals
        // once a child frees blocks.
        waiting.push_front(std::move(group_ptr));
        return;
    }
    insert_sorted_by_arrival_time(std::move(group_ptr), waiting);
}

void llama_paged_scheduler_impl::finish(llama_sequence_group & group) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    GGML_ASSERT(group.status == llama_sequence_group_status::FINISHED && "Request was not marked as finished.");
    // idempotent: teardown runs eagerly at update-time and again from the running-list
    // sweep that removes the group from the list
    if (group.torn_down) {
        return;
    }
    group.torn_down = true;
    group.pending_draft.clear();   // ⚠ teardown runs from TWO places; leave nothing behind
    // We prioritize user CB, otherwise we log by default
    if (on_finish_cb) {
        // TODO perhaps just have the callback take sequence_group and user_data
        on_finish_cb(group.request_id, group.logical_seq.data(), (int32_t) group.logical_seq.size(),
                     on_finish_user_data);
    } else {
        LLAMA_LOG_DEBUG("%s: Request: %d generated %d tokens.\n", __func__, group.request_id, group.n_decoded);
    }
    // Keep the full-block prefix in held_prefixes (extra ref via
    // fork_blocks) so a child that arrives after this request is gone
    // still SHARED-admits. Then free_blocks drops THIS request's refs;
    // the park keeps the prefix until eviction or teardown.
    park_finished_prefix(group);
    rebind_session_after_finish(group);
    kv_cache_manager->free_blocks(group);
    group.status = llama_sequence_group_status::FINISHED;
    // erase only OUR mapping: a new request may already have reused this id (the server
    // reuses slot ids), and erasing its entry orphans the new request -- the exact
    // "positions are decreasing" storm the P2-8 queue-not-reject arm caught
    auto it = id_to_group.find(group.request_id);
    if (it != id_to_group.end() && it->second == &group) {
        id_to_group.erase(it);
    }
}


void llama_paged_scheduler_impl::requeue_mixed_overflow(llama_sequence_group * group) {
    if (!group || group->status == llama_sequence_group_status::FINISHED) {
        return;
    }
    // Overflow after unique-suffix swap: leftover still cannot admit.
    // Keep HTTP live. A sibling RELEASE (or a later swap) grows leftover.
    // Do not touch the named master prefix. Do not 500.
    LLAMA_LOG_INFO("%s: DS4P-QUEUE mixed remap leftover short; waiter stays live (request %d)\n",
                   __func__, group->request_id);
    if (group->status == llama_sequence_group_status::WAITING) {
        return;
    }
    for (auto it = running.begin(); it != running.end(); ++it) {
        if (it->get() != group) {
            continue;
        }
        llama_sequence_group_ptr ptr = std::move(*it);
        running.erase(it);
        // vLLM PREEMPTED: resume this waiter before later arrivals once leftover grows.
        set_waiting(std::move(ptr), /*prepend=*/true);
        return;
    }
}

void llama_paged_scheduler_impl::fail_mixed_remap_once(llama_sequence_group * group) {
    // 16c: overflow queues. A 500 here was the 16b hole (child05).
    requeue_mixed_overflow(group);
}

void llama_paged_scheduler_impl::park_finished_prefix(llama_sequence_group & group) {
    if (!kv_cache_manager) {
        return;
    }
    // group.block_table is a staging field -- a live prefilled sequence may
    // show empty there while sequence_blocks still holds the real ids.
    if (group.block_table.empty()) {
        if (const auto * bl = kv_cache_manager->get_sequence_blocks(group.request_id)) {
            group.block_table = *bl;
        }
    }
    const uint32_t n_full = block_size ? (group.n_past / block_size) * block_size : 0;
    const uint32_t n_keep = std::min(group.n_past, (uint32_t) group.logical_seq.size());
    const bool     named  = request_sessions.find(group.request_id) != request_sessions.end();

    auto prefix_already_held = [&](uint32_t n_cmp) {
        if (n_cmp == 0 || group.logical_seq.size() < n_cmp) {
            return false;
        }
        for (const auto & h : held_prefixes) {
            if (!h || h->logical_seq.size() < n_cmp) {
                continue;
            }
            bool match = true;
            for (uint32_t i = 0; i < n_cmp; ++i) {
                if (h->logical_seq[i] != group.logical_seq[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return true;
            }
        }
        return false;
    };

    // Name-only hold: 0 full blocks, but the session name stays resolvable.
    // Children inherit nothing and prefill the short prefix. That is OK.
    // Losing the NAME is not.
    auto park_name_only = [&]() {
        if (!named || n_keep == 0 || prefix_already_held(n_keep)) {
            return;
        }
        llama_sequence_group hold;
        hold.request_id = next_hold_id--;
        hold.status     = llama_sequence_group_status::FINISHED;
        hold.logical_seq.assign(group.logical_seq.begin(),
                                group.logical_seq.begin() + n_keep);
        hold.n_past   = n_keep;
        hold.n_prompt = n_keep;
        held_prefixes.push_back(std::make_unique<llama_sequence_group>(std::move(hold)));
        LLAMA_LOG_INFO("%s: parked name-only %u-token prefix from finished request %d "
                       "(0 full blocks) so session stays resolvable\n",
                       __func__, n_keep, group.request_id);
    };

    if (n_full == 0 || group.block_table.empty() || group.logical_seq.size() < n_full) {
        park_name_only();
        return;
    }

    // Already parked (a child finishing the same prefix, or a longer hold
    // that already covers these tokens). Do not take a second ref.
    if (prefix_already_held(n_full)) {
        return;
    }

    llama_sequence_group hold;
    hold.request_id = next_hold_id--;
    hold.status     = llama_sequence_group_status::FINISHED;
    const uint32_t inherited = kv_cache_manager->fork_blocks(group, hold, n_full);
    if (inherited == 0 || hold.block_table.empty()) {
        park_name_only();
        return;
    }
    // fork_blocks trims logical_seq to the inherited span -- that is what
    // we want. Do not put the hold in id_to_group: negative ids are not slots.
    held_prefixes.push_back(std::make_unique<llama_sequence_group>(std::move(hold)));
    LLAMA_LOG_INFO("%s: parked %u-token prefix from finished request %d "
                   "(%zu blocks) for later share\n",
                   __func__, inherited, group.request_id,
                   held_prefixes.back()->block_table.size());
}

bool llama_paged_scheduler_impl::is_named_session_id(int32_t request_id) const {
    if (request_sessions.find(request_id) != request_sessions.end()) {
        return true;
    }
    for (const auto & kv : sessions) {
        if (kv.second == request_id) {
            return true;
        }
    }
    return false;
}

size_t llama_paged_scheduler_impl::held_prefix_n_blocks(int32_t request_id) const {
    for (const auto & h : held_prefixes) {
        if (!h || h->request_id != request_id) {
            continue;
        }
        if (!h->block_table.empty()) {
            return h->block_table.size();
        }
        if (kv_cache_manager) {
            if (const auto * bl = kv_cache_manager->get_sequence_blocks(request_id)) {
                return bl->size();
            }
        }
        return 0;
    }
    return 0;
}

bool llama_paged_scheduler_impl::evict_held_prefix() {
    if (!kv_cache_manager) {
        return false;
    }
    // Unique suffix first: drop trailing ref_cnt==1 blocks. A hold whose
    // prefix is still shared (ref_cnt>1) keeps those blocks -- children
    // still need them. A hold with no children has ref_cnt==1 on every
    // block, so this frees the whole unused prefix and returns capacity.
    //
    // Named session holds are not DESTROYED. Shortening one to admit a
    // child is how a parked master went 7 blocks / 112 tokens -> 4 and
    // the next fork inherited 64 of 112 (257458bdd). Product: swap the
    // unique suffix to CPU so GPU frees and the session stays whole.
    // Shared prefix GPU ids are not rewritten (ref_cnt>1 is never swapped).
    for (auto it = held_prefixes.begin(); it != held_prefixes.end(); ++it) {
        llama_sequence_group * h = it->get();
        if (!h) {
            continue;
        }
        if (is_named_session_id(h->request_id)) {
            if (h->block_table.empty()) {
                if (const auto * bl = kv_cache_manager->get_sequence_blocks(h->request_id)) {
                    h->block_table = *bl;
                }
            }
            const uint32_t swapped = kv_cache_manager->swap_out_unref_suffix(*h);
            if (swapped == 0) {
                LLAMA_LOG_DEBUG("%s: named session hold %d has no unref GPU suffix to swap.\n",
                                __func__, h->request_id);
                continue;
            }
            LLAMA_LOG_INFO("%s: swapped %u unique-suffix block(s) of named hold %d to CPU "
                           "(%zu blocks remain, n_past=%u, session not shortened)\n",
                           __func__, swapped, h->request_id, h->block_table.size(), h->n_past);
            return true;
        }
        if (h->block_table.empty()) {
            if (const auto * bl = kv_cache_manager->get_sequence_blocks(h->request_id)) {
                h->block_table = *bl;
            }
        }
        if (h->block_table.empty()) {
            // Name-only hold: no blocks to reclaim. close_session drops it.
            // Evicting it frees nothing and would make the session unresolvable.
            continue;
        }
        const uint32_t dropped = kv_cache_manager->release_unref_suffix(*h);
        if (dropped == 0) {
            continue;
        }
        h->n_past = (uint32_t) h->block_table.size() * block_size;
        if (h->logical_seq.size() > h->n_past) {
            h->logical_seq.resize(h->n_past);
        }
        h->n_prompt = h->n_past;
        LLAMA_LOG_DEBUG("%s: evicted %u unique-suffix block(s) from held prefix %d "
                        "(%zu blocks remain)\n",
                        __func__, dropped, h->request_id, h->block_table.size());
        if (h->block_table.empty()) {
            held_prefixes.erase(it);
        }
        return true;
    }
    return false;
}

bool llama_paged_scheduler_impl::abort_request(int32_t request_id) {
    // Resolve through id_to_group so a REUSED id (the server reuses slot ids) can never abort a
    // newer request by accident: the map always points at the current owner of the id, and finish()
    // already guards its erase the same way.
    auto found = id_to_group.find(request_id);
    if (found == id_to_group.end()) {
        return false;   // already finished, or never queued -- calling twice is safe
    }
    llama_sequence_group * target = found->second;

    for (llama_sequence_group_list * list : { &running, &swapped, &waiting }) {
        for (auto it2 = list->begin(); it2 != list->end(); ++it2) {
            if (it2->get() != target) {
                continue;
            }
            LLAMA_LOG_WARN("%s: request %d aborted by the server -- freeing its blocks and removing "
                           "it from the queue.\n", __func__, request_id);
            target->status = llama_sequence_group_status::FINISHED;
            // finish() frees the blocks, fires on_finish, erases the id mapping, and is idempotent.
            // NOT pushed to terminated_ids: that channel is how the SCHEDULER tells the SERVER about
            // capacity kills; here the server is the caller and has already failed the task -- using
            // the channel would error the slot a second time.
            finish(*target);
            list->erase(it2);
            return true;
        }
    }

    // Mapped but in no queue: the group is inside the CURRENT batch's candidate walk. The server's
    // single-threaded loop calls abort only after prepare_batch has returned, so this indicates a
    // caller from somewhere new -- refuse rather than mutate a list mid-walk, and say so.
    LLAMA_LOG_ERROR("%s: request %d is mapped but in no queue (mid-batch?) -- NOT aborted. If a new "
                    "call site triggered this, it is running concurrently with prepare_batch.\n",
                    __func__, request_id);
    return false;
}

// Try to swap a running sequence out to CPU.
// if the CPU pool is full, fall back to recomputation by resetting the sequence's decode state
// and sending it back to the waiting queue.
//
// Takes ownership of group_ptr.
// On return, the sequence is either in the swapped list (CPU pool had room)
// or the waiting list (recomputed).
void llama_paged_scheduler_impl::swap_out_or_recompute(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && "group_ptr is nullptr");
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr");

    const int32_t rid = group_ptr->request_id;

    // ★ UNSATISFIABLE-REQUEST GUARD. Measured: -ngpub 8 --kv-block-size 16 gives 128 tokens of GPU
    // capacity for a ~280-token job, and the scheduler LIVELOCKED -- 2,085,089 CPU->GPU swaps and
    // 2,085,090 GPU->CPU, a tight alternation with no forward progress, ending in the residency-set
    // teardown assert. The loop lives in process_swapped_list: swap_in succeeds, the growth
    // allocation fails, the group is swapped back out to keep its table CPU-consistent, break for
    // FCFS, repeat forever.
    //
    // ★ EVERY DECISION IN THAT LOOP IS INDIVIDUALLY CORRECT AND DOCUMENTED. The swap_out prevents a
    // GPU-id/CPU-id underflow (there is a comment recording the measured OOB at pos 192); the break
    // preserves FCFS; the growth check exists because a swap-returned table is one block short.
    // Three right local choices compose into an infinite loop because NOTHING ASKS WHETHER THE ROUND
    // TRIP ACHIEVED ANYTHING. A missing global invariant, not a bad line.
    //
    // Neither swapping nor recomputing can help once a sequence outgrows the ENTIRE pool: with one
    // request it is evicting itself to make room for itself, and after a recompute it re-grows into
    // the same wall. Retrying forever is the wrong answer to an impossible request -- say so, with
    // the numbers.
    const uint64_t pool_tokens = (uint64_t) kv_cache_manager->get_usable_gpu_blocks() * block_size;
    const uint32_t unref       = count_unref_blocks(*group_ptr);
    const uint32_t table_n     = group_ptr->block_table.empty()
        ? (blocks_of(*group_ptr) ? (uint32_t) blocks_of(*group_ptr)->size() : 0)
        : (uint32_t) group_ptr->block_table.size();
    const uint32_t shared_n    = table_n > unref ? table_n - unref : 0;
    const uint32_t need_blocks = (group_ptr->n_past + 1 + block_size - 1) / block_size;
    // Unsatisfiable only when THIS request's own table (shared prefix + its
    // unique tail) exceeds the entire pool. A child whose unique tail would
    // fit after a sibling leaves must not be take_terminated -- that was the
    // 500. Shared prefix blocks do not count against the unique demand.
    const bool alone_too_big = need_blocks > kv_cache_manager->get_usable_gpu_blocks();
    // Lone request (no shared prefix) that outgrew the pool: the original
    // livelock. A child with a shared prefix is NOT this -- its unique tail
    // would fit after a sibling leaves.
    if (alone_too_big && (unref == 0 || unref == table_n)) {
        // No sibling tail to free, and the request itself is larger than the pool.
        LLAMA_LOG_ERROR("%s: request %d needs %llu tokens of KV but the GPU block pool holds at most "
                        "%llu (%u blocks x %u). No eviction or recompute can create capacity that "
                        "does not exist -- terminating the request instead of retrying forever.\n",
                        __func__, rid, (unsigned long long) (group_ptr->n_past + 1),
                        (unsigned long long) pool_tokens,
                        kv_cache_manager->get_usable_gpu_blocks(), block_size);
        terminated_ids.push_back(rid);
        group_ptr->status = llama_sequence_group_status::FINISHED;
        finish(*group_ptr);
        return;
    }
    (void) shared_n;

    // A child with a shared prefix: drop only the unref tail, keep the prefix,
    // prepend to waiting (vLLM PREEMPTED). Swapping the whole table would
    // rewrite shared GPU ids to CPU ids while the master still holds GPU ids.
    if (unref > 0 && table_n > unref) {
        const uint32_t dropped = kv_cache_manager->release_unref_suffix(*group_ptr);
        group_ptr->n_past   = (uint32_t) group_ptr->block_table.size() * block_size;
        group_ptr->pending_draft.clear();
        // Replay tokens after the prefix. If the unique tail was decode-only
        // growth (n_prompt == inherited prefix), remaining_prompt would be 0
        // and populate_batch_from asserts -- keep it a decode group.
        group_ptr->n_decoded = 0;
        group_ptr->n_prompt  = (uint32_t) group_ptr->logical_seq.size();
        if (group_ptr->n_prompt <= group_ptr->n_past) {
            group_ptr->n_decoded = 1;
        }
        LLAMA_LOG_DEBUG("%s: (preempt_tail) request_id=%d dropped %u unref blocks, "
                        "prefix kept, prepended to waiting.\n",
                        __func__, rid, dropped);
        set_waiting(std::move(group_ptr), /*prepend=*/true);
        return;
    }

    const bool swap_ok = kv_cache_manager->swap_out(*group_ptr);
    if (swap_ok) {
        LLAMA_LOG_DEBUG("%s: (swapped_out) request_id=%d was swapped out to make room.\n", __func__, rid);
        set_swapped(std::move(group_ptr));
        return;
    }

    // There was not enough CPU memory to swap the request (recomputation).
    // KEEP logical_seq whole: it holds prompt + already-GENERATED tokens, and chunked
    // prefill replays the full sequence from n_past=0 -- truncating to n_prompt silently
    // discarded the generated tail and restarted the request from its prompt.
    kv_cache_manager->free_blocks(*group_ptr);
    kv_cache_manager->seq_rm(rid, -1, -1);
    group_ptr->n_past    = 0;
    group_ptr->n_decoded = 0;
    group_ptr->n_prompt  = (uint32_t) group_ptr->logical_seq.size();
    group_ptr->pending_draft.clear();   // ⚠ a draft staged against the OLD cache state must not
                                        // survive a replay from n_past=0

    LLAMA_LOG_DEBUG("%s: (recomputation) request_id=%d was sent for recomputation.\n", __func__, rid);
    set_waiting(std::move(group_ptr), /*prepend=*/true);
}

const llama_block_ids * llama_paged_scheduler_impl::blocks_of(const llama_sequence_group & group) const {
    if (!group.block_table.empty()) {
        return &group.block_table;
    }
    return kv_cache_manager ? kv_cache_manager->get_sequence_blocks(group.request_id) : nullptr;
}

uint32_t llama_paged_scheduler_impl::count_unref_blocks(const llama_sequence_group & group) const {
    const llama_block_ids * blocks = blocks_of(group);
    if (!blocks || !kv_cache_manager) {
        return 0;
    }
    uint32_t n = 0;
    for (uint32_t id : *blocks) {
        if (kv_cache_manager->get_block_ref_count(id) <= 1) {
            n++;
        }
    }
    return n;
}

llama_sequence_group * llama_paged_scheduler_impl::find_master_prefix_group() const {
    if (!kv_cache_manager) {
        return nullptr;
    }
    uint32_t max_ref = 1;
    for (const auto & g : running) {
        const llama_block_ids * blocks = g ? blocks_of(*g) : nullptr;
        if (!blocks) {
            continue;
        }
        for (uint32_t id : *blocks) {
            const uint32_t rc = kv_cache_manager->get_block_ref_count(id);
            if (rc > max_ref) {
                max_ref = rc;
            }
        }
    }
    if (max_ref <= 1) {
        return nullptr;  // no shared prefix in the running set
    }
    llama_sequence_group * master = nullptr;
    for (const auto & g : running) {
        if (!g) {
            continue;
        }
        const llama_block_ids * blocks = blocks_of(*g);
        if (!blocks) {
            continue;
        }
        bool holds_max = false;
        for (uint32_t id : *blocks) {
            if (kv_cache_manager->get_block_ref_count(id) == max_ref) {
                holds_max = true;
                break;
            }
        }
        if (!holds_max) {
            continue;
        }
        if (!master || g->t_arrival_time < master->t_arrival_time) {
            master = g.get();
        }
    }
    return master;
}

bool llama_paged_scheduler_impl::evict() {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    LLAMA_LOG_DEBUG("%s: Eviction requested...\n", __func__);
    if (running.empty()) {
        // Pool may be full of parked prefixes with no live runner.
        return evict_held_prefix();
    }

    llama_sequence_group * master = find_master_prefix_group();

    llama_sequence_group_list::iterator best = running.end();
    uint32_t best_unref = 0;
    int64_t  best_arrival = -1;
    for (auto it = running.begin(); it != running.end(); ++it) {
        llama_sequence_group * g = it->get();
        if (!g || g == master) {
            continue;  // NEVER the highest-ref_cnt master prefix
        }
        if (is_named_session_id(g->request_id)) {
            continue;  // named live master stays resident
        }
        if (kv_cache_manager->count_cpu_unique(*g) > 0) {
            continue;  // mixed stored table: do not whole-table swap_out
        }
        const uint32_t unref = count_unref_blocks(*g);
        if (unref == 0) {
            continue;  // no unique tail to free
        }
        // Prefer the fattest unref tail; FCFS tie-break evicts the newest.
        if (best == running.end() || unref > best_unref ||
            (unref == best_unref && g->t_arrival_time > best_arrival)) {
            best         = it;
            best_unref   = unref;
            best_arrival = g->t_arrival_time;
        }
    }
    if (best == running.end()) {
        LLAMA_LOG_DEBUG("%s: no unref-tail victim (master prefix stays).\n", __func__);
        // Non-session parked unique suffix next. Named session holds
        // swap their unref GPU suffix inside evict_held_prefix.
        return evict_held_prefix();
    }

    llama_sequence_group_ptr victim = std::move(*best);
    running.erase(best);
    GGML_ASSERT(victim && "request selected for eviction is nullptr.");
    swap_out_or_recompute(std::move(victim));
    return true;
}

void llama_paged_scheduler_impl::process_running_list(llama_sequence_group_raw_list & candidates) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");

    llama_sequence_group_list::iterator it = running.begin();
    while (it != running.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "group is nullptr.");

        if (group->status == llama_sequence_group_status::FINISHED) {
            finish(*group);
            it = running.erase(it);
            continue;
        }

        // Dynamically allocate more blocks to decode the request
        uint32_t current_capacity  = group->block_table.size() * block_size;
        // ★ STEP A. A decoding group needs room for its whole row set, not one token: the last
        // accepted token plus every drafted one. Rejected drafts leave KV behind at positions past
        // the new n_past, which the next step simply overwrites -- DS4P_SLOT_COVER proved the write
        // slot is a pure function of position -- so nothing is freed, but the blocks must EXIST or
        // the write walks off the end of the block table.
        const uint32_t n_rows_wanted = 1 + (uint32_t) group->pending_draft.size();
        uint32_t required_capacity = group->n_past + n_rows_wanted;
        LLAMA_LOG_DEBUG(
            "%s: (running) request_id=%d: current_capacity (tokens)=%d toks, required capacity (tokens) = %d toks\n",
            __func__, group->request_id, current_capacity, required_capacity);
        if (required_capacity >= current_capacity) {
            LLAMA_LOG_DEBUG("%s: (running_pending) request_id=%d: requires a new block to decode.\n", __func__,
                            group->request_id);
            // blocks needed to cover n_past .. n_past + n_rows_wanted - 1
            const uint32_t have   = (uint32_t) group->block_table.size() * block_size;
            const uint32_t needed = required_capacity > have
                                  ? (required_capacity - have + block_size - 1) / block_size : 1;
            bool success = kv_cache_manager->allocate(needed, *group);  // decode phase
            if (!success) {
                const int32_t cur_id = group->request_id;
                // Swap/recompute an unref tail (never the shared master prefix).
                evict();
                // evict() may have removed US or someone else -- re-find current
                it = running.end();
                for (auto jt = running.begin(); jt != running.end(); ++jt) {
                    if (jt->get() && jt->get()->request_id == cur_id) {
                        it = jt;
                        break;
                    }
                }
                if (it == running.end()) {
                    continue;  // we were the victim
                }
                group = it->get();
                success = kv_cache_manager->allocate(needed, *group);

                if (!success) {
                    // Still no room. Self-preempt only if we have an unref tail
                    // (a child). The master prefix stays running and retries
                    // next tick; it is not a candidate this step (no block for
                    // the next token -- batching it would OOB).
                    if (count_unref_blocks(*group) > 0 && group != find_master_prefix_group()) {
                        llama_sequence_group_ptr self = std::move(*it);
                        it                            = running.erase(it);
                        swap_out_or_recompute(std::move(self));
                        continue;
                    }
                    LLAMA_LOG_DEBUG("%s: request %d stays running without a new block "
                                    "(waiting for a child to free an unref tail).\n",
                                    __func__, group->request_id);
                    ++it;
                    continue;
                }
            }
            LLAMA_LOG_DEBUG("%s: (running_restored) request_id=%d: found a new block to continue decoding.\n", __func__,
                            group->request_id);
        }

        // A request is a candidate if we there is still room for generation without adding blocks
        // or if there was enough GPU memory to allocate another physical block.
        candidates.push_back(group);
        ++it;
    }
}

void llama_paged_scheduler_impl::process_swapped_list(llama_sequence_group_raw_list & candidates) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    llama_sequence_group_list::iterator it = swapped.begin();
    while (it != swapped.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "the group to swap is nullptr.");
        const bool success = kv_cache_manager->swap_in(*group);
        if (!success) {
            // We respect FCFS, so we stop here to prevent a younger swapped request from jumping ahead.
            break;
        }
        // the victim was usually evicted BECAUSE its growth allocation failed: after the
        // round-trip its table still lacks the block for the next position, and promoting
        // it unchecked batches pos n_past against a too-short table (measured: peers at 13
        // blocks, swap-returned group at 12, OOB at pos 192). Same growth rule as the
        // running list -- no capacity, no promotion.
        if (group->n_past + 1 > group->block_table.size() * block_size) {
            // ★ THE LIVELOCK EXITS HERE, and it must be caught on THIS path specifically. My first
            // guard went into swap_out_or_recompute -- which this code NEVER REACHES when the CPU
            // pool has room, because the swap_out below succeeds and we break. The fix was correct
            // and unreachable, and the re-run proved it: still 574,155 swaps, still hanging. A fix
            // placed on the path you assumed rather than the path measured is not a fix.
            //
            // When the sequence has outgrown the ENTIRE GPU pool, this round trip can never make
            // progress: swap_in, fail to grow, swap back out, break, repeat. Route it to
            // swap_out_or_recompute, whose capacity guard terminates it with the numbers.
            const uint64_t pool_tokens = (uint64_t) kv_cache_manager->get_usable_gpu_blocks() * block_size;
            // ⚠ INSTRUMENT, DO NOT GUESS. Two guards written from my model of the trigger condition
            // failed to fire while the livelock continued (574k then 757k swaps). Rather than write
            // a third, print the actual values at the loop point -- the same move that turned the
            // quantised-KV hunt from four hypotheses into a lookup.
            {
                static int n = 0;
                if (++n <= 20) {
                    LLAMA_LOG_INFO("%s: DS4P-THRASH n=%d rid=%d n_past=%u table_blocks=%zu "
                                   "block_size=%u pool_gpu_blocks=%u pool_tokens=%llu\n",
                                   __func__, n, group->request_id, group->n_past,
                                   group->block_table.size(), block_size,
                                   kv_cache_manager->get_num_gpu_blocks(),
                                   (unsigned long long) pool_tokens);
                }
            }
            if ((uint64_t) group->n_past + 1 > pool_tokens) {
                llama_sequence_group_ptr doomed = std::move(*it);
                it = swapped.erase(it);
                swap_out_or_recompute(std::move(doomed));
                continue;
            }

            if (!kv_cache_manager->allocate(1, *group)) {
                // the group is ALREADY swapped in (its table now holds GPU ids). Leaving it
                // in `swapped` would re-enter swap_in next tick and do_block_copy would
                // treat GPU ids as CPU ids: id - num_gpu_blocks UNDERFLOWS into a wild
                // offset ("tensor read out of bounds"). Restore consistency by swapping it
                // back out; if even that fails, recompute it.
                if (!kv_cache_manager->swap_out(*group)) {
                    llama_sequence_group_ptr back = std::move(*it);
                    it = swapped.erase(it);
                    swap_out_or_recompute(std::move(back));
                    continue;
                }
                break;  // stays swapped, table CPU-consistent; FCFS holds
            }
        }
        candidates.push_back(group);
        llama_sequence_group_ptr group_ptr = std::move(*it);
        LLAMA_LOG_DEBUG("%s: (swapped_in) request_id=%d back in for processing.\n", __func__, group_ptr->request_id);
        set_running(std::move(group_ptr));
        it = swapped.erase(it);
    }
}

void llama_paged_scheduler_impl::process_waiting_list(llama_sequence_group_raw_list & candidates,
                                                      int32_t                         remaining_token_budget) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    llama_sequence_group_list::iterator it    = waiting.begin();
    size_t                              count = 0;
    while (it != waiting.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "the waiting group is nullptr.");

        const int32_t tokens_needed = group->n_prompt + 1;
        // chunked prefill: a prompt longer than the remaining batch budget is still
        // admitted -- its KV blocks are allocated in full here, but the COMPUTE is
        // streamed in batch-sized chunks by populate_batch_from. (Prompts > n_batch
        // previously never admitted and waited forever; caught by the Q4 fork-cost
        // gate 2026-08-04.) Budget floor of 1 keeps candidate count <= n_batch so the
        // chunker can always give every candidate at least one token.
        if (remaining_token_budget < 1) {
            break;
        }
        if (n_seq_max_batch > 0 && candidates.size() >= (size_t) n_seq_max_batch) {
            break;
        }

        ++count;
        // When prefilling, we want to always guarantee at least one decode to avoid thrashing.
        // allocate() already counts n_prompt + n_decoded internally, so the argument is the
        // DELTA only: passing tokens_needed (= n_prompt+1) double-counted the prompt and
        // demanded ~2x the blocks -- admission serialized every fat request (running=1
        // always in the starved walls) and eviction/recompute became unreachable.
        //
        // Leftover GPU < this waiter's unique: swap parked unref suffixes
        // until leftover can admit, or nothing left to swap. If still short,
        // leave this waiter queued and try a sibling -- do NOT fail_mixed / 500.
        // Named master prefix is never rewritten (ref_cnt>1 is not swapped).
        {
            const uint32_t curr  = (uint32_t) group->block_table.size();
            const uint32_t total = group->n_prompt + group->n_decoded + 1;
            const uint32_t need  = block_size
                ? (uint32_t) std::ceil((float) total / (float) block_size) - curr
                : 0;
            const uint32_t cpu_u = kv_cache_manager->count_cpu_unique(*group);
            auto still_short = [&]() {
                const uint32_t scratch = kv_cache_manager->n_scratch_gpu_blocks();
                return (need > scratch) || (cpu_u > scratch);
            };
            while (still_short()) {
                if (!evict_held_prefix()) {
                    break;
                }
            }
            if (still_short()) {
                LLAMA_LOG_INFO("%s: DS4P-QUEUE request %d unique=%u cpu_u=%u scratch=%u; "
                               "sibling may run (no fail_mixed)\n",
                               __func__, group->request_id, need, cpu_u,
                               kv_cache_manager->n_scratch_gpu_blocks());
                ++it;
                continue;
            }
        }
        bool success = kv_cache_manager->allocate(1, *group);
        if (!success) {
            // vLLM PREEMPT: free an unref tail so this waiter can start. If the
            // only occupant is the shared master prefix, leave the waiter in
            // `waiting` -- do NOT 500 it. FCFS: do not skip to a younger waiter.
            evict();
            success = kv_cache_manager->allocate(1, *group);
        }
        if (!success) {
            // evict() freed nothing usable. allocate() now admits a
            // child with a GPU prefix by checking out CPU unique
            // (mixed stored table; remap onto scratch before the
            // kernel). Whole-table swap_out is still forbidden -- it
            // would rewrite master GPU ids. If CPU unique also cannot
            // fit, stay queued -- never 500.
            break;
        }
        candidates.push_back(group);
        remaining_token_budget -= std::min(tokens_needed, remaining_token_budget);
        llama_sequence_group_ptr group_ptr = std::move(*it);
        LLAMA_LOG_DEBUG("%s: (start) request_id=%d sent for processing.\n", __func__, group_ptr->request_id);
        set_running(std::move(group_ptr));
        it = waiting.erase(it);
    }
    if (count > 0) {
        LLAMA_LOG_DEBUG("%s: Started %ld waiting requests\n", __func__, count);
    }
}

int32_t llama_paged_scheduler_impl::calculate_global_slot_index(int32_t                 token_pos,
                                                                std::vector<uint32_t> & block_table) {
    GGML_ASSERT(block_size && "block_size needs to be greater than 0");
    const int32_t block_table_id = token_pos / block_size;
    const int32_t offset         = token_pos % block_size;

    const size_t block_table_size = block_table.size();
    if ((size_t) block_table_id >= block_table_size) {
        LLAMA_LOG_ERROR("%s: block_table_id=%d is OOB for pos=%d. Block table size=%ld.\n", __func__, block_table_id,
                        token_pos, block_table_size);
        LLAMA_LOG_ERROR("%s: block_table_contents: [ ", __func__);
        for (size_t id = 0; id < block_table_size; ++id) {
            LLAMA_LOG_ERROR("%d ", block_table[id]);
            if (id == block_table_size - 1) {
                LLAMA_LOG_ERROR("]\n");
            }
        }
        GGML_ASSERT(false && "block_table_id OOB");
    }
    const int32_t block_id = block_table.at(block_table_id);

    return (block_id * block_size) + offset;
}

void llama_paged_scheduler_impl::clear_batch(llama_batch & batch) {
    LLAMA_LOG_DEBUG("%s: clearing batch.", __func__);
    // Copy scratch -> CPU unique so the stored table stays mixed.
    // Prefix GPU ids never change. Safe no-op if nothing was remapped.
    kv_cache_manager->finish_mixed_decode();
    // Invalidate last scheduled batch info before freeing the arrays
    // (MUST be called before the delete[]).
    kv_cache_manager->set_paged_batch_info(nullptr);

    delete[] curr_info.write_slots;
    delete[] curr_info.block_table;
    delete[] curr_info.context_lens;
    delete[] curr_info.batch_offsets;
    delete[] curr_info.batch_lens;
    delete[] curr_info.prefill_pending;
    delete[] curr_info.seq_ids;
    curr_info = {};  // reset to defaults

    if (batch.n_tokens == 0) {
        return;
    }

    llama_batch_free(batch);
    batch.n_tokens = 0;
}

void llama_paged_scheduler_impl::populate_batch_from(llama_sequence_group_raw_list & candidates,
                                                     llama_batch &                         batch) {
    if (candidates.empty()) {
        LLAMA_LOG_DEBUG("%s: No candidates for this step.\n", __func__);
        batch.n_tokens = 0;
        return;
    }
    // a candidate collected early in the sweep can be EVICTED by a later group's growth
    // allocation (evict() takes an unref tail, never the master prefix): its table then holds CPU block ids and
    // batching it aborts on the id check ("block_table_id OOB", swap wall 2026-08-04).
    // Only groups still RUNNING may enter the batch.
    llama_sequence_group_raw_list live;
    live.reserve(candidates.size());
    for (auto * g : candidates) {
        if (g->status == llama_sequence_group_status::RUNNING) {
            live.push_back(g);
        } else {
            LLAMA_LOG_DEBUG("%s: request %d evicted mid-sweep (status %d), dropped from batch\n",
                            __func__, g->request_id, (int) g->status);
        }
    }
    candidates.swap(live);
    if (candidates.empty()) {
        LLAMA_LOG_DEBUG("%s: all candidates evicted mid-sweep.\n", __func__);
        batch.n_tokens = 0;
        return;
    }

    // Mixed-table: only as many CPU-unique blocks as currently-free GPU
    // (watermark leftovers + already-swapped unique). Not a reserved slice.
    // Others stay RUNNING and retry next step. Do not whole-table swap.
    {
        uint32_t max_need = 0;
        for (auto * g : candidates) {
            const uint32_t need = kv_cache_manager->count_cpu_unique(*g);
            if (need > max_need) {
                max_need = need;
            }
        }
        while (max_need > kv_cache_manager->n_scratch_gpu_blocks()) {
            if (!evict_held_prefix()) {
                break;
            }
        }
        llama_sequence_group_raw_list fitted;
        fitted.reserve(candidates.size());
        uint32_t scratch_acc = 0;
        const uint32_t scratch_cap = kv_cache_manager->n_scratch_gpu_blocks();
        for (auto * g : candidates) {
            const uint32_t need = kv_cache_manager->count_cpu_unique(*g);
            if (need > 0 && need > scratch_cap) {
                requeue_mixed_overflow(g);
                continue;
            }
            if (need > 0 && scratch_acc + need > scratch_cap) {
                LLAMA_LOG_DEBUG("%s: request %d deferred (mixed unique %u, scratch left %u)\n",
                                __func__, g->request_id, need,
                                scratch_cap > scratch_acc ? scratch_cap - scratch_acc : 0);
                continue;
            }
            scratch_acc += need;
            fitted.push_back(g);
        }
        candidates.swap(fitted);
        if (candidates.empty()) {
            batch.n_tokens = 0;
            return;
        }
    }

    int32_t total_tokens = 0;
    int32_t batch_size   = candidates.size();
    int32_t max_blocks   = 0;

    LLAMA_LOG_DEBUG("%s: Creating batch from candidates (%d requests). n_batch=%d\n", __func__, batch_size, n_batch);

    // Chunked prefill: compute each candidate's token share for THIS step. Decode groups
    // take 1; prefill groups take their remaining prompt clamped so that every candidate
    // after them still gets at least one token (admission guarantees batch_size <= n_batch).
    // The fill loop below MUST consume exactly these shares.
    std::vector<int32_t> chunk_tokens(batch_size);
    {
        // P1-7a: a live decode shares its batch (= its GPU graph) with any prefill chunk,
        // so each decode step costs the CHUNK's compute -- but SMALLER quanta add more
        // steps, and each shape-changing step pays a fixed graph-rebuild cost. Measured
        // (5K prefill beside a live decode, box GPU): quantum 512 = +84.7% live tpot,
        // quantum 64 = +126.8% AND slower prefill -- a U-curve dominated by per-step
        // fixed cost. Default keeps full-budget chunks; DS4P_PREFILL_QUANTUM exposes the
        // knob for measurement. The real <10% fix is stream overlap or shape-stable
        // chunking for graph reuse (parked with data in the P1-7a witness).
        bool has_live_decode = false;
        for (int32_t i = 0; i < batch_size; ++i) {
            has_live_decode |= candidates[i]->n_decoded > 0;
        }
        int32_t prefill_quantum = (int32_t) n_batch;
        if (has_live_decode) {
            if (const char * s = getenv("DS4P_PREFILL_QUANTUM")) {
                const int q = atoi(s);
                if (q > 0) {
                    prefill_quantum = q;
                }
            }
        }

        int32_t budget = (int32_t) n_batch;
        for (int32_t i = 0; i < batch_size; ++i) {
            llama_sequence_group * group = candidates[i];
            GGML_ASSERT(group && "candidate request is nullptr.");
            const int32_t reserve_after = batch_size - 1 - i;  // 1 token each for the rest
            if (group->n_decoded > 0) {
                // ★ STEP B. A decoding group normally contributes ONE row. With a staged draft it
                // contributes 1 + n_draft: the last accepted token, then each drafted token, so the
                // target can verify all of them in a single forward pass.
                //
                // ⚠ Clamped to the remaining budget rather than assumed to fit. set_draft() already
                // rejects a draft larger than n_batch, but that check cannot see how much of the
                // budget the OTHER candidates in this step have taken. Silently over-committing here
                // would starve a sibling sequence, and the chunker's own assert would then fire on a
                // candidate that did nothing wrong.
                const int32_t want = 1 + (int32_t) group->pending_draft.size();
                chunk_tokens[i] = std::min(want, std::max(1, budget - reserve_after));
            } else {
                const int32_t remaining_prompt = (int32_t) group->n_prompt - (int32_t) group->n_past;
                GGML_ASSERT(remaining_prompt > 0 && "prefill candidate with no prompt remainder");
                chunk_tokens[i] = std::min(std::min(remaining_prompt, budget - reserve_after), prefill_quantum);
            }
            GGML_ASSERT(chunk_tokens[i] >= 1 && "chunker starved a candidate");
            budget -= chunk_tokens[i];
            total_tokens += chunk_tokens[i];
        }
    }

    for (const auto & group : candidates) {
        max_blocks = std::max(max_blocks, (int32_t) group->block_table.size());
    }

    GGML_ASSERT(total_tokens <= (int32_t) n_batch && "total_tokens exceeds n_batch — token budget logic is broken");

    // Initialize the batch (assumed it was cleared before)
    batch = llama_batch_init(total_tokens, 0, 1);
    GGML_ASSERT(batch.token != nullptr && "llama_batch_init failed to allocate tokens.");

    batch.n_tokens = total_tokens;

    curr_info.n_seq            = batch_size;
    curr_info.n_tokens         = total_tokens;
    curr_info.n_blocks_per_seq = max_blocks;

    curr_info.write_slots     = new int32_t[total_tokens];
    curr_info.block_table     = new int32_t[batch_size * max_blocks];
    curr_info.context_lens    = new int32_t[batch_size];
    curr_info.batch_offsets   = new int32_t[batch_size];
    curr_info.batch_lens      = new int32_t[batch_size];
    curr_info.prefill_pending = new int32_t[batch_size];
    curr_info.seq_ids         = new int32_t[batch_size];
    LLAMA_LOG_DEBUG("%s: created llama_batch: n_seq=%d, n_tokens=%d, n_blocks_per_seq=%d\n", __func__, curr_info.n_seq,
                    batch.n_tokens, curr_info.n_blocks_per_seq);

    int32_t token_offset = 0;
    for (int seq_id = 0; seq_id < batch_size; ++seq_id) {
        llama_sequence_group * group = candidates[seq_id];
        GGML_ASSERT(group && "Make sure the candidates are not nullptr.");

        const bool    is_prefill = group->n_decoded == 0;
        // P1-6: a forked group arrives with n_past > 0 (prefix inherited by reference), so
        // prefill must feed only the REMAINDER. Without this the child re-reads from
        // logical_seq[0] while writing at n_past.. -- wrong tokens at wrong positions.
        // Chunked prefill: the chunker above may have clamped the remainder to the batch
        // budget; mid-prompt chunks emit no logits and must not be sampled.
        const int32_t n_prefill_done = is_prefill ? (int32_t) group->n_past : 0;
        const int32_t new_tokens     = chunk_tokens[seq_id];
        const bool    mid_prefill    = is_prefill && (n_prefill_done + new_tokens < (int32_t) group->n_prompt);

        // ★ PREFILL PROGRESS. The static path prints `prompt processing, n_tokens = N, progress = ...`
        // every chunk; the paged path printed ONE allocation line and then nothing until the request
        // completed. On a 17-minute 225k prefill a working run and a hung one are the same picture, and
        // this lane has already killed TWO healthy 40-minute runs over exactly that ambiguity.
        //
        // ⚠ RATE-LIMITED TO 5% CROSSINGS, not per chunk. At -ub 128 a 1M prompt is ~8,000 chunks, and
        // this lane has one 15 MB/s log-storm on file (`Scheduler status` at INFO, 898 MB/min) that
        // landed on the PAGED arm only -- which would have shown paging losing on speed for a reason
        // that was not paging. Stateless bucket compare, so no per-group field to keep in sync.
        //
        // ⚠ The final chunk always prints, so a run always ends on 1.00 rather than on whatever bucket
        // it happened to cross last. A progress line that stops at 0.95 reads as a stall.
        //
        // ⚠ Skipped entirely when the prompt fits in one chunk: nothing to watch, and it would fire on
        // every single-token decode step. Note this loop runs only for real scheduled requests, so it
        // cannot report progress for llama-server's ~21 startup graph builds -- the trap that made an
        // earlier gate score 630 "fallbacks" for work that had not happened yet.
        // ⚠⚠ A PERCENTAGE RATE-LIMIT ALONE IS NOT ENOUGH, AND THE FAILURE IS AT THE TARGET REGIME.
        // Shipped with 5% buckets only. MEASURED the same night on a 116,212-token prompt: a bucket is
        // 5,810 tokens, and at that depth's throughput the line went **~3 minutes between prints**. I
        // could not tell running from hung and went back to `ps` and CPU-time deltas -- which is the
        // exact ambiguity this line exists to remove. The bigger the prompt, the longer the silence; a
        // 1M prompt would print at 50,000-token intervals.
        // ⇒ PERCENTAGE BOUNDS THE STORM, TIME BOUNDS THE SILENCE. Both are needed; I built one.
        //
        // ⚠ AND THE TIMER IS PER-REQUEST, NOT A FUNCTION STATIC. A shared "last printed at" would let
        // one busy sequence SUPPRESS every other sequence's progress at -np > 1 -- a guard for A
        // disabling B, the shape this fork has hit three times in one file. Keyed by request_id.
        static std::unordered_map<int32_t, int64_t> pp_last_us;
        static std::unordered_map<int32_t, int64_t> pp_start_us;
        if (is_prefill && group->n_prompt > (uint32_t) new_tokens) {
            const int32_t done_after  = n_prefill_done + new_tokens;
            const int     bucket_pre  = (int) ((int64_t) n_prefill_done * 20 / (int64_t) group->n_prompt);
            const int     bucket_post = (int) ((int64_t) done_after     * 20 / (int64_t) group->n_prompt);
            const int64_t now_us      = ggml_time_us();
            // ⚠ ENV-TUNABLE SO THE BRANCH CAN BE TESTED AT ALL. On a fast prefill a 20 s floor NEVER
            // fires, so the first verification of this fix exercised the bucket path only and proved
            // nothing about the silence path -- a fix whose branch has never been observed to run is
            // the same as no fix. DS4P_PP_FLOOR_S=0 makes it print per chunk, which is how the branch
            // is shown reachable. Default 20 s; not a tuning knob, a testability hook.
            static const int64_t floor_us = []{
                const char * e = getenv("DS4P_PP_FLOOR_S");
                return (int64_t) ((e ? atof(e) : 20.0) * 1e6);
            }();
            if (n_prefill_done == 0) {
                pp_start_us[group->request_id] = now_us;
                pp_last_us [group->request_id] = now_us;
            }
            const int64_t last_us  = pp_last_us.count(group->request_id) ? pp_last_us[group->request_id] : now_us;
            const int64_t start_us = pp_start_us.count(group->request_id) ? pp_start_us[group->request_id] : now_us;
            const bool    stale    = (now_us - last_us) >= floor_us;
            // ⚠ rid IS LOAD-BEARING AT -np > 1. Two concurrent prefills interleave into one sawtooth
            // (0.35 -> 0.10 -> 0.40) that reads exactly like the stall this line exists to rule out.
            // ⚠ ELAPSED + tok/s, because the static path reports both through the server's own
            // print_timing. Without them BOTH arms are watchable and only ONE is timeable, which is
            // how a parity question turns into a stopwatch-and-log-mtime exercise.
            if (bucket_post != bucket_pre || !mid_prefill || stale) {
                pp_last_us[group->request_id] = now_us;
                const double el = (double) (now_us - start_us) / 1e6;
                // ⚠⚠ WARN, NOT INFO, AND THAT IS THE WHOLE POINT OF THE LINE.
                // Written at LLAMA_LOG_INFO first. MEASURED: 20 correct lines at `-lv 5`, and
                // **ZERO at the server's default verbosity 3** -- libllama INFO does not reach a
                // normal user's console. An instrument whose entire purpose is that someone watching
                // a 17-minute prefill can tell running from hung, visible only to whoever already
                // knows to pass `-lv 5`, is the same defect one level up: correct producer, no reader.
                // The static path's equivalent is the server's own SLT_INF, which IS visible by
                // default; WARN is what buys parity of visibility from inside libllama.
                LLAMA_LOG_WARN("%s: paged: rid=%d prompt processing, n_tokens = %6d / %u, "
                               "progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                               __func__, group->request_id, done_after, group->n_prompt,
                               (double) done_after / (double) group->n_prompt,
                               el, el > 0.0 ? (double) done_after / el : 0.0);
            }
            // ⚠ RELEASE THE PER-REQUEST STATE ON THE LAST CHUNK. Two unordered_maps keyed by
            // request_id, in a long-lived process, with entries added per request and never removed,
            // is a leak -- small per entry and unbounded over a server's lifetime. This lane already
            // has one finding titled "paged KV cache leaks every context and buffer it allocates";
            // adding a second leak while fixing an observability gap would be a poor trade.
            if (!mid_prefill) {
                pp_last_us.erase(group->request_id);
                pp_start_us.erase(group->request_id);
            }
        }

        // ★★ KV CONTENT CHECKSUM at a FIXED lifecycle point: the LAST prefill chunk.
        //
        // Every measurement in this investigation has been BOOKKEEPING -- slot coverage, block
        // accounting, freelist order, mirror sizes, prompt_n. All of them agree between a passing
        // request 1 and a corrupted request 2. None of them looked at the bytes in the cache.
        //
        // debug_seq_kv_checksum reads back THROUGH the group's block table, i.e. resolves each token
        // the way the kernel does -- so a mismatch means the values attention would actually see
        // differ, not merely that some buffer differs.
        //
        // ⚠ The stage predicate is the SCHEDULER'S OWN (is_prefill && !mid_prefill), not a hand-rolled
        // "have we finished prefill" of mine. A private predicate could disagree with the scheduler's
        // and the disagreement would surface as a checksum mismatch I would then debug as a defect.
        if (getenv("DS4P_KVSUM") && is_prefill && !mid_prefill && kv_cache_manager != nullptr) {
            static int32_t kvsum_seq = 0;
            // ★ BISECT ON N. The checksum takes n_tokens, so sweeping it locates WHERE the two
            // runs start to diverge -- the first positional handle this defect has offered.
            //
            // The coarse default bracketed the earliest damage to (512, 1024] on a triggering prompt
            // and to (7168, np] on a clean one. Pinning it further needs points INSIDE that bracket,
            // so the list is settable: DS4P_KVSUM_NS="600,700,800,900,1000". np is always appended.
            ++kvsum_seq;
            const int32_t np = (int32_t) group->n_prompt;
            std::vector<int32_t> Ns;
            if (const char * s = getenv("DS4P_KVSUM_NS")) {
                for (const char * p = s; *p; ) {
                    const int v = atoi(p);
                    if (v > 0) { Ns.push_back(v); }
                    while (*p && *p != ',') { ++p; }
                    if (*p == ',') { ++p; }
                }
            }
            if (Ns.empty()) {
                Ns = { 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168 };
            }
            Ns.push_back(np);
            // ⚠ ALL LAYERS, NOT ONE. This printed sv[3] alone, so every positional claim it produced was
            // a claim about LAYER 3. A layer outside the sample could diverge earlier and invert the
            // causal story -- chunk 2 would be READING corrupt K/V rather than being the first thing to
            // break. DS4P_KVSUM_LAYERS raises the sample; the line now carries every layer it read.
            const int32_t nlmax = getenv("DS4P_KVSUM_LAYERS") ? atoi(getenv("DS4P_KVSUM_LAYERS")) : 8;
            std::vector<double> sv((size_t) std::max(1, nlmax), 0.0);
            for (size_t i = 0; i < Ns.size(); ++i) {
                const int32_t n = Ns[i] > np ? np : Ns[i];
                std::fill(sv.begin(), sv.end(), 0.0);
                const int32_t nl = kv_cache_manager->debug_seq_kv_checksum(*group, n, sv.data(), nlmax);
                std::string acc;
                for (int32_t l = 0; l < nl; ++l) {
                    char b[64];
                    snprintf(b, sizeof(b), " L%d=%.10g", l, sv[l]);
                    acc += b;
                }
                LLAMA_LOG_WARN("DS4P-KVSUM req#%d N=%d layers=%d%s\n", kvsum_seq, n, nl, acc.c_str());
            }
        }

        if (is_prefill) {
            GGML_ASSERT(group->logical_seq.size() >= (size_t) (n_prefill_done + new_tokens) && "logical_seq too small for prefill");
        } else {
            GGML_ASSERT(!group->logical_seq.empty() && "logical_seq empty during decode");
        }

        // Kernel sees all-GPU ids. Stored table may stay mixed.
        llama_block_ids kernel_table;
        while (!kv_cache_manager->prepare_mixed_decode(*group, kernel_table)) {
            if (!evict_held_prefix()) {
                requeue_mixed_overflow(group);
                llama_batch_free(batch);
                batch.n_tokens = 0;
                return;
            }
        }

        for (int token_idx = 0; token_idx < new_tokens; ++token_idx) {
            int32_t batch_start_id = token_offset + token_idx;

            // ★ STEP B. Decode row 0 is the last accepted token (today's behaviour). Rows 1..n are the
            // staged draft, in order. With no draft, new_tokens == 1 and this is byte-identical.
            batch.token[batch_start_id] =
                is_prefill ? group->logical_seq[n_prefill_done + token_idx]
                           : (token_idx == 0 ? group->logical_seq.back()
                                             : group->pending_draft[token_idx - 1]);
            batch.pos[batch_start_id]   = group->n_past + token_idx;  // n_past starts at 0

            batch.n_seq_id[batch_start_id]  = 1;
            batch.seq_id[batch_start_id][0] = group->request_id;

            // ★ STEP C. Prefill still emits logits only on its last row. A DECODE row set needs logits
            // on EVERY row: row 0 predicts the first drafted token, row k verifies draft[k-1] and
            // predicts the next. Without them the accept-prefix cannot be computed at all.
            //
            // Verified before writing this: output_resolve_row (llama-context.cpp:957) maps a BATCH
            // TOKEN INDEX to an output row via output_ids, so the caller samples at
            // batch_offsets[i] + k with no row math of its own -- and it THROWS BY NAME if a row's
            // logits were never requested, so getting this mask wrong is loud rather than silent.
            batch.logits[batch_start_id] =
                is_prefill ? (!mid_prefill && (token_idx == (new_tokens - 1)))
                           : true;

            int32_t token_pos                     = group->n_past + token_idx;
            curr_info.write_slots[batch_start_id] = calculate_global_slot_index(token_pos, kernel_table);
            LLAMA_LOG_DEBUG("%s: llama_batch seq_id: %d (req_id %d) token %d: pos: %d, global_slot_idx=%d\n", __func__,
                            seq_id, group->request_id, token_idx, token_pos, curr_info.write_slots[batch_start_id]);
            if (getenv("DS4P_EMIT_PROBE")) {
                LLAMA_LOG_WARN("DS4P-EMIT rid=%d row=%d pos=%d tok=%d n_past=%u lseq=%zu lseq_back=%d\n",
                               group->request_id, token_idx, token_pos, (int) batch.token[batch_start_id],
                               group->n_past, group->logical_seq.size(),
                               group->logical_seq.empty() ? -1 : (int) group->logical_seq.back());
            }
        }

        // Populate block table (1D): [batch_size * max_blocks]
        // kernel_table is the remapped all-GPU view; stored group table may be mixed.
        const int32_t curr_block_table_size = (int32_t) kernel_table.size();
        for (int block = 0; block < max_blocks; ++block) {
            int  flattened_id                   = (seq_id * max_blocks) + block;  // row-major
            bool need_padding                   = block >= curr_block_table_size;
            curr_info.block_table[flattened_id] = need_padding ? -1 : (int32_t) kernel_table[block];
        }

        curr_info.context_lens[seq_id]    = group->n_past + new_tokens;
        curr_info.batch_offsets[seq_id]   = token_offset;
        curr_info.batch_lens[seq_id]      = new_tokens;
        curr_info.prefill_pending[seq_id] = mid_prefill ? 1 : 0;
        curr_info.seq_ids[seq_id]         = group->request_id;

        // ★ CHUNK LEDGER. The damage lands on the FIRST token of the SECOND prefill chunk, at position
        // == ubatch exactly (measured at ub=512 -> 512 and ub=400 -> 400, with a clean control at 432
        // showing nothing in that range). So the quantities that decide where that token is written are
        // the ones to compare between a request that answers correctly and the poisoned one after it:
        // n_past entering the chunk, the resolved write slot of its first and last token, the group's
        // own block-table length, and the STRIDE the consumer will index with.
        //
        // ⚠ Both requests are logged, not just the failing one -- request 1 is the only baseline that
        // says what these values SHOULD be, and an error-only probe cannot produce it.
        if (getenv("DS4P_CHUNKLOG")) {
            LLAMA_LOG_WARN("DS4P-CHUNK rid=%d prefill=%d mid=%d n_past=%u new=%d "
                           "slot_first=%d slot_last=%d bt_len=%d stride=%d\n",
                           group->request_id, is_prefill ? 1 : 0, mid_prefill ? 1 : 0,
                           group->n_past, new_tokens,
                           curr_info.write_slots[token_offset],
                           curr_info.write_slots[token_offset + new_tokens - 1],
                           curr_block_table_size, max_blocks);
        }
        token_offset += new_tokens;
    }
}

// new_tokens contain 1 token per sequence in the batch
void llama_paged_scheduler_impl::update(const llama_batch &              batch,
                                        const std::vector<llama_token> & new_tokens,
                                        const int8_t *                   stop_flags,
                                        const int32_t *                  n_accepted) {
    GGML_ASSERT((int32_t) new_tokens.size() >= curr_info.n_seq && "new_tokens size does not match with batch size.");
    GGML_ASSERT(stop_flags != nullptr && "stop_flags can't be null");
    // Kernel has consumed the remapped all-GPU view. Write scratch back so
    // the stored table stays mixed (prefix GPU ids unchanged).
    kv_cache_manager->finish_mixed_decode();

    for (int i = 0; i < curr_info.n_seq; ++i) {
        int32_t token_offset = curr_info.batch_offsets[i];
        int32_t request_id   = batch.seq_id[token_offset][0];

        auto it = id_to_group.find(request_id);
        if (it == id_to_group.end()) {
            LLAMA_LOG_WARN("%s: request_id %d not found in scheduler, skipping\n", __func__, request_id);
            continue;
        }

        llama_sequence_group * group = it->second;
        GGML_ASSERT(group && "group is nullptr.");

        // chunked prefill: a mid-prompt chunk advances the prefill cursor and nothing
        // else -- no sampled token exists for it (new_tokens[i] is a caller dummy),
        // n_decoded must stay 0 so the next step still takes the prefill path
        if (curr_info.prefill_pending && curr_info.prefill_pending[i]) {
            kv_cache_manager->set_seq_max_pos(group->request_id,
                                              batch.pos[token_offset + curr_info.batch_lens[i] - 1]);
            if (kv_cache_manager->seq_pos_min(group->request_id) == -1) {
                kv_cache_manager->set_seq_min_pos(group->request_id, batch.pos[token_offset]);
            }
            group->n_past += curr_info.batch_lens[i];
            continue;
        }

        // TTFT
        if (group->n_decoded == 0) {
            group->t_first_token_us = ggml_time_us();
        }

        // Setting token ranges
        llama_pos range_min = kv_cache_manager->seq_pos_min(group->request_id);
        if (range_min == -1) {
            kv_cache_manager->set_seq_min_pos(group->request_id, batch.pos[token_offset]);
        }
        // ⚠⚠ STEP E -- A FIFTH ONE-TOKEN ASSUMPTION, IN THE SAME FUNCTION AS D'S. This set the
        // sequence's max position from the LAST SUBMITTED row. Speculation submits N+1 and may keep
        // fewer, so the range must end at the last ACCEPTED position or the cache believes the
        // sequence extends past where n_past says it does -- the same advance-vs-append fusion that
        // corrupted logical_seq, wearing different clothes.
        //
        // Computed AFTER n_acc below would be cleaner; it is kept here because range_min must be set
        // before either, so n_acc is hoisted instead.
        const int32_t n_sub_e = curr_info.batch_lens[i];
        const int32_t n_acc_e = (n_accepted == nullptr || n_accepted[i] < 0) ? n_sub_e : n_accepted[i];
        const int32_t last_accepted_idx = token_offset + n_acc_e - 1;
        kv_cache_manager->set_seq_max_pos(group->request_id, batch.pos[last_accepted_idx]);

        // ★ STEP D of paged speculation. With n_accepted == nullptr this is byte-identical to the
        // previous code: n_acc == 1, so n_past/n_decoded advance by batch_lens[i] (which IS 1 on the
        // decode path) and exactly one token is appended.
        //
        // With speculation, a sequence SUBMITS batch_lens[i] tokens (draft + 1) and may KEEP fewer.
        // Both counters must then advance by the ACCEPTED count, not the submitted one -- that is the
        // entire rollback story. Rejected tokens leave KV behind at positions beyond the new n_past,
        // and the next step simply OVERWRITES them, because DS4P_SLOT_COVER proved the write slot is
        // a pure function of position (slots[t] == btab[pos/bs]*bs + pos%bs, 0 mismatches over ~50k
        // tokens). Nothing is freed and nothing is reclaimed.
        //
        // ⚠ n_past AND logical_seq MUST ADVANCE BY THE SAME NUMBER. Every positional reader depends
        // on it: the prefix-match loops at :199 and :272, the prefill read at :797, the fork
        // inheritance in llama-kv-cache-paged.cpp:388,414. Line :190 already warns that n_past and
        // logical_seq.size() are not interchangeable -- this is the invariant that keeps them
        // reconcilable.
        // ⚠ TWO LAYOUTS FOR new_tokens, AND THEY MUST NOT BE CONFUSED. My first draft of this wrote
        // new_tokens[i * n_sub + k], which assumes every sequence submits the SAME count. It does not
        // -- batch_lens is per-sequence -- so that would have read the wrong sequence's tokens the
        // moment two sequences drafted different amounts. Caught before building, but it is exactly
        // the silent-corruption shape this lane keeps paying for, so the contract is spelled out:
        //
        //   n_accepted == nullptr   new_tokens has ONE entry per sequence, indexed [i]   (today)
        //   n_accepted != nullptr   new_tokens is laid out at the BATCH offsets, so
        //                           sequence i's tokens are [batch_offsets[i] .. +n_acc)
        //
        // The second form is unambiguous for ragged submits because batch_offsets already carries
        // the per-sequence base the rest of this function uses.
        // ⚠⚠ THE DEFAULT IS n_sub, NOT 1. My first version defaulted to 1, believing "one token per
        // sequence" was today's decode behaviour. It is not: this path also handles the FINAL PREFILL
        // CHUNK, where batch_lens[i] is the chunk size -- hundreds of tokens. Today's behaviour is
        // n_past += batch_lens[i], and defaulting to 1 advanced it by one per chunk instead.
        //
        // The assert below caught it on the first run, on four known-good architectures. Without it
        // the failure would have been a slow, plausible-looking context corruption at long prompts --
        // the ~50k defect's exact signature.
        // ⚠⚠⚠ ADVANCE-COUNT AND APPEND-COUNT ARE NOT THE SAME NUMBER, AND FUSING THEM IS A SILENT
        // LONG-CONTEXT CORRUPTION. The original code advanced n_past by batch_lens[i] and appended
        // EXACTLY ONE token, always. On a FINAL PREFILL CHUNK batch_lens[i] is the chunk size, so a
        // single loop over one fused count appends the sampled token chunk-size times: a 7-token
        // prompt puts 7 copies of the first sampled token into logical_seq and shifts every later
        // token by 6.
        //
        // I committed exactly that, labelled "no-op, checkpointed", and all four verified archs
        // stayed GREEN -- because nothing in a single short request reads the INTERIOR of
        // logical_seq. Decode reads .back(). The interior is read by the preemption/recompute requeue
        // (:417-424, which replays logical_seq into KV under memory pressure), fork inheritance
        // (llama-kv-cache-paged.cpp:414), the prefix-share matcher (:199) and on_finish (:348) --
        // none of which a short cold prompt exercises.
        //
        // So the legacy branch stays BYTE-IDENTICAL and the ragged branch owns the loop. They are
        // written apart rather than unified, because the thing that makes them look unifiable is
        // exactly the thing that is false.
        // ★ THE SENTINEL. A whole-array nullptr means legacy for every row; a NEGATIVE entry means
        // legacy for THAT row. Needed because prefill_pending only marks MID-chunk rows -- the FINAL
        // prefill chunk has prefill_pending == 0 and still requires advance=lens / append=ONE. With
        // -np>1 a batch can mix a prefilling sequence with a decoding one, so the rule has to be
        // per-row. Decided before it was exercised, so it is not invented while staring at a failure.
        // ⚠ ONE LAYOUT PER CALL, NOT PER ROW. The wrapper sizes tokens_vec from the batch offsets
        // whenever n_accepted is non-null, so a mixed array with the impl reading [i] for sentinel
        // rows would have the two disagreeing about the SAME buffer. So the contract is:
        //     n_accepted == NULL  -> one-per-sequence layout, read [i]
        //     n_accepted != NULL  -> BATCH-OFFSET layout for EVERY row, sentinel rows included
        // A sentinel row still contributes exactly ONE token; it just reads it at batch_offsets[i].
        const int32_t n_sub = n_sub_e;
        if (n_accepted == nullptr || n_accepted[i] < 0) {
            const uint32_t np_before = group->n_past;
            group->n_past    += n_sub;
            group->n_decoded += n_sub;
            group->logical_seq.push_back(n_accepted == nullptr ? new_tokens[i]
                                                               : new_tokens[token_offset]);  // ONE
            if (getenv("DS4P_NPAST_PROBE")) {
                LLAMA_LOG_WARN("DS4P-NPAST site=%s legacy rid=%d n_sub=%d n_past %u -> %u\n",
                               getenv("DS4P_CALLSITE") ? getenv("DS4P_CALLSITE") : "?", group->request_id, n_sub, np_before, group->n_past);
            }
        } else {
            const int32_t n_acc = n_accepted[i];
            GGML_ASSERT(n_acc >= 1 && n_acc <= n_sub &&
                        "accepted count must be between 1 and the number of tokens submitted");
                const uint32_t np_before = group->n_past;
            group->n_past    += n_acc;
            group->n_decoded += n_acc;
            if (getenv("DS4P_NPAST_PROBE")) {
                LLAMA_LOG_WARN("DS4P-NPAST site=%s spec rid=%d n_sub=%d n_acc=%d n_past %u -> %u\n",
                               getenv("DS4P_CALLSITE") ? getenv("DS4P_CALLSITE") : "?", group->request_id, n_sub, n_acc, np_before, group->n_past);
            }
            for (int32_t k = 0; k < n_acc; ++k) {
                group->logical_seq.push_back(new_tokens[token_offset + k]);
            }
        }

        // ★ The staged draft has now been consumed (accepted or rejected). Clearing it here is the
        // NORMAL path; the other three transitions out of decoding clear it too (recompute, swap,
        // finish), because a draft that survives any of them is replayed as accepted context.
        group->pending_draft.clear();

        // Default stop flags are n_seq_max
        if (stop_flags[i] || group->n_past >= n_seq_max_ctx) {
            group->status = llama_sequence_group_status::FINISHED;
            // eager teardown: free blocks and release the id NOW. Waiting for the next
            // running-list sweep leaves a one-step window where the server relaunches on
            // this id, queue_request overwrites the map entry, and the late sweep then
            // erased the NEW request's mapping -- orphaning it mid-flight (the storm's
            // injection point). finish() is idempotent; the sweep still removes the
            // group from the running list.
            finish(*group);
        }
    }
}

bool llama_paged_scheduler_impl::set_draft(int32_t request_id, const llama_token * draft, int32_t n_draft) {
    llama_sequence_group * group = get_group_from_id(request_id);
    if (group == nullptr) {
        LLAMA_LOG_ERROR("%s: request_id=%d does not exist.\n", __func__, request_id);
        return false;
    }
    if (n_draft <= 0 || draft == nullptr) {
        group->pending_draft.clear();          // explicit "no draft this step"
        return true;
    }

    // ⚠ REFUSE ON A PREFILLING GROUP. There is no last-accepted token to draft from until the
    // prompt has been consumed, and staging one would emit rows against a position range the
    // prefill cursor still owns.
    if (group->n_decoded == 0) {
        LLAMA_LOG_ERROR("%s: request_id=%d is still prefilling (n_decoded=0); nothing to draft from.\n",
                        __func__, request_id);
        return false;
    }

    // ⚠ CAP AGAINST THE BATCH BUDGET. paged requires n_batch == n_ubatch (refused at context
    // construction; scheduler init is the second wall). An oversized draft used to surface later
    // as a startup abort far from its cause. The +1 is
    // the last accepted token, which shares the row block with the draft.
    if (n_draft + 1 > n_batch) {
        LLAMA_LOG_ERROR("%s: request_id=%d draft of %d tokens (+1 verify row) exceeds n_batch=%d.\n",
                        __func__, request_id, n_draft, n_batch);
        return false;
    }

    group->pending_draft.assign(draft, draft + n_draft);
    return true;
}

void llama_paged_scheduler_impl::set_on_finish(llama_paged_on_finish_cb cb, void * user_data) {
    on_finish_cb        = cb;
    on_finish_user_data = user_data;
}

llama_sequence_group * llama_paged_scheduler_impl::find_parent_group(int32_t parent_request_id) const {
    auto it = id_to_group.find(parent_request_id);
    if (it != id_to_group.end() && it->second != nullptr) {
        return it->second;
    }
    for (const auto & h : held_prefixes) {
        if (h && h->request_id == parent_request_id) {
            return h.get();
        }
    }
    return nullptr;
}

void llama_paged_scheduler_impl::rebind_session_after_finish(const llama_sequence_group & group) {
    auto rit = request_sessions.find(group.request_id);
    if (rit == request_sessions.end()) {
        return;
    }
    const std::string sid = rit->second;
    request_sessions.erase(rit);

    const uint32_t n_full = block_size ? (group.n_past / block_size) * block_size : 0;
    const uint32_t n_keep = std::min(group.n_past, (uint32_t) group.logical_seq.size());
    const uint32_t n_cmp  = n_full > 0 ? n_full : n_keep;
    int32_t        hold_id = 0;
    if (n_cmp > 0 && group.logical_seq.size() >= n_cmp) {
        for (const auto & h : held_prefixes) {
            if (!h || h->logical_seq.size() < n_cmp) {
                continue;
            }
            bool match = true;
            for (uint32_t i = 0; i < n_cmp; ++i) {
                if (h->logical_seq[i] != group.logical_seq[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                hold_id = h->request_id;
                break;
            }
        }
    }
    if (hold_id != 0) {
        sessions[sid] = hold_id;
        request_sessions[hold_id] = sid;
        LLAMA_LOG_INFO("%s: session '%s' now names held prefix %d\n",
                       __func__, sid.c_str(), hold_id);
    } else {
        sessions.erase(sid);
        LLAMA_LOG_INFO("%s: session '%s' dropped -- no prefix to keep\n",
                       __func__, sid.c_str());
    }
}

bool llama_paged_scheduler_impl::bind_session(const std::string & session_id, int32_t request_id) {
    if (session_id.empty()) {
        return false;
    }
    llama_sequence_group * g = find_parent_group(request_id);
    if (g == nullptr) {
        LLAMA_LOG_ERROR("%s: request %d not found; cannot bind session '%s'\n",
                        __func__, request_id, session_id.c_str());
        return false;
    }
    auto sit = sessions.find(session_id);
    if (sit != sessions.end() && sit->second != request_id) {
        request_sessions.erase(sit->second);
    }
    auto rit = request_sessions.find(request_id);
    if (rit != request_sessions.end() && rit->second != session_id) {
        sessions.erase(rit->second);
    }
    sessions[session_id] = request_id;
    request_sessions[request_id] = session_id;
    LLAMA_LOG_INFO("%s: session '%s' bound to request %d\n",
                   __func__, session_id.c_str(), request_id);
    return true;
}

bool llama_paged_scheduler_impl::queue_forked_from_session(llama_sequence_group group,
                                                          const std::string & session_id) {
    auto it = sessions.find(session_id);
    if (it == sessions.end()) {
        LLAMA_LOG_ERROR("%s: session '%s' not found\n",
                        __func__, session_id.c_str());
        return false;
    }
    if (find_parent_group(it->second) == nullptr) {
        LLAMA_LOG_ERROR("%s: session '%s' names request %d but parent group not found\n",
                        __func__, session_id.c_str(), it->second);
        return false;
    }
    return queue_forked_request(std::move(group), it->second);
}

bool llama_paged_scheduler_impl::close_session(const std::string & session_id) {
    auto it = sessions.find(session_id);
    if (it == sessions.end()) {
        return false;
    }
    const int32_t id = it->second;
    sessions.erase(it);
    request_sessions.erase(id);

    // Live master: just unbind. Its own refs stay until it finishes.
    if (id_to_group.count(id)) {
        LLAMA_LOG_INFO("%s: session '%s' unbound from live request %d\n",
                       __func__, session_id.c_str(), id);
        return true;
    }

    // Another session still names this hold -- drop only our name.
    for (const auto & kv : sessions) {
        if (kv.second == id) {
            LLAMA_LOG_INFO("%s: session '%s' closed; hold %d still named by '%s'\n",
                           __func__, session_id.c_str(), id, kv.first.c_str());
            return true;
        }
    }

    for (auto hit = held_prefixes.begin(); hit != held_prefixes.end(); ++hit) {
        if (!(*hit) || (*hit)->request_id != id) {
            continue;
        }
        llama_sequence_group * h = hit->get();
        if (h->block_table.empty() && kv_cache_manager) {
            if (const auto * bl = kv_cache_manager->get_sequence_blocks(h->request_id)) {
                h->block_table = *bl;
            }
        }
        // free_blocks decrements THIS hold's refs only. Children that
        // inherited the prefix keep theirs. Unique suffix (ref_cnt==1)
        // returns to the pool. Shared prefix stays.
        if (kv_cache_manager && !h->block_table.empty()) {
            kv_cache_manager->free_blocks(*h);
        }
        held_prefixes.erase(hit);
        LLAMA_LOG_INFO("%s: session '%s' closed; dropped extra refs on hold %d\n",
                       __func__, session_id.c_str(), id);
        return true;
    }

    LLAMA_LOG_INFO("%s: session '%s' closed; hold %d already gone\n",
                   __func__, session_id.c_str(), id);
    return true;
}

bool llama_paged_scheduler_impl::has_session(const std::string & session_id) const {
    return sessions.find(session_id) != sessions.end();
}

int32_t llama_paged_scheduler_impl::session_request_id(const std::string & session_id) const {
    auto it = sessions.find(session_id);
    return it == sessions.end() ? 0 : it->second;
}

llama_sequence_group * llama_paged_scheduler_impl::get_group_from_id(int32_t request_id) const {
    return id_to_group.count(request_id) ? id_to_group.at(request_id) : nullptr;
}

const llama_paged_batch_info * llama_paged_scheduler_impl::get_curr_batch_info() const {
    return &curr_info;
}
