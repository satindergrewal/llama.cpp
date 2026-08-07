#include "llama-paged-scheduler-impl.h"

#include "llama-impl.h"

llama_paged_scheduler_impl::llama_paged_scheduler_impl(uint32_t               n_ctx,
                                                       uint32_t               block_sz,
                                                       int32_t                n_batch,
                                                       llama_kv_cache_paged * kv_manager) :
    n_seq_max_ctx(n_ctx),
    block_size(block_sz),
    n_batch(n_batch),
    kv_cache_manager(kv_manager) {}

bool llama_paged_scheduler_impl::check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const {
    if (n_candidates == 0 && (n_swapped > 0 || n_waiting > 0)) {
        LLAMA_LOG_ERROR(
            "%s: Scheduler deadlock detected. "
            "%d sequence(s) are swapped out and %d are waiting, "
            "but there are not enough free GPU blocks to make progress. "
            "Hint: increase n_gpu_blocks (currently %d) or reduce n_sequences.\n",
            __func__, n_swapped, n_waiting, kv_cache_manager->get_num_gpu_blocks());
        return true;
    }
    return false;
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

    const int32_t remaining = n_batch - get_curr_decode_tokens();
    process_waiting_list(candidates, remaining);

    const uint32_t n_running    = running.size();
    const uint32_t n_swapped    = swapped.size();
    const uint32_t n_waiting    = waiting.size();
    const uint32_t n_candidates = candidates.size();

    LLAMA_LOG_INFO("%s: Scheduler status: running=%d, swapped=%d, waiting=%d, candidates=%d\n", __func__, n_running,
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
        } else {
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
            uint32_t               best_n   = 0;
            llama_sequence_group * best_src = nullptr;

            if (kv_cache_manager != nullptr && !group.logical_seq.empty()) {
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

            if (best_n > 0 && best_src != nullptr) {
                // fork_blocks reads src.block_table, which is empty on a live group. Give it a
                // source view whose table is the cache's authoritative copy.
                llama_sequence_group src_view = *best_src;
                if (const auto * bl = kv_cache_manager->get_sequence_blocks(best_src->request_id)) {
                    src_view.block_table = *bl;
                }
                const uint32_t shared = kv_cache_manager->fork_blocks(src_view, group, best_n);
                if (shared > 0) {
                    group.n_past = shared;
                    LLAMA_LOG_INFO("%s: request %d admitted SHARED: %u of %u prompt tokens inherited "
                                   "from live request %d (%zu blocks), %u left to prefill\n",
                                   __func__, group.request_id, shared, group.n_prompt,
                                   best_src->request_id, group.block_table.size(),
                                   group.n_prompt - shared);
                }
            }

            // cold request (or a fork, which brings its own blocks): nothing may be left
            // parked under this id or it pins pool blocks nobody is ever going to claim
            kv_cache_manager->discard_restored(group.request_id);
        }
    }

    auto group_ptr = std::make_unique<llama_sequence_group>(std::move(group));

    id_to_group[group_ptr->request_id] = group_ptr.get();

    set_waiting(std::move(group_ptr));
    return true;
}

bool llama_paged_scheduler_impl::queue_forked_request(llama_sequence_group group, int32_t parent_request_id) {
    // hybrid archs: the recurrent members hold per-seq state that cannot be rewound to
    // the fork point -- inherited attention KV without matching recurrent state is wrong
    // (and measured as a segfault on the fixture). Degrade loudly to a full prefill.
    if (is_hybrid) {
        LLAMA_LOG_WARN("%s: request %d: fork unsupported on hybrid archs (recurrent state "
                       "cannot rewind); queueing as a normal request\n",
                       __func__, group.request_id);
        return queue_request(std::move(group));
    }

    auto it = id_to_group.find(parent_request_id);
    if (it == id_to_group.end() || it->second == nullptr) {
        LLAMA_LOG_ERROR("%s: parent request %d not found; queueing as a normal request\n",
                        __func__, parent_request_id);
        return queue_request(std::move(group));
    }

    const llama_sequence_group & parent = *it->second;

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
    insert_sorted_by_arrival_time(std::move(group_ptr), swapped);
}

void llama_paged_scheduler_impl::set_waiting(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::WAITING &&
                "Request is already waiting.");
    group_ptr->status = llama_sequence_group_status::WAITING;
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
    // We prioritize user CB, otherwise we log by default
    if (on_finish_cb) {
        // TODO perhaps just have the callback take sequence_group and user_data
        on_finish_cb(group.request_id, group.logical_seq.data(), (int32_t) group.logical_seq.size(),
                     on_finish_user_data);
    } else {
        LLAMA_LOG_DEBUG("%s: Request: %d generated %d tokens.\n", __func__, group.request_id, group.n_decoded);
    }
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
    if ((uint64_t) group_ptr->n_past + 1 > pool_tokens) {
        LLAMA_LOG_ERROR("%s: request %d needs %llu tokens of KV but the GPU block pool holds at most "
                        "%llu (%u blocks x %u). No eviction or recompute can create capacity that "
                        "does not exist -- terminating the request instead of retrying forever.\n",
                        __func__, rid, (unsigned long long) (group_ptr->n_past + 1),
                        (unsigned long long) pool_tokens,
                        kv_cache_manager->get_usable_gpu_blocks(), block_size);
        // finish() asserts the status first, then frees blocks and erases our id mapping.
        terminated_ids.push_back(rid);   // dedicated channel: ONLY capacity kills
        group_ptr->status = llama_sequence_group_status::FINISHED;
        finish(*group_ptr);
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

    LLAMA_LOG_DEBUG("%s: (recomputation) request_id=%d was sent for recomputation.\n", __func__, rid);
    set_waiting(std::move(group_ptr));
}

void llama_paged_scheduler_impl::evict() {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    LLAMA_LOG_DEBUG("%s: Eviction requested...\n", __func__);
    if (running.empty()) {
        return;
    }

    llama_sequence_group_ptr most_recent_request = std::move(running.back());
    running.pop_back();
    GGML_ASSERT(most_recent_request && "request selected for eviction is nullptr.");

    swap_out_or_recompute(std::move(most_recent_request));
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
        uint32_t required_capacity = group->n_past + 1;
        LLAMA_LOG_DEBUG(
            "%s: (running) request_id=%d: current_capacity (tokens)=%d toks, required capacity (tokens) = %d toks\n",
            __func__, group->request_id, current_capacity, required_capacity);
        if (required_capacity >= current_capacity) {
            LLAMA_LOG_DEBUG("%s: (running_pending) request_id=%d: requires a new block to decode.\n", __func__,
                            group->request_id);
            bool success = kv_cache_manager->allocate(1, *group);  // decode phase
            if (!success) {
                if (running.size() > 1) {
                    const bool curr_is_back = (std::next(it) == running.end());
                    // Evict pops the back of the list (most recent request)
                    evict();

                    // Evict might have removed the current group from running
                    if (curr_is_back) {
                        // Current group was evicted
                        it = running.end();
                        continue;
                    }

                    // Try allocating again after eviction
                    success = kv_cache_manager->allocate(1, *group);
                }

                if (!success) {
                    // If allocate failed, it means we must evict the current request
                    llama_sequence_group_ptr self = std::move(*it);
                    it                            = running.erase(it);

                    swap_out_or_recompute(std::move(self));
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

        ++count;
        // When prefilling, we want to always guarantee at least one decode to avoid thrashing.
        // allocate() already counts n_prompt + n_decoded internally, so the argument is the
        // DELTA only: passing tokens_needed (= n_prompt+1) double-counted the prompt and
        // demanded ~2x the blocks -- admission serialized every fat request (running=1
        // always in the starved walls) and eviction/recompute became unreachable.
        const bool success = kv_cache_manager->allocate(1, *group);
        if (!success) {
            // We respect FCFS, so we stop here to prevent a younger waiting request from jumping ahead.
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
    // allocation (evict() pops running.back()): its table then holds CPU block ids and
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
                chunk_tokens[i] = 1;
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

        if (is_prefill) {
            GGML_ASSERT(group->logical_seq.size() >= (size_t) (n_prefill_done + new_tokens) && "logical_seq too small for prefill");
        } else {
            GGML_ASSERT(!group->logical_seq.empty() && "logical_seq empty during decode");
        }

        for (int token_idx = 0; token_idx < new_tokens; ++token_idx) {
            int32_t batch_start_id = token_offset + token_idx;

            batch.token[batch_start_id] = is_prefill ? group->logical_seq[n_prefill_done + token_idx]
                                                     : group->logical_seq.back();
            batch.pos[batch_start_id]   = group->n_past + token_idx;  // n_past starts at 0

            batch.n_seq_id[batch_start_id]  = 1;
            batch.seq_id[batch_start_id][0] = group->request_id;

            // only the last token, and never for a mid-prompt chunk
            batch.logits[batch_start_id] = !mid_prefill && (token_idx == (new_tokens - 1));

            int32_t token_pos                     = group->n_past + token_idx;
            curr_info.write_slots[batch_start_id] = calculate_global_slot_index(token_pos, group->block_table);
            LLAMA_LOG_DEBUG("%s: llama_batch seq_id: %d (req_id %d) token %d: pos: %d, global_slot_idx=%d\n", __func__,
                            seq_id, group->request_id, token_idx, token_pos, curr_info.write_slots[batch_start_id]);
        }

        // Populate block table (1D): [batch_size * max_blocks]
        const int32_t curr_block_table_size = group->block_table.size();
        for (int block = 0; block < max_blocks; ++block) {
            int  flattened_id                   = (seq_id * max_blocks) + block;  // row-major
            bool need_padding                   = block >= curr_block_table_size;
            curr_info.block_table[flattened_id] = need_padding ? -1 : group->block_table[block];
        }

        curr_info.context_lens[seq_id]    = group->n_past + new_tokens;
        curr_info.batch_offsets[seq_id]   = token_offset;
        curr_info.batch_lens[seq_id]      = new_tokens;
        curr_info.prefill_pending[seq_id] = mid_prefill ? 1 : 0;
        curr_info.seq_ids[seq_id]         = group->request_id;
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
        int32_t last_token_in_batch_idx = token_offset + curr_info.batch_lens[i] - 1;
        kv_cache_manager->set_seq_max_pos(group->request_id, batch.pos[last_token_in_batch_idx]);

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
        const int32_t n_sub = curr_info.batch_lens[i];
        if (n_accepted == nullptr || n_accepted[i] < 0) {
            group->n_past    += n_sub;
            group->n_decoded += n_sub;
            group->logical_seq.push_back(new_tokens[i]);   // ONE, always -- as before
        } else {
            const int32_t n_acc = n_accepted[i];
            GGML_ASSERT(n_acc >= 1 && n_acc <= n_sub &&
                        "accepted count must be between 1 and the number of tokens submitted");
            group->n_past    += n_acc;
            group->n_decoded += n_acc;
            for (int32_t k = 0; k < n_acc; ++k) {
                group->logical_seq.push_back(new_tokens[token_offset + k]);
            }
        }

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

void llama_paged_scheduler_impl::set_on_finish(llama_paged_on_finish_cb cb, void * user_data) {
    on_finish_cb        = cb;
    on_finish_user_data = user_data;
}

llama_sequence_group * llama_paged_scheduler_impl::get_group_from_id(int32_t request_id) const {
    return id_to_group.count(request_id) ? id_to_group.at(request_id) : nullptr;
}

const llama_paged_batch_info * llama_paged_scheduler_impl::get_curr_batch_info() const {
    return &curr_info;
}
