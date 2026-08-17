#pragma once

#include "llama-kv-cache-paged.h"

#include <clocale>
#include <string>
#include <unordered_map>
#include <vector>

enum class llama_scheduler_status {
    OK,
    DEADLOCK,  // cannot make progress
};

// Dense always forks. Hybrid forks only if RS can rewind, or there is no RS.
// True non-rewindable hybrid (SSM) must refuse -- do not degrade to cold/APC.
inline bool llama_paged_fork_allowed(bool is_hybrid, bool supports_rs_rollback,
                                     bool has_recurrent_state) {
    if (!is_hybrid) {
        return true;
    }
    return supports_rs_rollback || !has_recurrent_state;
}

class llama_paged_scheduler_impl {
  public:
    // n_seq_max_batch: how many sequences may share ONE decode batch.
    // 0 = no extra cap (tests / examples that do not go through llama_decode).
    // The server passes llama_n_seq_max(ctx). That is BATCH WIDTH, not an
    // admission ceiling: queue_request still accepts every request the pool
    // can hold. Raising -np to fake concurrency would also raise this and
    // kill the champion / slice n_ctx. Do not do that.
    llama_paged_scheduler_impl(uint32_t n_ctx, uint32_t block_sz, int32_t n_batch,
                               llama_kv_cache_paged * kv_manager, uint32_t n_seq_max_batch = 0);
    ~llama_paged_scheduler_impl();

    llama_scheduler_status step(llama_batch & batch);

    // n_warm: how many leading tokens of group.logical_seq the caller has already restored
    // into the paged cache (P1-5 admit). 0 = a cold request, which is the normal case.
    bool                   queue_request(llama_sequence_group group, uint32_t n_warm = 0);

    // P1-6: queue `group` as a COW fork of an existing request -- it inherits the parent's
    // prefix blocks by reference (fork_blocks) instead of prefilling them again.
    bool                   queue_forked_request(llama_sequence_group group, int32_t parent_request_id);

    // Named session: a harness says "this request is the master". The name
    // follows the live request, then the parked prefix after finish().
    // Omitted session_id is a no-op -- existing callers stay unchanged.
    bool                   bind_session(const std::string & session_id, int32_t request_id);
    bool                   queue_forked_from_session(llama_sequence_group group, const std::string & session_id);
    // Drop the session's extra hold refs. Shared prefix stays while children
    // still hold refs. Unique suffix returns to the pool.
    bool                   close_session(const std::string & session_id);
    bool                   has_session(const std::string & session_id) const;
    int32_t                session_request_id(const std::string & session_id) const;
    size_t                 n_held_prefixes() const { return held_prefixes.size(); }
    // Held prefix length for a parked id (0 if unknown). Tests assert a
    // named master stays at N, not N-k, across a tight-pool child wait.
    size_t                 held_prefix_n_blocks(int32_t request_id) const;
    // ★ STEP D of paged speculation (FINDINGS-paged-no-speculation.md).
    //
    // `n_accepted` is OPTIONAL and nullable. When null, every sequence accepts exactly ONE token --
    // today's behaviour, byte-identical, which is what makes this step a pure refactor and keeps the
    // two out-of-server callers (tests/test-paged-kv-e2e.cpp, examples/paged/paged.cpp) compiling
    // unchanged.
    //
    // When present, n_accepted[i] is how many of sequence i's SUBMITTED tokens were kept. Speculation
    // submits N+1 and may keep fewer.
    //
    // ⚠ THE SAFETY CONDITION, and every positional reader of logical_seq depends on it: logical_seq
    // extends by the ACCEPTED count and n_past advances by the ACCEPTED count -- the SAME number,
    // every step. Rejected tokens are never appended, so the prefix-match loops (impl.cpp:199, :272)
    // and the prefill read (:797) stay correct. If those two ever advance by different amounts, every
    // fork and prefix path silently corrupts. impl.cpp:190 already warns that n_past and
    // logical_seq.size() are not interchangeable.
    void update(const llama_batch & batch, const std::vector<llama_token> & new_tokens, const int8_t * stop_flags,
                const int32_t * n_accepted = nullptr);
    // ★ SPECULATIVE DECODING INPUT CHANNEL. step() builds decode rows from logical_seq.back() --
    // state the scheduler already owns -- so drafted tokens have no way in without this.
    // Returns false (and stages nothing) if the request is unknown, still prefilling, or the draft
    // does not fit the batch budget.
    bool set_draft(int32_t request_id, const llama_token * draft, int32_t n_draft);

    // ★ SERVER-SIDE ABORT. The server can fail a request the scheduler still considers live --
    // the DEADLOCK verdict path and task-cancel both release the slot server-side, and until this
    // existed NOTHING told the scheduler, so the dead request kept its queue entry and its block
    // claims forever. Measured 2026-08-11: after one designed pool-capacity termination, a 12-token
    // request against a 64-block pool re-deadlocked indefinitely ("2 waiting"). Removes the request
    // from whichever queue holds it, frees its blocks (via finish()), erases the id mapping.
    // Returns false if the id is unknown (already finished, or never queued) -- safe to call twice.
    bool abort_request(int32_t request_id);

    void set_on_finish(llama_paged_on_finish_cb cb, void * user_data);
    llama_sequence_group *         get_group_from_id(int32_t request_id) const;
    const llama_paged_batch_info * get_curr_batch_info() const;

    // hybrid archs keep recurrent state next to the paged pool. Fork is allowed
    // when that state can rewind (supports_rs_rollback) or there is none to rewind.
    // True non-rewindable hybrid (SSM) must refuse -- do not degrade to cold/APC.
    void set_hybrid(bool v) { is_hybrid = v; }
    void set_supports_rs_rollback(bool v) { supports_rs_rollback = v; }
    void set_has_recurrent_state(bool v) { has_recurrent_state = v; }
    bool can_fork() const {
        return llama_paged_fork_allowed(is_hybrid, supports_rs_rollback, has_recurrent_state);
    }
    // True only when the last queue_forked_* actually called fork_blocks.
    // Distinguishes a real fork from an APC share after a silent degrade.
    bool last_fork_used_blocks() const { return last_fork_used_blocks_; }
    // Whole-block tokens the last queue_request inherited by refcount (0 if
    // none). Stock /v1/chat/completions uses this as timings.cache_n.
    uint32_t last_inherit_tokens() const { return last_inherit_tokens_; }

    // DEBUG accessor for the fork-residual checksum API
    const llama_kv_cache_paged * kv_cache() const { return kv_cache_manager; }

  private:
    bool is_hybrid             = false;
    bool supports_rs_rollback  = false;
    bool has_recurrent_state   = false;
    bool last_fork_used_blocks_ = false;
    uint32_t last_inherit_tokens_ = 0;

    void insert_sorted_by_arrival_time(llama_sequence_group_ptr new_group, llama_sequence_group_list & list);

    bool check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const;
    bool check_livelock(uint32_t n_swapped, uint32_t prev_n_swapped);

  public:
    // ★ DEDICATED ABORT CHANNEL. on_finish is GENERIC -- it fires for normal completions too, and
    // request_id IS the slot id, which the server REUSES. So a drain keyed on on_finish cannot tell
    // "the old group for slot 0 finished" from "the request currently on slot 0 was killed", and it
    // errors live requests. Measured: every cache_prompt=true repeat got a spurious 500.
    // finish() already warns about exactly this ("a new request may already have reused this id").
    // Only capacity terminations land here.
    std::vector<int32_t> terminated_ids;

  private:

    void set_running(llama_sequence_group_ptr group);
    void set_swapped(llama_sequence_group_ptr group);
    // prepend=true is vLLM PREEMPTED: the victim resumes before later arrivals
    // once blocks free. New admits stay FCFS (prepend=false).
    void set_waiting(llama_sequence_group_ptr group, bool prepend = false);

    void finish(llama_sequence_group & group);
    // Mixed remap leftover still short after unique-suffix swap:
    // requeue the waiter (HTTP live). Do not terminate / 500.
    // fail_mixed_remap_once is a compatibility wrapper that queues.
    void requeue_mixed_overflow(llama_sequence_group * group);
    void fail_mixed_remap_once(llama_sequence_group * group);
    // Keep the finished request's full-block prefix so later independent
    // arrivals can still SHARED-admit (vLLM APC). Not a radix tree.
    // A named session with n_past > 0 but n_full == 0 still parks a
    // name-only hold (0 blocks) so children can resolve the name.
    void park_finished_prefix(llama_sequence_group & group);
    // After park: point the session name at the hold. Keep the name
    // even if the hold has 0 full blocks. Drop only if nothing to name.
    void rebind_session_after_finish(const llama_sequence_group & group);
    llama_sequence_group * find_parent_group(int32_t parent_request_id) const;
    // Reclaim GPU from a parked prefix. Non-session holds: drop unref
    // suffix (release). Named session holds: swap unref suffix to CPU
    // without shortening n_past / logical_seq. NEVER a block with
    // ref_cnt > 1 (would rewrite a child's inherited GPU ids).
    bool evict_held_prefix();
    bool is_named_session_id(int32_t request_id) const;

    int32_t get_curr_decode_tokens() const;

    const llama_block_ids * blocks_of(const llama_sequence_group & group) const;
    uint32_t                count_unref_blocks(const llama_sequence_group & group) const;
    llama_sequence_group *  find_master_prefix_group() const;
    bool                    evict();  // true if a victim was swapped/recomputed

    void process_running_list(llama_sequence_group_raw_list & candidates);
    void process_swapped_list(llama_sequence_group_raw_list & candidates);
    void process_waiting_list(llama_sequence_group_raw_list & candidates, int32_t remaining_token_bugdet);

    void swap_out_or_recompute(llama_sequence_group_ptr group_ptr);

    int32_t calculate_global_slot_index(int32_t token_pos, std::vector<uint32_t> & block_table);

    void clear_batch(llama_batch & batch);
    void populate_batch_from(llama_sequence_group_raw_list & candidates, llama_batch & batch);

    llama_sequence_group_list running;
    llama_sequence_group_list swapped;
    llama_sequence_group_list waiting;

    // Finished requests' prefixes. Full-block holds carry extra refs
    // for APC. Name-only holds (0 blocks) keep a session resolvable
    // when n_past < block_size. Negative request_ids so a reused slot
    // id cannot collide with a hold.
    llama_sequence_group_list held_prefixes;
    int32_t                   next_hold_id = -2;

    // Used for fast lookups
    std::unordered_map<int32_t, llama_sequence_group *> id_to_group;

    // session_id -> live request_id or parked hold id. Reverse map so
    // finish() can retarget the name onto the hold without a scan.
    std::unordered_map<std::string, int32_t> sessions;
    std::unordered_map<int32_t, std::string> request_sessions;

    const uint32_t         n_seq_max_ctx;
    const uint32_t         block_size;
    const int32_t          n_batch;
    const uint32_t         n_seq_max_batch;
    llama_kv_cache_paged * kv_cache_manager = nullptr;
    llama_paged_batch_info curr_info;

    uint32_t n_livelock_steps   = 0;
    uint32_t prev_n_swapped     = 0;
    uint32_t max_livelock_steps = 20;

    // Callback for output tracking
    llama_paged_on_finish_cb on_finish_cb        = nullptr;
    void *                   on_finish_user_data = nullptr;
};
