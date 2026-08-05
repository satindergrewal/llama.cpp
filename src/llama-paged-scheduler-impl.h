#pragma once

#include "llama-kv-cache-paged.h"

#include <clocale>
#include <vector>

enum class llama_scheduler_status {
    OK,
    DEADLOCK,  // cannot make progress
};

class llama_paged_scheduler_impl {
  public:
    llama_paged_scheduler_impl(uint32_t n_ctx, uint32_t block_sz, int32_t n_batch, llama_kv_cache_paged * kv_manager);

    llama_scheduler_status step(llama_batch & batch);

    // n_warm: how many leading tokens of group.logical_seq the caller has already restored
    // into the paged cache (P1-5 admit). 0 = a cold request, which is the normal case.
    bool                   queue_request(llama_sequence_group group, uint32_t n_warm = 0);

    // P1-6: queue `group` as a COW fork of an existing request -- it inherits the parent's
    // prefix blocks by reference (fork_blocks) instead of prefilling them again.
    bool                   queue_forked_request(llama_sequence_group group, int32_t parent_request_id);
    void update(const llama_batch & batch, const std::vector<llama_token> & new_tokens, const int8_t * stop_flags);
    void set_on_finish(llama_paged_on_finish_cb cb, void * user_data);
    llama_sequence_group *         get_group_from_id(int32_t request_id) const;
    const llama_paged_batch_info * get_curr_batch_info() const;

    // hybrid archs keep recurrent state next to the paged pool; a fork cannot rewind
    // that state to the fork point, so forking must degrade to a full-prefill request
    void set_hybrid(bool v) { is_hybrid = v; }

    // DEBUG accessor for the fork-residual checksum API
    const llama_kv_cache_paged * kv_cache() const { return kv_cache_manager; }

  private:
    bool is_hybrid = false;

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
    void set_waiting(llama_sequence_group_ptr group);

    void finish(llama_sequence_group & group);

    int32_t get_curr_decode_tokens() const;

    void evict();
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

    // Used for fast lookups
    std::unordered_map<int32_t, llama_sequence_group *> id_to_group;

    const uint32_t         n_seq_max_ctx;
    const uint32_t         block_size;
    const int32_t          n_batch;
    llama_kv_cache_paged * kv_cache_manager = nullptr;
    llama_paged_batch_info curr_info;

    uint32_t n_livelock_steps   = 0;
    uint32_t prev_n_swapped     = 0;
    uint32_t max_livelock_steps = 20;

    // Callback for output tracking
    llama_paged_on_finish_cb on_finish_cb        = nullptr;
    void *                   on_finish_user_data = nullptr;
};
