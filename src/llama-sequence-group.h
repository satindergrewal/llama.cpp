#pragma once

#include "llama.h"

#include <cstdint>
#include <list>
#include <memory>
#include <vector>

enum class llama_sequence_group_status { PENDING, WAITING, RUNNING, SWAPPED, FINISHED };

using llama_block_ids = std::vector<uint32_t>;

struct llama_sequence_group {
    int32_t                     request_id = -1;
    llama_sequence_group_status status     = llama_sequence_group_status::PENDING;

    int64_t t_arrival_time   = 0;
    int64_t t_first_token_us = 0;

    uint32_t n_prompt  = 0;
    uint32_t n_decoded = 0;
    uint32_t n_past    = 0;

    // finish() idempotence: teardown may run eagerly at update-time AND from the
    // running-list sweep; the second call must no-op (double free_blocks otherwise)
    bool torn_down = false;

    std::vector<llama_token> logical_seq;
    llama_block_ids          block_table;

    // ★ SPECULATIVE DECODING: tokens staged by llama_paged_scheduler_set_draft() for the NEXT
    // prepare_batch(). step() emits 1 + pending_draft.size() rows for this group instead of one.
    //
    // ⚠⚠ THIS MUST BE CLEARED ON EVERY TRANSITION OUT OF A DECODING STATE, not just after it is
    // consumed. The group can leave RUNNING three other ways, and a draft that survives any of them
    // is fed as though it were accepted context:
    //   update()         normal consumption
    //   recompute        n_past/n_decoded reset to 0 and logical_seq is REPLAYED from the start
    //   set_swapped()    parked, resumed later against a different cache state
    //   finish()         teardown, which runs from TWO places (see torn_down)
    // State that survives a transition nobody considered, read by a path no checkpoint exercises, is
    // exactly the shape of the logical_seq corruption this file already paid for.
    std::vector<llama_token> pending_draft;
};

using llama_sequence_group_raw_list = std::vector<llama_sequence_group *>;
using llama_sequence_group_ptr      = std::unique_ptr<llama_sequence_group>;
using llama_sequence_group_list     = std::list<llama_sequence_group_ptr>;
