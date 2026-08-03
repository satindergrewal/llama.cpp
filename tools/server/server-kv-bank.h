#pragma once

// P1-5 (3b lane): disk-persisted KV bank behind the RAM prompt cache.
// v0 increment 1: spill-on-eviction only (writer + witness); probe/admit comes next.
// Bring-up config via DS4P_KV_BANK=<dir> (off when unset); flag plumbing follows once
// the admit side lands. Design: ornith-1m/docs/DESIGN-P15-DISK-BANKS.md.

#include <cstdint>
#include <string>

struct server_prompt_cache_state;

class server_kv_bank {
  public:
    static server_kv_bank & instance();

    bool active() const { return !dir.empty(); }

    // best-effort spill of an entry about to be evicted; never throws, never blocks
    // serving on failure (log + drop, per the design)
    void spill(const server_prompt_cache_state & entry);

  private:
    server_kv_bank();

    std::string dir;      // empty = disabled
    uint64_t    n_spilled = 0;
};
