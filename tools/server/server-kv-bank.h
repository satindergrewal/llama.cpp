#pragma once

// P1-5 (3b lane): disk-persisted KV bank behind the RAM prompt cache.
// v0 increment 1: spill-on-eviction only (writer + witness); probe/admit comes next.
// Bring-up config via DS4P_KV_BANK=<dir> (off when unset); flag plumbing follows once
// the admit side lands. Design: ornith-1m/docs/DESIGN-P15-DISK-BANKS.md.

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct server_prompt_cache_state;
struct server_tokens;

class server_kv_bank {
  public:
    static server_kv_bank & instance();

    bool active() const { return !dir.empty(); }

    // best-effort spill of an entry about to be evicted; never throws, never blocks
    // serving on failure (log + drop, per the design)
    void spill(const server_prompt_cache_state & entry);

    // probe the bank for the entry whose token prefix best matches tokens_new and, when it
    // clears the similarity floor, rebuild it into `out` for the caller to admit into the
    // RAM tier (which then loads/tail-replays through the normal path). Returns false on
    // miss, unreadable file, or identity mismatch -- never on a half-filled `out`.
    bool probe(const server_tokens & tokens_new, const std::string & identity_cur,
               server_prompt_cache_state & out);

  private:
    server_kv_bank();

    void enforce_cap();

    std::string dir;       // empty = disabled
    uint64_t    cap_bytes = 0; // DS4P_KV_BANK_CAP_MIB; 0 = uncapped
    uint64_t    n_spilled = 0;
    uint64_t    n_admitted = 0;
    uint64_t    n_probe_miss = 0;
};
