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

    // flags override env (env stays as the bring-up path); called once at server init
    void configure(const std::string & dir_in, int32_t cap_mib);

    // best-effort spill of an entry about to be evicted; never throws, never blocks
    // serving on failure (log + drop, per the design)
    void spill(const server_prompt_cache_state & entry);

    // probe the bank for the entry whose token prefix best matches tokens_new and, when it
    // clears the similarity floor, rebuild it into `out` for the caller to admit into the
    // RAM tier (which then loads/tail-replays through the normal path). Returns false on
    // miss, unreadable file, or identity mismatch -- never on a half-filled `out`.
    bool probe(const server_tokens & tokens_new, const std::string & identity_cur,
               server_prompt_cache_state & out);

    // P1-5 ECONOMICS. A restore is only worth doing when reading the state back is cheaper
    // than recomputing it, and which way that goes is a property of the MODEL, not of the
    // feature: state bytes per token are fixed by the architecture, while prefill cost per
    // token scales with parameter count. Measured on qwen3-4b IQ4_KT, reading 1.59 GiB of
    // KV costs about what prefilling 11K tokens costs, so admitting unconditionally made
    // the warm request 1.64x SLOWER than the cold one (2,862 vs 1,742 ms wall). Both rates
    // are therefore measured at runtime and each request decides for itself -- there is no
    // constant here to get wrong on the next model.
    //
    // Feed ONLY genuinely cold prefills in here. A warm request prefills a single token in
    // a window that contains its own restore, and letting that into the estimate would have
    // the estimator learning from the very thing it is deciding about.
    void note_prefill(size_t n_tokens, double ms);

    // The COMPLETE cost of turning a bank file into live KV, measured by the caller: the
    // read, the 1.6 GB of buffer the entry is rebuilt into, and llama_state_seq_set_data's
    // upload. Timing only the fread understated it by ~2.6x and would have had the gate
    // admitting restores it should decline -- a confident wrong answer, which is worse than
    // the honest bootstrap.
    void note_restore(size_t bytes, double ms);
    bool bank_rate_known() const { return ewma_restore_mib_per_ms > 0.0; }

  private:
    server_kv_bank();

    void enforce_cap();

    std::string dir;       // empty = disabled
    uint64_t    cap_bytes = 0; // DS4P_KV_BANK_CAP_MIB; 0 = uncapped
    uint64_t    n_spilled = 0;
    uint64_t    n_admitted = 0;
    uint64_t    n_probe_miss = 0;
    uint64_t    n_uneconomic = 0;

    // 0 = never measured; until both sides are known the bank admits and learns from it
    double ewma_prefill_ms_per_tok = 0.0;
    double ewma_restore_mib_per_ms = 0.0;
};
