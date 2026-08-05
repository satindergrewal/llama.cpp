#pragma once

#include "llama-batch.h"
#include "llama-block-manager.h"
#include "llama-graph.h"
#include "llama-memory.h"
#include "llama-sequence-group.h"

#include <cmath>

//
// llama_kv_cache_paged
//

class llama_kv_cache_paged : public llama_memory_i {
  public:
    llama_kv_cache_paged(uint32_t head_dim,
                         uint32_t n_head_kv,
                         uint32_t block_size,
                         uint32_t n_layers,
                         uint32_t n_ubatch,
                         uint32_t n_seq_max);

    // Multi-device init: one backend per layer, so a model split across devices
    // (llama.cpp splits by LAYER, not by head) keeps each layer's KV on the same
    // device as the layer itself. Block ids stay device-agnostic because every
    // device allocates the SAME number of blocks, so a block id is valid anywhere.
    void init_multi(const std::vector<ggml_backend_t> & layer_backends,
                    ggml_backend_t backend_cpu,
                    enum ggml_type type,
                    uint32_t       n_gpu_blocks,
                    uint32_t       n_cpu_blocks,
                    float          watermark);

    void init(ggml_backend_t backend_gpu,
              ggml_backend_t backend_cpu,
              enum ggml_type type,
              uint32_t       n_gpu_blocks,
              uint32_t       n_cpu_blocks,
              float          watermark);  // percentage

    bool allocate(int32_t num_tokens, llama_sequence_group & group);
    void free_blocks(llama_sequence_group & group);

    // P1-6 COW fork: point dst at src's blocks for the shared prefix and take a reference
    // on each -- zero KV bytes copied, vs the static path's O(n_ctx) cell copy. The tail
    // block is only shared when it is FULL; a partially-filled tail would be written by
    // both sequences, so dst gets a private copy of it (boundary copy-on-fork).
    // Returns the number of tokens dst inherits.
    uint32_t fork_blocks(const llama_sequence_group & src, llama_sequence_group & dst, uint32_t n_shared_tokens);
    bool swap_in(llama_sequence_group & group);
    bool swap_out(llama_sequence_group & group);

    // ★ SELF-DRIVE (DS4P_PAGED_DRIVE). set_paged_batch_info() is called from exactly ONE place --
    // llama_paged_scheduler_impl::step() -- and llama-server never calls it, so has_paged_batch_info()
    // is always false there and no graph can consume the pool. That is the last link in the Ornith
    // consumer chain (audit findings 5/7).
    //
    // The full fix is P2-8 continuous batching. But the BLOCK ALLOCATOR is ours, not the
    // scheduler's (block_manager below, allocate()/free_blocks()), so a SINGLE sequence can be
    // driven here directly without that rewrite. This owns the group and the info for one ubatch.
    //
    // Scope, deliberately narrow: n_seq == 1, one ubatch at a time. Multi-seq admission, eviction
    // and preemption stay with the scheduler. Freed and re-allocated on every call.
    // ★ ATTENTION-ONLY POOL (#2436). The pool allocated one KV tensor for EVERY model layer, but a
    // hybrid's recurrent layers hold no KV at all -- Ornith-9B is 32 layers of which only 16 are
    // attention, so HALF the pool was allocated for layers that can never use it.
    //
    // Set BEFORE init(): layers marked false get no tensor and get_k()/get_v() return nullptr.
    // That composes with llm_graph_context::paged_layer_supported(), which already treats a null
    // KV tensor as "not pageable" -- so no caller needs to know about this filter.
    void set_layer_filter(std::vector<uint8_t> has_kv);

    // Per-layer head geometry, for interleaved-SWA architectures where global and sliding layers
    // have different head_dim. Must be called BEFORE init()/init_multi() -- it decides allocation.
    // Passing empty vectors (or not calling this) keeps the uniform behaviour exactly.
    void set_layer_geometry(std::vector<uint32_t> head_dims, std::vector<uint32_t> n_heads_kv);

    bool self_drive_begin(int32_t n_tokens);
    void self_drive_end();
    void self_drive_release_info();
    bool self_drive_enabled() const;
    // true while WE own the current batch info (as opposed to a real scheduler)
    bool self_drive_active() const { return sd_active; }

    void     set_paged_batch_info(const llama_paged_batch_info * info);
    uint32_t get_num_gpu_blocks() const;

    // DEBUG (fork-residual discriminator): additive checksum of the group's first
    // n_tokens of KV per layer, read back through its block table. Returns layers written.
    int32_t debug_seq_kv_checksum(const llama_sequence_group & group, int32_t n_tokens,
                                  double * out_sums, int32_t max_layers) const;

    //
    // llama_memory_i
    //
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) override;

    // 3b (paged hybrid): same contract as init_batch but the caller owns the ubatch split.
    // The hybrid wrapper must split per its recurrent side's constraints and feed the SAME
    // ubatches to both members, so it cannot let this cache consume balloc itself.
    llama_memory_context_ptr init_batch_with_ubatches(std::vector<llama_ubatch> ubatches);

    // whether the scheduler has set batch info for the current batch (the init_batch
    // ordering precondition); lets callers route conditionally instead of tripping the assert
    bool has_paged_batch_info() const { return last_paged_info != nullptr; }

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    struct ggml_tensor * get_kv_tensor(int layer_idx) const;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;

    void seq_cp(llama_seq_id /*seq_id_src*/,
                llama_seq_id /*seq_id_dst*/,
                llama_pos /*p0*/,
                llama_pos /*p1*/) override { /* implement later CoW mechanism */
    }

    void seq_keep(llama_seq_id /*seq_id*/) override {}

    void seq_add(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, llama_pos /*shift*/) override {}

    void seq_div(llama_seq_id /*seq_id*/, llama_pos /*p0*/, llama_pos /*p1*/, int /*d*/) override {}

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load -- see the .cpp for what is implemented and what is not.
    void state_write(llama_io_write_i & io,
                     llama_seq_id seq_id             = -1,
                     llama_state_seq_flags flags     = 0) const override;

    void state_read(llama_io_read_i & io,
                    llama_seq_id seq_id             = -1,
                    llama_state_seq_flags flags     = 0) override;

    // P1-5 ADMIT handshake. state_read materialises a restored sequence's KV into real
    // blocks but has no scheduler group to attach them to yet (it runs on the server's
    // admit path, before the request is queued). These two calls are how the blocks get
    // from the cache to the group -- or back to the pool if nobody wants them.
    //
    // take_restored_blocks: hand over at most n_tokens_wanted tokens' worth of restored
    // blocks, exactly once. Blocks covering tokens beyond the cap are released here (the
    // admitted prompt diverged before the end of the stored KV). Returns tokens handed
    // over, 0 if there is no parked restore for this sequence.
    uint32_t take_restored_blocks(llama_seq_id seq_id, uint32_t n_tokens_wanted, llama_block_ids & out_blocks);

    // release a parked restore nobody adopted -- unadopted blocks are a leak, not a cache
    void discard_restored(llama_seq_id seq_id);

    //
    // Helpers to llama_memory_i
    //
    void set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min);
    void set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max);

  private:
    void concat_block_ids(llama_block_ids & to_block_table, const llama_block_ids & from_block_table);

    // maximal runs of consecutive same-device block ids, as (first_id, count) -- lets the
    // state serdes move a whole run per backend call instead of one call per block
    std::vector<std::pair<uint32_t, uint32_t>> contiguous_runs(const llama_block_ids & blocks) const;
    void do_block_copy(const llama_block_ids & src_ids, const llama_block_ids & new_ids, bool to_gpu);

    // Master physical buffer
    // For CUDA: memory is interleaved
    // For other backends: we treat the exact same memory buffer as two virtual views
    std::vector<struct ggml_tensor *> kv_gpu_layers;
    std::vector<struct ggml_tensor *> kv_cpu_layers;

    enum ggml_type kv_type;

    llama_block_manager block_manager;

    // Non-owning pointer to the batch currently being processed.
    // Lifetime: set by the scheduler at the end of step(), cleared at the
    // start of the next step() (before the batch's paged_* arrays are freed).
    // The ordering in llama_paged_scheduler_impl::clear_batch is load-bearing;
    // do not reorder without updating init_batch's contract.
    const llama_paged_batch_info * last_paged_info = nullptr;

    // self-drive state (see self_drive_begin). Owned here so it outlives the graph build.
    llama_sequence_group   sd_group;
    llama_paged_batch_info sd_info;
    bool                   sd_active = false;

    // empty = every layer has KV (default, unchanged behaviour)
    std::vector<uint8_t> layer_has_kv;

    // ★ PER-LAYER GEOMETRY. Empty = every layer uses the scalars below, which is the old behaviour
    // exactly -- so architectures with uniform heads are untouched and carry no risk from this.
    //
    // The pool already stored ONE TENSOR PER LAYER; only the DESCRIPTION of their shape was
    // pool-wide, by convention rather than necessity. Making it per-layer is what lets an
    // interleaved-SWA model page its global AND its sliding layers from a single pool, instead of
    // a second pool with its own block table, allocator and scheduler state.
    //
    // Block COUNT stays shared and so do block ids: a block id must be valid in every layer, and it
    // is, because each layer's tensor has the same number of blocks regardless of row width.
    std::vector<uint32_t> layer_head_dim;     // empty, or n_layers entries
    std::vector<uint32_t> layer_n_heads_kv;   // empty, or n_layers entries
    std::vector<uint32_t> layer_block_bytes;  // derived in init(); empty = use block_bytes

    uint32_t hd_of(uint32_t il)  const { return layer_head_dim.empty()    ? head_dim    : layer_head_dim[il];    }
    uint32_t hkv_of(uint32_t il) const { return layer_n_heads_kv.empty()  ? n_heads_kv  : layer_n_heads_kv[il];  }
    // ⚠ USE THIS, NEVER block_bytes, anywhere a LAYER's bytes are meant. do_block_copy staged
    // through one buffer sized from the pool-wide value; with per-layer widths that either truncates
    // a copy or reads past the source, on the GPU<->CPU eviction path -- which runs rarely, under
    // memory pressure, and is not exercised by any gate here. Silent corruption under load is the
    // worst thing to ship, so the accessor exists to make the pool-wide value hard to use by accident.
    uint32_t bb_of(uint32_t il)  const { return layer_block_bytes.empty() ? block_bytes : layer_block_bytes[il]; }

    const uint32_t head_dim;
    const uint32_t n_heads_kv;
    const uint32_t block_size;
    const uint32_t n_layers;
    const uint32_t n_ubatch;
    const uint32_t n_seq_max;
    uint32_t       num_gpu_blocks;
    uint32_t       num_cpu_blocks;
    uint32_t       block_bytes;

    // one context+buffer per distinct device holding layers (multi-device path)
    std::vector<struct ggml_context *>     gpu_ctxs;
    std::vector<ggml_backend_buffer_t>     gpu_bufs;

    ggml_backend_t gpu_backend;
    ggml_backend_t cpu_backend;

    struct seq_range {
        llama_pos min = -1;
        llama_pos max = -1;
    };

    std::unordered_map<llama_seq_id, seq_range> sequence_positions;

    // P1-5 PREREQUISITE: which physical blocks each sequence owns.
    //
    // Why this has to exist: state_write/state_read are CACHE-level interface methods,
    // but until now the cache could not answer "which blocks hold sequence N's KV". It
    // held only sequence_positions (min/max pos) and a NON-OWNING pointer to the CURRENT
    // batch's block table -- so a sequence's blocks were unknowable outside the batch it
    // happened to be in. That is precisely why state_write/state_read were empty stubs,
    // and why the disk KV bank (P1-5) could never spill: llama_state_seq_get_size_ext
    // returned 0 because there was nothing the cache could enumerate.
    //
    // Maintained at the six points where the cache already sees a group's request_id and
    // block_table together (allocate / free_blocks / fork_blocks / swap_in / swap_out),
    // so it stays exact across preemption and forking rather than being rebuilt by
    // guesswork. Kept private to the cache: the scheduler remains the owner of policy,
    // this is only the cache's record of physical residency.
    std::unordered_map<llama_seq_id, llama_block_ids> sequence_blocks;

    void note_seq_blocks(const llama_sequence_group & group);

    // P1-5 RESTORE: KV that state_read has already written into real blocks, parked until
    // the scheduler queues the matching request and adopts it. The park exists because the
    // two halves run in different places: state_read is driven by the server's prompt-cache
    // admit (llama_state_seq_set_data_ext), which happens BEFORE llama_paged_scheduler_-
    // add_request creates the group. Holding the blocks across that gap is the whole
    // handshake -- see take_restored_blocks / discard_restored.
    struct restored_seq {
        llama_block_ids blocks;
        llama_pos       p_min = -1;
        llama_pos       p_max = -1;
    };

    std::unordered_map<llama_seq_id, restored_seq> restored_seqs;
};

class llama_kv_cache_paged_context : public llama_memory_context_i {
  public:
    llama_kv_cache_paged_context(llama_kv_cache_paged * parent, const std::vector<llama_ubatch> & in_ubatch) :
        manager(parent),
        ubatches(in_ubatch) {
        i_cur = 0;
    }

    llama_kv_cache_paged_context(llama_memory_status status) : status(status) {}

    void    set_batch_data(const llama_paged_batch_info & info);
    int32_t get_n_tokens() const;
    int32_t get_batch_size() const;
    int32_t get_max_blocks() const;

    int32_t * get_write_slots() const;
    int32_t * get_block_table() const;
    int32_t * get_context_lens() const;
    int32_t * get_batch_offsets() const;
    int32_t * get_batch_lens() const;

    void set_n_tokens(int32_t new_n_tokens);
    void set_batch_size(int32_t new_batch_size);
    void set_max_blocks(int32_t new_max_blocks);

    struct ggml_tensor * get_k(int layer_idx) const;
    struct ggml_tensor * get_v(int layer_idx) const;

    //
    // llama_memory_context_i
    //
    bool                 next() override;
    bool                 apply() override;
    const llama_ubatch & get_ubatch() const override;

    llama_memory_status get_status() const override { return status; }

  private:
    const llama_kv_cache_paged * manager;

    //
    // batch processing context
    //
    std::vector<llama_ubatch> ubatches;
    size_t                    i_cur = 0;      // index of ubatch to process

    int32_t * paged_write_slots   = nullptr;  // [n_tokens]
    int32_t * paged_block_table   = nullptr;  // [batch_size, max_blocks]
    int32_t * paged_context_lens  = nullptr;  // [batch_size]
    int32_t * paged_batch_offsets = nullptr;  // [batch_size]
    int32_t * paged_batch_lens    = nullptr;  // [batch_size]

    int32_t n_tokens   = 0;
    int32_t batch_size = 0;
    int32_t max_blocks = 0;

    llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS;
};
