#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

class llama_kv_cache_paged;

//
// llama_memory_hybrid_iswa
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache_iswa to
//   support models where each layer may be either attention-based (with SWA support) or recurrent

class llama_memory_hybrid_iswa : public llama_memory_i {
public:
    llama_memory_hybrid_iswa(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr);

    // defined in the .cpp where llama_kv_cache_paged is complete (unique_ptr member)
    ~llama_memory_hybrid_iswa();

    // 3b (DS4P_PAGED_HYBRID): hand ownership of a constructed+init'd paged attention pool
    // to this wrapper. Built by create_memory, where the backends are in scope (same shape
    // as the flat-arch paged path). Inert until the paged hybrid graph path lands.
    void set_attn_paged(llama_kv_cache_paged * paged);

    llama_kv_cache_paged * get_mem_attn_paged() const;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_iswa specific API
    //

    llama_kv_cache_iswa * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache_iswa> mem_attn;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;

    // 3b: optional paged attention pool (replaces mem_attn's role when active); see
    // set_attn_paged. Not const: handed in post-construction by create_memory.
    std::unique_ptr<llama_kv_cache_paged> mem_attn_paged;
};

class llama_memory_hybrid_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_iswa_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_iswa_context(llama_memory_hybrid_iswa * mem);

    // init update
    explicit llama_memory_hybrid_iswa_context(
        llama_memory_hybrid_iswa * mem,
                   llama_context * lctx,
                            bool   optimize);

    // init success
    llama_memory_hybrid_iswa_context(
           llama_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_iswa_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_hybrid_iswa_context
    //

    const llama_kv_cache_iswa_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

    // 3b: the paged attention context when the wrapper's paged pool is active AND the
    // scheduler has set batch info for it; nullptr otherwise (static path unaffected)
    const llama_kv_cache_paged_context * get_attn_paged() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;

    // 3b: set post-construction by the wrapper's init_batch when the paged pool is active
    // and scheduler batch info is present; see set_attn_paged_ctx
    llama_memory_context_ptr ctx_attn_paged;

    const llama_memory_status status;

public:
    void set_attn_paged_ctx(llama_memory_context_ptr ctx) { ctx_attn_paged = std::move(ctx); }
};
