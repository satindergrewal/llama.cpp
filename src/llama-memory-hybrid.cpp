#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-kv-cache-paged.h"

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
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
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llama_memory_hybrid::~llama_memory_hybrid() = default;

void llama_memory_hybrid::set_attn_paged(llama_kv_cache_paged * paged) {
    mem_attn_paged.reset(paged);
}

llama_kv_cache_paged * llama_memory_hybrid::get_mem_attn_paged() const {
    return mem_attn_paged.get();
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // When the paged pool is active AND its scheduler has set batch info, carry the paged
        // context alongside (same ubatches -- the recurrent split constraints win). Dark until
        // the scheduler drives it: has_paged_batch_info() is false without that, so this cannot
        // trip the init ordering assert. Mirrors llama_memory_hybrid_iswa::init_batch.
        // ⚠⚠ THIS JUSTIFICATION IS STALE AS OF 2026-08-05 -- READ BEFORE RELYING ON IT.
        // The premise below ("nothing in llama-server calls the scheduler") is now FALSE: the
        // server calls llama_paged_scheduler_add_request AND llama_paged_scheduler_prepare_batch
        // (2 sites in server-context.cpp). And the consumer chain this bridged was FIXED the same
        // day by wiring the generic capability-based paged consumer, so a graph CAN now consume the
        // pool without this.
        //
        // MEASURED CONSEQUENCE: with the server driving the scheduler AND this bridge active, TWO
        // group objects allocate for the same sequence from one pool (8 checkouts for a 4-block
        // need; one object IS sd_group, the other is a scheduler group). Large pools hide it.
        // At -ngpub 8 it exhausts the pool, self_drive_begin fails to grow, FREES its own KV, bails
        // to the static path, and the output corrupts from token 48. One-factor, marker-verified:
        //     DRIVE_ON   paged_dispatches=2  selfdrive=8  checkouts=8  -> corrupt
        //     DRIVE_OFF  paged_dispatches=2  selfdrive=0  checkouts=4  -> clean
        // The full quant-KV e2e gate passes ALL FIVE ARMS with this bridge OFF.
        //
        // NOT removed here: this lane has verified two architectures and one gate suite, which is
        // narrower than "safe to delete" (the taint probe and other callers are unaudited). Left
        // behaviourally untouched, annotated so the next reader does not inherit a false premise.
        // It was TRUE when written; it aged the moment the thing it described was fixed.
        //
        // --- original comment, kept verbatim for provenance ---
        // ★ SELF-DRIVE bridge (DS4P_PAGED_DRIVE). Nothing in llama-server calls the scheduler's
        // step(), so has_paged_batch_info() is never true there and no graph can consume the pool
        // -- the last link of the Ornith consumer chain. Drive a single sequence here using the
        // cache's OWN allocator; the scheduler keeps multi-seq admission/eviction.
        // Deliberately narrow: exactly one ubatch, n_seq == 1. Anything else falls through
        // untouched, so the scheduler-driven path and the static default are both unchanged.
        // ⚠ WAS GATED ON !has_paged_batch_info(). That made self-drive fire EXACTLY ONCE -- on the
        // 2-token warmup probe -- and every later prefill and decode step then ran on that stale
        // slot map, because nothing ever clears the info. The decode gate PASSED anyway, which is
        // precisely why it had to be instrumented: a green result produced by the wrong mechanism.
        //
        // Self-drive now re-derives per ubatch. self_drive_begin() already calls self_drive_end()
        // first, so the previous allocation and info are released before the new ones are built.
        // Still skipped when a real scheduler is driving (sd_active false + info present).
        if (mem_attn_paged && mem_attn_paged->self_drive_enabled() &&
            ubatches.size() == 1 && ubatches[0].n_seqs == 1 &&
            (mem_attn_paged->self_drive_active() || !mem_attn_paged->has_paged_batch_info())) {
            mem_attn_paged->self_drive_begin((int32_t) ubatches[0].n_tokens);
        }

        // ★ DS4P_DECODE_TRACE: is the batch info visible HERE, when the paged child context would be
        // built? Measured on Qwen3.6: pool built, scheduler running, request registered, block
        // checked out -- and get_attn_paged() still returns nullptr in the graph, so all 110 layer
        // instances take the static path. Either prepare_batch is not setting info for this hybrid
        // composition, or init_batch runs before it. Log it instead of reasoning about it.
        if (getenv("DS4P_DECODE_TRACE")) {
            LLAMA_LOG_WARN("DS4P-HYB init_batch: mem_attn_paged=%d has_paged_batch_info=%d ubatches=%zu\n",
                           mem_attn_paged != nullptr,
                           mem_attn_paged ? (int) mem_attn_paged->has_paged_batch_info() : -1,
                           ubatches.size());
        }

        llama_memory_context_ptr paged_ctx;
        if (mem_attn_paged && mem_attn_paged->has_paged_batch_info()) {
            paged_ctx = mem_attn_paged->init_batch_with_ubatches(ubatches); // copy: hybrid ctx owns the originals
        }

        auto ctx = std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));

        if (paged_ctx) {
            // ⚠⚠ EMIT THE VALUE THE FALLBACK WARNING TELLS YOU TO COMPARE AGAINST.
            // `build_attn_paged_or_null`'s static-path warning says, verbatim:
            //     "compare against DS4P-SET: same pointer = the wrapper set it and the consumer
            //      still sees null; different = the graph holds a stale context"
            // Measured 2026-08-10 across two logs: **110 and 210 mentions of DS4P-SET, and ZERO
            // independent set-value lines.** Every occurrence was inside the warning's own text.
            // **The instruction pointed at a number no log has ever printed** -- an annotation
            // referring to itself, which is worse than none, because a reader spends time hunting
            // for the counterpart. (measured by Grok, verified here before the fix)
            //
            // DEBUG level: costs nothing at the -lv 4 the gates run, and appears at the -lv 5
            // anyone reaches for when they are actually debugging this.
            LLAMA_LOG_DEBUG("%s: DS4P-SET attn paged ctx=%p on hybrid ctx=%p\n",
                            __func__, (const void *) paged_ctx.get(), (const void *) ctx.get());
            ctx->set_attn_paged_ctx(std::move(paged_ctx));
        }

        return ctx;
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return mem_attn->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);

    // ★ DELEGATE TO THE PAGED POOL. Without this the paged KV is NEVER SERIALISED: on the paged
    // path the static members hold almost nothing (the KV lives in the pool), so a slot save wrote
    // 716 B of metadata for a 27-token sequence whose real KV is ~1 MiB -- 0.068% -- and reported
    // SUCCESS with n_saved=27. The serialiser existed and was correct; it simply had NO CALLER.
    // grep "attn_paged->state_write" across src/ returned zero hits before this line.
    //
    // Ordering is load-bearing: paged goes LAST on write and LAST on read, so the two streams stay
    // in the same order. Mismatch them and the read desynchronises silently -- it will not throw,
    // it will restore the wrong bytes.
    if (mem_attn_paged) {
        mem_attn_paged->state_write(io, seq_id, flags);
    }
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);

    // MUST mirror state_write's order exactly -- see the note there.
    if (mem_attn_paged) {
        mem_attn_paged->state_read(io, seq_id, flags);
    }
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

const llama_kv_cache_paged_context * llama_memory_hybrid_context::get_attn_paged() const {
    return static_cast<const llama_kv_cache_paged_context *>(ctx_attn_paged.get());
}
