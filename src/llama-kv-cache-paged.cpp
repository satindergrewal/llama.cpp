#include "llama-io.h"

#include <stdexcept>
#include "llama-kv-cache-paged.h"

#include <algorithm>

#include "llama-impl.h"

//
// llama_kv_cache_paged
//


// KV pool fill value. 0 normally; 0xFF under DS4P_KV_POISON, which makes every fp16 a NaN so
// that ANY read of an unwritten block turns the output to NaN and fails loudly.
//
// PROVEN INVARIANT (keep this probe -- it is how the invariant was established): nothing ever
// reads an unwritten KV block. Filling all four pools with NaN left the server's output sha
// BIT-IDENTICAL to the zeroed control (8cb6c8212e both ways), with 4 markers proving the fill
// ran and 0 in the control. The scalar path bounds its token loop rather than reading and then
// masking, so unwritten positions are never visited at all.
//
// The marker is NOT optional. Getting here took three tries: the first patched only two of the
// four clear sites and the test used a third; the second used LLAMA_LOG_INFO, which is off in a
// bare test binary; and test-paged-vs-cpu turned out never to construct a paged cache at all,
// so it could not answer the question in principle. Each failure looked exactly like "ALL
// PASSED".
//
// ⚠ REFUTED, do not retry blind: skipping the clear does NOT make the pool lazily committed.
// Measured RSS 58.67 GB with the clear skipped vs 58.68 GB with it, on a deliberately huge
// 20x-headroom pool. The residency path (rset addAllocation + requestResidency in
// ggml-metal-device.m) wires the buffer regardless of whether anything touches it, so
// kvcached-style lazy commit on Metal needs sparse/placement MTLHeap, not a missing memset.
static uint8_t ds4p_kv_fill(const char * where) {
    if (getenv("DS4P_KV_POISON") == nullptr) {
        return 0;
    }
    // fprintf, not LLAMA_LOG_INFO: the llama logger may be disabled in a bare test binary,
    // and a marker that depends on log configuration is a marker that can lie by omission --
    // which is exactly how the first two attempts at this probe reported nothing.
    fprintf(stderr, "DS4P-KV-POISON: filling %s with 0xFF (every fp16 = NaN)\n", where);
    return 0xFF;
}

// true when layer il should own a KV tensor. Empty filter = all layers (unchanged default).
// Defined HERE, above init(): it was originally placed next to allocate() further down and the
// build failed with "use of undeclared identifier" at four init sites. Declaration order matters.
static inline bool ds4p_layer_kv(const std::vector<uint8_t> & f, uint32_t il) {
    return f.empty() || (il < f.size() && f[il] != 0);
}

void llama_kv_cache_paged::set_layer_geometry(std::vector<uint32_t> head_dims,
                                              std::vector<uint32_t> heads_kv) {
    if (head_dims.empty() || heads_kv.empty()) {
        layer_head_dim.clear();
        layer_n_heads_kv.clear();
        return;
    }

    // Loud rather than clever: a geometry vector of the wrong length would index out of bounds in
    // hd_of()/hkv_of() on some later layer -- a crash or a corruption far from the mistake.
    GGML_ASSERT(head_dims.size() == n_layers && heads_kv.size() == n_layers &&
                "per-layer geometry must have exactly n_layers entries");

    // DS4P_NO_LAYER_GEOMETRY=1 forces the uniform pool back, so this feature's effect is a
    // ONE-FACTOR measurement inside a single binary rather than a cross-build comparison -- same
    // reasoning as DS4P_NO_LAYER_FILTER, and the same fallback if an untested arch misbehaves.
    if (getenv("DS4P_NO_LAYER_GEOMETRY")) {
        LLAMA_LOG_INFO("%s: DS4P_NO_LAYER_GEOMETRY -- uniform pool geometry (per-layer disabled)\n", __func__);
        layer_head_dim.clear();
        layer_n_heads_kv.clear();
        return;
    }

    layer_head_dim   = std::move(head_dims);
    layer_n_heads_kv = std::move(heads_kv);

    // State the DISTINCT geometries. "per-layer geometry is on" says nothing about whether it
    // changed anything -- on a uniform model this prints one line and is correctly
    // indistinguishable from the feature being off.
    for (uint32_t il = 0; il < n_layers; ++il) {
        bool seen = false;
        for (uint32_t j = 0; j < il; ++j) {
            if (layer_head_dim[j] == layer_head_dim[il] && layer_n_heads_kv[j] == layer_n_heads_kv[il]) { seen = true; break; }
        }
        if (seen) { continue; }
        uint32_t n = 0;
        for (uint32_t j = 0; j < n_layers; ++j) {
            if (layer_head_dim[j] == layer_head_dim[il] && layer_n_heads_kv[j] == layer_n_heads_kv[il]) { ++n; }
        }
        LLAMA_LOG_INFO("%s: per-layer geometry: head_dim=%u n_head_kv=%u on %u/%u layers\n",
                       __func__, layer_head_dim[il], layer_n_heads_kv[il], n, n_layers);
    }
}

void llama_kv_cache_paged::set_layer_filter(std::vector<uint8_t> has_kv) {
    // DS4P_NO_LAYER_FILTER=1 restores the old all-layers pool. Two reasons it exists: it makes the
    // saving a ONE-FACTOR measurement instead of a cross-build comparison, and it is the fallback
    // if the filter ever misbehaves on an arch I have not tested.
    if (getenv("DS4P_NO_LAYER_FILTER")) {
        LLAMA_LOG_INFO("%s: DS4P_NO_LAYER_FILTER -- allocating ALL layers (filter disabled)\n", __func__);
        layer_has_kv.clear();
        return;
    }

    size_t n_kv = 0;
    for (uint8_t v : has_kv) { n_kv += v ? 1 : 0; }
    LLAMA_LOG_INFO("%s: attention-only pool: %zu of %zu layers hold KV\n",
                   __func__, n_kv, has_kv.size());

    layer_has_kv = std::move(has_kv);
}

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max) :
    kv_type(GGML_TYPE_F16),
    head_dim(head_dim),
    n_heads_kv(n_heads_kv),
    block_size(block_size),
    n_layers(n_layers),
    n_ubatch(n_ubatch),
    n_seq_max(n_seq_max),
    num_gpu_blocks(0),
    num_cpu_blocks(0),
    gpu_backend(nullptr),
    cpu_backend(nullptr) {}

void llama_kv_cache_paged::init_multi(const std::vector<ggml_backend_t> & layer_backends,
                                      ggml_backend_t backend_cpu,
                                      enum ggml_type type,
                                      uint32_t       n_gpu_blocks,
                                      uint32_t       n_cpu_blocks,
                                      float          watermark) {
    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");
    GGML_ASSERT(layer_backends.size() == n_layers && "need one backend per layer");
    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");

    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    kv_type        = type;
    cpu_backend    = backend_cpu;
    gpu_backend    = layer_backends[0];  // representative; per-layer truth is in the vector
    // ⚠ WAS head_dim * ggml_type_size(kv_type). ggml_type_size is bytes per BLOCK, not per
    // element, so for q8_0 (32 elements in a 34-byte block) this multiplied by 34 instead of
    // dividing by 32 -- roughly 34x oversized, with every stride wrong. Harmless while f16 is the
    // only admitted type (1 element, 2 bytes, so the two happen to agree), which is exactly why it
    // survived: the bug is invisible until the feature it breaks is switched on.
    // ggml_row_size handles both cases.
    block_bytes    = 2 * block_size * n_heads_kv * ggml_row_size(kv_type, head_dim);

    // Per-layer bytes, derived from per-layer geometry when present. Everything that copies or
    // serialises a LAYER's block must use bb_of(il), never the pool-wide block_bytes.
    layer_block_bytes.clear();
    if (!layer_head_dim.empty()) {
        layer_block_bytes.resize(n_layers);
        for (uint32_t il = 0; il < n_layers; ++il) {
            layer_block_bytes[il] = 2 * block_size * hkv_of(il) * (uint32_t) ggml_row_size(kv_type, hd_of(il));
        }
    }

    // Group layers by the device that holds them. llama.cpp's --tensor-split splits
    // by LAYER, so layer il's KV must live on dev_layer(il). Every device allocates
    // the same n_gpu_blocks, which is what keeps a block id valid on all of them and
    // leaves the block table, scheduler and attention kernel completely unchanged.
    std::vector<ggml_backend_t> distinct;
    for (auto * be : layer_backends) {
        if (std::find(distinct.begin(), distinct.end(), be) == distinct.end()) {
            distinct.push_back(be);
        }
    }

    LLAMA_LOG_INFO("%s: paged KV across %zu device(s), n_gpu_blocks=%u per device, "
                   "n_cpu_blocks=%u, block_size=%u, watermark=%0.2f\n",
                   __func__, distinct.size(), n_gpu_blocks, n_cpu_blocks, block_size, watermark);

    kv_gpu_layers.assign(n_layers, nullptr);

    for (auto * be : distinct) {
        size_t n_here = 0;
        for (uint32_t il = 0; il < n_layers; ++il) {
            if (layer_backends[il] == be && ds4p_layer_kv(layer_has_kv, il)) n_here++;
        }

        struct ggml_init_params gp;
        gp.mem_size   = ggml_tensor_overhead() * 5 * n_here;
        gp.mem_buffer = NULL;
        gp.no_alloc   = true;

        struct ggml_context * ctx = ggml_init(gp);
        GGML_ASSERT(ctx && "failed to create paged KV context");

        for (uint32_t il = 0; il < n_layers; ++il) {
            if (layer_backends[il] != be) continue;
            if (!ds4p_layer_kv(layer_has_kv, il)) continue;   // attention-only pool
            kv_gpu_layers[il] =
                ggml_new_tensor_4d(ctx, type, hd_of(il), block_size, 2 * hkv_of(il), n_gpu_blocks);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        GGML_ASSERT(buf && "Failed to allocate paged KV buffer on a device");
        ggml_backend_buffer_clear(buf, ds4p_kv_fill("multi-device GPU pool"));

        LLAMA_LOG_INFO("%s:   device %s holds %zu layer(s), %.2f MiB\n", __func__,
                       ggml_backend_name(be), n_here,
                       ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);

        gpu_ctxs.emplace_back(ctx);
        gpu_bufs.emplace_back(buf);
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }
        GGML_ASSERT(kv_gpu_layers[il] && "layer tensor not allocated");
        GGML_ASSERT(kv_gpu_layers[il]->buffer && "layer tensor has null buffer");
    }

    // CPU mirror for the swap path, identical to the single-device init: one tensor
    // per layer, allocated on the CPU backend with pinned memory for faster PCIe.
    struct ggml_init_params cpu_params;
    cpu_params.mem_size           = ggml_tensor_overhead() * 5 * n_layers;
    cpu_params.mem_buffer         = NULL;
    cpu_params.no_alloc           = true;
    struct ggml_context * ctx_cpu = ggml_init(cpu_params);
    for (uint32_t il = 0; il < n_layers; ++il) {
        kv_cpu_layers.push_back(
            ggml_new_tensor_4d(ctx_cpu, type, hd_of(il), block_size, 2 * hkv_of(il), n_cpu_blocks));
    }
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    owned_ctxs.emplace_back(ctx_cpu);
    owned_bufs.emplace_back(buf_cpu);
    ggml_backend_buffer_clear(buf_cpu, ds4p_kv_fill("CPU pool (A)"));
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }
        GGML_ASSERT(kv_cpu_layers[il]->buffer && "CPU layer tensor has null buffer");
    }

    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
}

void llama_kv_cache_paged::init(ggml_backend_t backend_gpu,
                                ggml_backend_t backend_cpu,
                                enum ggml_type type,
                                uint32_t       n_gpu_blocks,
                                uint32_t       n_cpu_blocks,
                                float          watermark) {
    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");
    GGML_ASSERT(backend_gpu && "backend_gpu is nullptr");
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend_gpu);
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        LLAMA_LOG_WARN(
            "%s: no GPU device found, allocating KV block pool on CPU. "
            "This is valid for testing but it will be slow.\n",
            __func__);
    }

    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");

    LLAMA_LOG_INFO(
        "%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, watermark=%0.2f\n", __func__,
        n_gpu_blocks, n_cpu_blocks, block_size, watermark);
    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    kv_type        = type;
    gpu_backend    = backend_gpu;
    cpu_backend    = backend_cpu;
    // ⚠ WAS head_dim * ggml_type_size(kv_type). ggml_type_size is bytes per BLOCK, not per
    // element, so for q8_0 (32 elements in a 34-byte block) this multiplied by 34 instead of
    // dividing by 32 -- roughly 34x oversized, with every stride wrong. Harmless while f16 is the
    // only admitted type (1 element, 2 bytes, so the two happen to agree), which is exactly why it
    // survived: the bug is invisible until the feature it breaks is switched on.
    // ggml_row_size handles both cases.
    block_bytes    = 2 * block_size * n_heads_kv * ggml_row_size(kv_type, head_dim);

    // Per-layer bytes, derived from per-layer geometry when present. Everything that copies or
    // serialises a LAYER's block must use bb_of(il), never the pool-wide block_bytes.
    layer_block_bytes.clear();
    if (!layer_head_dim.empty()) {
        layer_block_bytes.resize(n_layers);
        for (uint32_t il = 0; il < n_layers; ++il) {
            layer_block_bytes[il] = 2 * block_size * hkv_of(il) * (uint32_t) ggml_row_size(kv_type, hd_of(il));
        }
    }

    // Set up GPU context and tensor
    // Interleaved shape: [num_blocks, 2, n_heads_kv, block_size, head_dim] (5D)
    struct ggml_init_params gpu_params;
    gpu_params.mem_size   = ggml_tensor_overhead() * 5 * n_layers;
    gpu_params.mem_buffer = NULL;
    gpu_params.no_alloc   = true;

    struct ggml_context * ctx_gpu = ggml_init(gpu_params);

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!ds4p_layer_kv(layer_has_kv, il)) {   // attention-only pool: recurrent layers hold no KV
            kv_gpu_layers.push_back(nullptr);
            continue;
        }
        // Since GGML_MAX_DIMS is set to 4, we flatten the layout to be 4D: [num_blocks, 2 * n_heads_kv, block_size, head_dim]
        ggml_tensor * kv_layer_gpu =
            ggml_new_tensor_4d(ctx_gpu, type, hd_of(il), block_size, 2 * hkv_of(il), n_gpu_blocks);
        kv_gpu_layers.push_back(kv_layer_gpu);
    }

    // Allocate on GPU backend
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    GGML_ASSERT(buf_gpu && "Failed to allocate GPU KV cache buffer");
    owned_ctxs.emplace_back(ctx_gpu);
    owned_bufs.emplace_back(buf_gpu);
    // POISON PROBE (DS4P_KV_POISON): fill with 0xFF so every fp16 is NaN. If nothing ever
    // reads an unwritten block, results are unchanged and the gates still pass -- which is
    // the premise that makes deferring this clear (and thus lazy commit) safe. If something
    // does read one, everything turns NaN and the gate fails LOUDLY instead of silently.
    ggml_backend_buffer_clear(buf_gpu, ds4p_kv_fill("single-device GPU pool"));
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }
        GGML_ASSERT(kv_gpu_layers[il]->buffer && "GPU layer tensor has null buffer");
    }

    // For non CUDA backends, we would split views to allow for standard ggml operators to work out of the box

    // Set up CPU context and tensor (for swapping)
    struct ggml_init_params cpu_params;
    cpu_params.mem_size           = ggml_tensor_overhead() * 5 * n_layers;
    cpu_params.mem_buffer         = NULL;
    cpu_params.no_alloc           = true;
    struct ggml_context * ctx_cpu = ggml_init(cpu_params);
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!ds4p_layer_kv(layer_has_kv, il)) {
            kv_cpu_layers.push_back(nullptr);
            continue;
        }
        ggml_tensor * kv_layer_cpu =
            ggml_new_tensor_4d(ctx_cpu, type, hd_of(il), block_size, 2 * hkv_of(il), n_cpu_blocks);
        kv_cpu_layers.push_back(kv_layer_cpu);
    }

    // Allocate on the CPU backend (using pinned memory for faster PCIe transfer)
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    owned_ctxs.emplace_back(ctx_cpu);
    owned_bufs.emplace_back(buf_cpu);
    ggml_backend_buffer_clear(buf_cpu, ds4p_kv_fill("CPU pool (B)"));
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }
        GGML_ASSERT(kv_cpu_layers[il]->buffer && "CPU layer tensor has null buffer");
    }

    // Setting up our block accountant
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
}



bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    uint32_t curr_block_count     = group.block_table.size();
    uint32_t total_num_tokens     = group.n_prompt + group.n_decoded + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_gpu_blocks(num_requested_blocks)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_requested_blocks, "allocate");
    concat_block_ids(group.block_table, new_ids);
    note_seq_blocks(group);
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}

uint32_t llama_kv_cache_paged::fork_blocks(const llama_sequence_group & src, llama_sequence_group & dst,
                                           uint32_t n_shared_tokens) {
    // only the agreed prefix may be inherited -- the parent's own generated tail diverges
    const uint32_t n_src_tokens = std::min<uint32_t>(n_shared_tokens, (uint32_t) src.logical_seq.size());
    if (n_src_tokens == 0 || src.block_table.empty()) {
        return 0;
    }

    // blocks are inherited from the SHARED span only -- deriving the count from the
    // parent's block_table would hand over the parent's own generated tokens too
    const uint32_t tail_fill     = n_src_tokens % block_size;
    uint32_t       n_full_blocks = n_src_tokens / block_size;
    if (n_full_blocks > (uint32_t) src.block_table.size()) {
        n_full_blocks = (uint32_t) src.block_table.size();
    }

    dst.block_table.clear();
    dst.block_table.insert(dst.block_table.end(), src.block_table.begin(),
                           src.block_table.begin() + n_full_blocks);
    block_manager.share_blocks(dst.block_table);
    note_seq_blocks(dst);   // a fork's child owns (shares) the inherited blocks

    uint32_t n_inherited = n_full_blocks * block_size;

    // NOTE: the partially-filled tail block is NOT inherited. Copying it across the
    // gpu/cpu id spaces is fiddly (and was a real out-of-bounds bug), while the cost of
    // NOT inheriting it is at most block_size-1 re-prefilled tokens -- nothing against a
    // 100K prefix. Deleted rather than fixed.

    dst.logical_seq.assign(src.logical_seq.begin(), src.logical_seq.begin() + std::min<size_t>(n_inherited, src.logical_seq.size()));
    dst.n_past   = n_inherited;
    dst.n_prompt = n_inherited;

    LLAMA_LOG_INFO("%s: forked request %d -> %d: %u blocks shared by reference, %u tokens inherited "
                   "(partial tail of %u tokens re-prefilled)\n",
                   __func__, src.request_id, dst.request_id, n_full_blocks, n_inherited, tail_fill);

    return n_inherited;
}

// P1-5 prerequisite: record (or clear) a sequence's physical block residency. Called
// wherever the cache already observes a group's request_id and block_table together, so
// the map tracks reality across preemption, swap and fork instead of being reconstructed.
void llama_kv_cache_paged::note_seq_blocks(const llama_sequence_group & group) {
    if (group.block_table.empty()) {
        sequence_blocks.erase(group.request_id);
    } else {
        sequence_blocks[group.request_id] = group.block_table;
    }
}

void llama_kv_cache_paged::free_blocks(llama_sequence_group & group) {
    if (group.block_table.empty()) {
        return;
    }

    // ★ DS4P_SYNC_ON_FREE -- test for the #1(c) race.
    //
    // Blocks are returned to the pool the moment a request finishes, but nothing waits for the GPU
    // work referencing them to RETIRE. Checkout is LIFO, so the block just freed is literally the
    // next one handed out: the incoming request starts writing its KV into a block while the
    // outgoing request's in-flight writes are still landing in it.
    //
    // This is the only shape left. The kernel receives BYTE-IDENTICAL arguments on a corrupt rep and
    // a clean one (4201 ARGDUMP lines, no diff), so the host bookkeeping is correct and the
    // divergence is in the DATA. A cold server has never freed a block, which is why it is 0/24.
    // GGML_METAL_CONCURRENCY_DISABLE did not help because it serialises ops WITHIN a graph, not
    // across command buffers belonging to different requests; FIFO did not help because delaying
    // reuse is not the same as WAITING for completion.
    // ⚠ The marker is NOT decoration. A null backend handle would make both calls no-ops, and then
    // "sync changed nothing" would mean "the sync never happened" -- an absence read as a result.
    if (getenv("DS4P_SYNC_ON_FREE") != nullptr) {
        LLAMA_LOG_ERROR("DS4P-SYNC-ON-FREE gpu=%d cpu=%d\n", gpu_backend != nullptr, cpu_backend != nullptr);
        if (gpu_backend) { ggml_backend_synchronize(gpu_backend); }
        if (cpu_backend) { ggml_backend_synchronize(cpu_backend); }
    }

    // ⚠ request_id IS the slot id, and the server REUSES slot ids. A group that finishes AFTER a new
    // request has already taken the same id must NOT wipe the new request's mapping. The scheduler
    // guards the identical operation (llama-paged-scheduler-impl.cpp:250, "a new request may already
    // have reused this id ... erasing its entry orphans the new request"); this path did not.
    //
    // That gap is the residual -np>1 corruption: the live request loses its KV mapping mid-flight,
    // so attention sees almost no keys and the model repeats the last token of its own prompt.
    // Measured with slot_reuse_probe.sh -- WARM (a request finished on the slot first) 4/6 corrupt,
    // COLD (no prior request) 0/6, same tool, same binary, one factor.
    //
    // ★ Ownership is sampled HERE, before any block is released, and that placement is load-bearing.
    // It cannot be checked after the release/clear below: by then these blocks are back in the pool
    // and a new request may hold the very same ids in the very same order, so a post-hoc comparison
    // can match by coincidence and re-arm the bug it was added to fix.
    const auto it_own     = sequence_blocks.find(group.request_id);
    const bool still_ours = (it_own != sequence_blocks.end() && it_own->second == group.block_table);

    llama_block_ids blocks_to_free_gpu;
    llama_block_ids blocks_to_free_cpu;

    for (uint32_t block_id : group.block_table) {
        if (block_manager.is_gpu(block_id)) {
            blocks_to_free_gpu.push_back(block_id);
        } else {
            blocks_to_free_cpu.push_back(block_id);
        }
    }

    if (!blocks_to_free_gpu.empty()) {
        block_manager.release_gpu_blocks(blocks_to_free_gpu);
    }
    if (!blocks_to_free_cpu.empty()) {
        block_manager.release_cpu_blocks(blocks_to_free_cpu);
    }

    group.block_table.clear();

    // The blocks above are released UNCONDITIONALLY -- a stale group's blocks must return to the
    // pool regardless of who owns the id now, or the pool leaks on every reused slot. Only the
    // MAPPING is conditional: it belongs to whoever holds the id today.
    if (still_ours) {
        seq_rm(group.request_id, llama_pos{}, llama_pos{});
        sequence_blocks.erase(group.request_id);
    }
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");

    // ★ PRESENCE MARKER at INFO, not DEBUG. A gate that starves GPU blocks to force eviction and
    // then compares outputs would pass PERFECTLY if no eviction ever happened -- the run would just
    // be an ordinary run. "Outputs match" is only evidence when the path under test actually
    // executed, and this is the only place that can say so. The DEBUG line above cannot: it is off
    // at the verbosity the gates use, which is how a path stays untested while looking covered.
    {
        static int n_swaps = 0;
        LLAMA_LOG_INFO("%s: DS4P-EVICT swap #%d %s: %u blocks\n", __func__,
                       ++n_swaps, to_gpu ? "CPU->GPU" : "GPU->CPU", num_blocks);
    }

    const auto & src_layers = to_gpu ? kv_cpu_layers : kv_gpu_layers;
    const auto & dst_layers = to_gpu ? kv_gpu_layers : kv_cpu_layers;

    GGML_ASSERT(src_layers.size() == n_layers && "src layer count mismatch.");
    GGML_ASSERT(dst_layers.size() == n_layers && "src layer count mismatch.");

    // Buffer on HOST to faciliate block data transfer
    // Note: an optimization would be to use views and async copies. Beware of
    // memory overhead heurisitcs.
    // ⚠ SIZED PER LAYER, NOT ONCE. This was a single buffer of the pool-wide block_bytes, reused
    // across every layer. With per-layer head geometry a 256-wide layer's block is half a 512-wide
    // one, so one shared size either truncates a copy or reads past the source -- on the GPU<->CPU
    // eviction path, which runs rarely, only under memory pressure, and is not exercised by any
    // gate in this lane. Silent corruption under load is the worst thing to ship, so the buffer is
    // resized inside the loop and every offset uses bb_of(il).
    std::vector<uint8_t> staging;

    for (uint32_t il = 0; il < n_layers; ++il) {
        // ★ ATTENTION-ONLY POOL: a filtered (recurrent) layer owns no tensor on either side.
        // Without this guard the swap path dereferences nullptr -- the hazard that made this
        // change worth doing carefully rather than only touching the allocation loops.
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }

        const size_t lbb = bb_of(il);
        if (staging.size() < lbb) { staging.resize(lbb); }

        struct ggml_tensor * src_main = src_layers[il];
        struct ggml_tensor * dst_main = dst_layers[il];

        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];

            // GPU and CPu blocks may differ (usually CPU < GPU)
            // We substract the diffence to calculate where the local starts before we calculate offsets
            const uint32_t src_local = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            const size_t src_offset = (size_t) src_local * lbb;
            const size_t dst_offset = (size_t) dst_local * lbb;

            // Put src tensor into HOST staging buffer
            ggml_backend_tensor_get(src_main, staging.data(), src_offset, lbb);
            // Put tensor from HOST staging into dst tensor
            ggml_backend_tensor_set(dst_main, staging.data(), dst_offset, lbb);
        }
    }
}

bool llama_kv_cache_paged::swap_in(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    // A potential optimization to reduce thrashing is to have a heuristic to check if
    // if we can continue decoding after swap_in.
    if (!block_manager.has_free_gpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_blocks, "swap_in");
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/true);

    free_blocks(group);
    group.block_table = new_ids;
    note_seq_blocks(group);
    return true;
}

bool llama_kv_cache_paged::swap_out(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_cpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_cpu_blocks(num_blocks);
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/false);

    free_blocks(group);
    group.block_table = new_ids;
    note_seq_blocks(group);
    return true;
}

void llama_kv_cache_paged::set_paged_batch_info(const llama_paged_batch_info * info) {
    last_paged_info = info;
}

uint32_t llama_kv_cache_paged::get_num_gpu_blocks() const {
    return num_gpu_blocks;
}

void llama_kv_cache_paged::concat_block_ids(llama_block_ids &       to_block_table,
                                            const llama_block_ids & from_block_table) {
    to_block_table.insert(to_block_table.end(), from_block_table.begin(), from_block_table.end());
}

// llama_memory_i

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool /*embd_all*/) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        // Failed to find a suitable split
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        // Do not use balloc's internal batch. It does not carry any paged metadata.
        // No batch info = nothing scheduled this decode (e.g. load-time probe decodes
        // before a scheduler exists). Fail the prepare gracefully instead of aborting --
        // callers treat it as decode-failure, and the 4d server loop always sets info.
        if (last_paged_info == nullptr) {
            LLAMA_LOG_WARN("%s: no paged batch info set (no scheduler drove this decode) -> failed prepare\n", __func__);
            break;
        }

        auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));
        ctx->set_batch_data(*last_paged_info);
        return ctx;
    } while (false);

    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch_with_ubatches(std::vector<llama_ubatch> ubatches) {
    if (ubatches.empty()) {
        return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    if (last_paged_info == nullptr) {
        LLAMA_LOG_WARN("%s: no paged batch info set -> failed prepare\n", __func__);
        return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));
    ctx->set_batch_data(*last_paged_info);
    return ctx;
}

// Used by llama_context scheduler to dry-run
llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    LLAMA_LOG_DEBUG("%s: reserving graph for n_ubatch=%d, n_seq_max=%d, num_gpu_blocks=%d\n", __func__, n_ubatch,
                    n_seq_max, num_gpu_blocks);

    // Create a "dummy" ubatch that represents the maximum capacity
    // of the system to let the scheduler reserve enough space for metadata.
    llama_ubatch ubatch = {};
    ubatch.n_tokens     = n_ubatch;   // maximum tokens
    ubatch.n_seqs       = n_seq_max;  // maximum sequences
    ubatch.n_pos        = 1;

    std::vector<llama_ubatch> ubatches = { ubatch };

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);

    ctx->set_batch_size(n_seq_max);       // maximum possible sequences
    ctx->set_n_tokens(n_ubatch);          // representative token count
    ctx->set_max_blocks(num_gpu_blocks);  // every block could theoretically belong to one seq

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    std::vector<llama_ubatch> dummy_ubatch = {};
    auto                      ctx          = std::make_unique<llama_kv_cache_paged_context>(this, dummy_ubatch);
    // TODO maybe confirm block counts or clean up stale pointers
    return ctx;
}

struct ggml_tensor * llama_kv_cache_paged::get_kv_tensor(int layer_idx) const {
    return kv_gpu_layers[layer_idx];
}

int32_t llama_kv_cache_paged::debug_seq_kv_checksum(const llama_sequence_group & group, int32_t n_tokens,
                                                    double * out_sums, int32_t max_layers) const {
    const int32_t nl = std::min<int32_t>((int32_t) n_layers, max_layers);
    std::vector<uint8_t> buf;
    for (int32_t il = 0; il < nl; ++il) {
        ggml_tensor * kv = kv_gpu_layers[il];
        if (kv == nullptr) {
            out_sums[il] = -1.0;
            continue;
        }
        const size_t nb_token = kv->nb[1];
        const size_t nb_head  = kv->nb[2];
        const size_t nb_block = kv->nb[3];
        const size_t row_sz   = (size_t) head_dim * ggml_type_size(kv->type);  // one head's dims for one token
        buf.resize(row_sz);
        double sum = 0.0;
        for (int32_t t = 0; t < n_tokens; ++t) {
            const size_t bt = (size_t) t / block_size;
            if (bt >= group.block_table.size()) {
                break;  // beyond the group's allocated span
            }
            const size_t off_base = (size_t) group.block_table[bt] * nb_block + (size_t) (t % block_size) * nb_token;
            // ★ DS4P_KVSUM_ROWDUMP="lo-hi": raw first-8-halves of the K row for tokens in [lo,hi],
            // layers 0-2 -- POST-EXECUTION pool truth through the same addressing as the checksum.
            // Encode-time tensor_get in the op handler lies on Metal (graph encodes before it
            // runs); this is the valid placement. Byte PATTERN tells the story: garbage vs
            // an identifiable other-token's row.
            if (il < 3) {
                static int rd_lo = -1, rd_hi = -2;
                static bool rd_init = false;
                if (!rd_init) {
                    rd_init = true;
                    if (const char * e = getenv("DS4P_KVSUM_ROWDUMP")) { sscanf(e, "%d-%d", &rd_lo, &rd_hi); }
                }
                if (t >= rd_lo && t <= rd_hi) {
                    // Per head-plane (K then V) x per row segment: a full-row bit-sum plus the
                    // first 4 halves. The first-8-halves-of-K version printed IDENTICAL rows
                    // while the full checksum differed -- the divergence hides deeper in the row
                    // or in the V plane, so cover the WHOLE row this time.
                    for (uint32_t hp = 0; hp < 2 * n_heads_kv; ++hp) {
                        std::vector<uint16_t> rr(row_sz / 2);
                        ggml_backend_tensor_get(kv, rr.data(), off_base + (size_t) hp * nb_head, row_sz);
                        uint64_t bitsum = 0;
                        for (uint16_t x : rr) { bitsum += x; }
                        fprintf(stderr, "DS4P-ROWDUMP L%d t=%d hp=%u blk=%d off=%d bitsum=%llu head4: %04x %04x %04x %04x\n",
                                il, t, hp, (int) group.block_table[bt], (int) (t % block_size),
                                (unsigned long long) bitsum, rr[0], rr[1], rr[2], rr[3]);
                    }
                }
            }
            for (uint32_t h = 0; h < 2 * n_heads_kv; ++h) {  // K heads then V heads (interleaved layout)
                ggml_backend_tensor_get(kv, buf.data(), off_base + (size_t) h * nb_head, row_sz);
                const uint16_t * half_vals = (const uint16_t *) buf.data();
                for (size_t i = 0; i < row_sz / 2; ++i) {
                    sum += (double) half_vals[i];  // raw-bit additive sum: bit-equality detector, not a norm
                }
            }
        }
        out_sums[il] = sum;
    }
    return nl;
}

void llama_kv_cache_paged::clear(bool /*data*/) {
    sequence_positions.clear();

    while (!restored_seqs.empty()) {
        discard_restored(restored_seqs.begin()->first);
    }
}

// P1-5: serialise ONE sequence's KV out of the paged cache.
//
// This is now possible because sequence_blocks records which physical blocks the
// sequence owns; before that map existed the cache could not enumerate them, which is
// why this body was `{}` and why llama_state_seq_get_size_ext returned 0 (so the disk
// KV bank could never spill a byte).
//
// Format: a small header, then for every layer, every block's raw bytes in the
// sequence's block-table ORDER (not physical id order) -- so a restore does not need the
// original physical ids to be free, only the same count. Blocks may live on GPU or CPU;
// the id space is global with CPU ids >= num_gpu_blocks, which is why the local index is
// recomputed per block exactly as do_block_copy does.
void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags) const {
    const auto it = sequence_blocks.find(seq_id);
    if (seq_id < 0 || it == sequence_blocks.end() || it->second.empty()) {
        return;   // nothing resident for this sequence: write nothing, size stays 0
    }

    const llama_block_ids & blocks = it->second;

    const uint32_t n_blocks = (uint32_t) blocks.size();
    io.write(&n_blocks,    sizeof(n_blocks));
    io.write(&block_size,  sizeof(block_size));
    io.write(&head_dim,    sizeof(head_dim));
    io.write(&n_heads_kv,  sizeof(n_heads_kv));
    io.write(&n_layers,    sizeof(n_layers));
    io.write(&block_bytes, sizeof(block_bytes));

    auto pos = sequence_positions.find(seq_id);
    const llama_pos p_min = pos != sequence_positions.end() ? pos->second.min : -1;
    const llama_pos p_max = pos != sequence_positions.end() ? pos->second.max : -1;
    io.write(&p_min, sizeof(p_min));
    io.write(&p_max, sizeof(p_max));

    // COALESCED. A freshly checked-out block table is mostly CONSECUTIVE ids, so copying
    // block-at-a-time turns one large transfer into n_blocks * n_layers small ones -- 5,688
    // separate 64 KiB device copies for a 2K prompt, measured at 1.7 GB/s where the link
    // does far better in bulk. Runs of consecutive ids on the same device move in one call.
    const auto runs = contiguous_runs(blocks);

    std::vector<uint8_t> staging;

    for (uint32_t il = 0; il < n_layers; ++il) {
        // ⚠ FILTERED LAYERS OWN NO TENSOR. This loop had no guard, so with an attention-only pool
        // active it would dereference nullptr here -- the same hazard do_block_copy was fixed for,
        // missed in the serdes because nothing in this lane saves paged state yet. Absence of a
        // caller is not absence of a bug; it just moves the crash to whoever adds one.
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }

        const size_t lbb = bb_of(il);   // per-layer, never the pool-wide block_bytes

        for (const auto & run : runs) {
            const bool     gpu   = block_manager.is_gpu(run.first);
            const uint32_t local = gpu ? run.first : run.first - num_gpu_blocks;
            const size_t   bytes = (size_t) run.second * lbb;

            struct ggml_tensor * layer = gpu ? kv_gpu_layers[il] : kv_cpu_layers[il];

            if (staging.size() < bytes) {
                staging.resize(bytes);
            }
            ggml_backend_tensor_get(layer, staging.data(), (size_t) local * lbb, bytes);
            io.write(staging.data(), bytes);
        }
    }
}

// Split a block table into maximal runs of consecutive ids that live on the same device,
// as (first_id, count) pairs. The order of bytes on the wire is unchanged -- a run is just
// a batch of the same per-block copies -- so a state written by one build restores on
// another regardless of how the pool happened to fragment.
std::vector<std::pair<uint32_t, uint32_t>> llama_kv_cache_paged::contiguous_runs(const llama_block_ids & blocks) const {
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    if (blocks.empty()) {
        return runs;
    }

    uint32_t first = blocks[0];
    uint32_t count = 1;

    for (size_t i = 1; i < blocks.size(); ++i) {
        const bool consecutive = blocks[i] == blocks[i - 1] + 1;
        const bool same_device = block_manager.is_gpu(blocks[i]) == block_manager.is_gpu(first);

        if (consecutive && same_device) {
            count++;
        } else {
            runs.emplace_back(first, count);
            first = blocks[i];
            count = 1;
        }
    }
    runs.emplace_back(first, count);

    return runs;
}

// P1-5 RESTORE. The mirror image of state_write: check the geometry, take real blocks,
// fill them with the stored bytes, and PARK them under seq_id.
//
// Why parking instead of installing: this runs from the server's prompt-cache admit
// (server_prompt_cache::load -> llama_state_seq_set_data_ext), which happens BEFORE the
// request is queued -- there is no scheduler group yet, so there is no block_table to
// write into. The cache therefore materialises the KV and holds it for the short window
// until llama_paged_scheduler_impl::queue_request adopts it via take_restored_blocks.
// That is the same handover a COW fork already does, with the disk bank standing in for
// the live parent, which is why this reuses the proven path instead of inventing one.
//
// Every failure below THROWS rather than returning a partial restore. llama_context::
// state_seq_set_data catches it and returns 0, the server's prompt_load treats that as a
// failed restore and recomputes. Loud, not fatal, never silently wrong -- the earlier
// GGML_ABORT here was the right principle with the wrong mechanism (it killed the server
// on the first admit, and an admit always comes eventually).
void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags) {
    if (seq_id < 0) {
        throw std::runtime_error("paged KV state_read: needs a concrete sequence id");
    }

    // an earlier admit for this slot that was never queued would otherwise pin its blocks
    discard_restored(seq_id);

    uint32_t r_n_blocks = 0, r_block_size = 0, r_head_dim = 0, r_heads_kv = 0, r_layers = 0, r_block_bytes = 0;
    io.read(&r_n_blocks,    sizeof(r_n_blocks));
    io.read(&r_block_size,  sizeof(r_block_size));
    io.read(&r_head_dim,    sizeof(r_head_dim));
    io.read(&r_heads_kv,    sizeof(r_heads_kv));
    io.read(&r_layers,      sizeof(r_layers));
    io.read(&r_block_bytes, sizeof(r_block_bytes));

    llama_pos p_min = -1;
    llama_pos p_max = -1;
    io.read(&p_min, sizeof(p_min));
    io.read(&p_max, sizeof(p_max));

    // GEOMETRY IS IDENTITY. A bank file written under a different model, KV quant, head
    // count or block size is still a pile of bytes that would load without complaint and
    // then serve another model's attention. P0-2's revalidate checks the TOKEN claim; this
    // checks the PHYSICAL layout, and neither one substitutes for the other.
    if (r_block_size != block_size || r_head_dim != head_dim || r_heads_kv != n_heads_kv ||
        r_layers != n_layers || r_block_bytes != block_bytes) {
        throw std::runtime_error(
            "paged KV state_read: geometry mismatch (stored block_size/head_dim/n_heads_kv/"
            "n_layers/block_bytes = " + std::to_string(r_block_size) + "/" + std::to_string(r_head_dim) +
            "/" + std::to_string(r_heads_kv) + "/" + std::to_string(r_layers) + "/" + std::to_string(r_block_bytes) +
            ", cache = " + std::to_string(block_size) + "/" + std::to_string(head_dim) + "/" +
            std::to_string(n_heads_kv) + "/" + std::to_string(n_layers) + "/" + std::to_string(block_bytes) +
            ") -- refusing to reinterpret it, falling back to recompute");
    }

    if (r_n_blocks == 0 || p_max < 0) {
        return;   // a well-formed entry that holds nothing: no restore, no error
    }

    // A restore competes for the same scarce pool as live requests. If it does not fit,
    // the request must prefill normally -- never evict someone else to make room for a
    // cache hit, which would turn a latency win into a latency loss for another agent.
    if (!block_manager.has_free_gpu_blocks(r_n_blocks)) {
        throw std::runtime_error("paged KV state_read: only " + std::to_string(r_n_blocks) +
                                 " blocks would fit the restore and the pool cannot spare them"
                                 " -- falling back to recompute");
    }

    llama_block_ids ids = block_manager.checkout_gpu_blocks(r_n_blocks, "state_read");

    // COALESCED, same reason as state_write: measured 204 ms to upload 355 MiB as 5,688
    // separate 64 KiB copies (~1.7 GB/s), which is most of why a warm admit lost to a cold
    // prefill at 2K. Consecutive ids on the same device go up in one call.
    const auto runs = contiguous_runs(ids);

    std::vector<uint8_t> staging;

    for (uint32_t il = 0; il < n_layers; ++il) {
        // Mirrors state_write exactly -- filtered layers own no tensor, and the two loops must
        // skip the SAME layers or the byte stream desynchronises silently on restore.
        if (!ds4p_layer_kv(layer_has_kv, il)) { continue; }

        const size_t lbb = bb_of(il);

        for (const auto & run : runs) {
            const bool     gpu   = block_manager.is_gpu(run.first);
            const uint32_t local = gpu ? run.first : run.first - num_gpu_blocks;
            const size_t   bytes = (size_t) run.second * lbb;

            struct ggml_tensor * layer = gpu ? kv_gpu_layers[il] : kv_cpu_layers[il];

            if (staging.size() < bytes) {
                staging.resize(bytes);
            }
            io.read(staging.data(), bytes);
            ggml_backend_tensor_set(layer, staging.data(), (size_t) local * lbb, bytes);
        }
    }

    // ★ SYNCHRONIZE. ggml_backend_tensor_set is ASYNCHRONOUS on CUDA, so without this the
    // function returns once the copies are ENQUEUED, not once the KV is resident. Two
    // consequences, one measurable and one latent:
    //   - the caller's admit timer stopped early, so the economics gate believed a restore
    //     cost ~208 ms when the wall clock put it near 615 ms, and it therefore admitted
    //     restores it should have declined. I was timing the enqueue and calling it the
    //     transfer -- the same class as trusting a log line that prints before the work.
    //   - correctness rests on the copies landing before the first decode reads them, which
    //     is true only while they share a stream. Making it explicit costs nothing here
    //     (this path already moved hundreds of MiB) and removes the assumption.
    if (gpu_backend != nullptr) {
        ggml_backend_synchronize(gpu_backend);
    }
    if (cpu_backend != nullptr) {
        ggml_backend_synchronize(cpu_backend);
    }

    sequence_positions[seq_id]  = seq_range{ p_min, p_max };
    restored_seqs[seq_id]       = restored_seq{ std::move(ids), p_min, p_max };

    LLAMA_LOG_INFO("%s: seq %d: restored %u blocks (%u tokens) from state -- awaiting adoption\n",
                   __func__, seq_id, r_n_blocks, (uint32_t) (p_max + 1));
}

uint32_t llama_kv_cache_paged::take_restored_blocks(llama_seq_id seq_id, uint32_t n_tokens_wanted,
                                                   llama_block_ids & out_blocks) {
    auto it = restored_seqs.find(seq_id);
    if (it == restored_seqs.end()) {
        return 0;
    }

    const uint32_t n_have = it->second.p_max >= 0 ? (uint32_t) (it->second.p_max + 1) : 0;
    const uint32_t n_take = std::min(n_have, n_tokens_wanted);

    if (n_take == 0) {
        discard_restored(seq_id);
        return 0;
    }

    llama_block_ids & blocks = it->second.blocks;

    // The caller's prompt may diverge from the stored one before the end of the stored KV.
    // Blocks past the agreed span hold tokens this request will never ask for, so they go
    // straight back to the pool rather than riding along in the block table -- allocate()
    // derives its request count from block_table.size(), so an over-long table would make
    // it compute a NEGATIVE need in unsigned arithmetic and refuse every future block.
    const uint32_t n_keep = (n_take + block_size - 1) / block_size;
    if (n_keep < blocks.size()) {
        llama_block_ids tail(blocks.begin() + n_keep, blocks.end());
        llama_block_ids tail_gpu;
        llama_block_ids tail_cpu;
        for (uint32_t b : tail) {
            (block_manager.is_gpu(b) ? tail_gpu : tail_cpu).push_back(b);
        }
        if (!tail_gpu.empty()) {
            block_manager.release_gpu_blocks(tail_gpu);
        }
        if (!tail_cpu.empty()) {
            block_manager.release_cpu_blocks(tail_cpu);
        }
        blocks.resize(n_keep);
    }

    out_blocks = blocks;

    sequence_blocks[seq_id]    = blocks;
    sequence_positions[seq_id] = seq_range{ 0, (llama_pos) (n_take - 1) };

    restored_seqs.erase(it);

    return n_take;
}

void llama_kv_cache_paged::discard_restored(llama_seq_id seq_id) {
    auto it = restored_seqs.find(seq_id);
    if (it == restored_seqs.end()) {
        return;
    }

    llama_block_ids gpu;
    llama_block_ids cpu;
    for (uint32_t b : it->second.blocks) {
        (block_manager.is_gpu(b) ? gpu : cpu).push_back(b);
    }
    if (!gpu.empty()) {
        block_manager.release_gpu_blocks(gpu);
    }
    if (!cpu.empty()) {
        block_manager.release_cpu_blocks(cpu);
    }

    LLAMA_LOG_DEBUG("%s: seq %d: released %zu unadopted restored blocks\n",
                    __func__, seq_id, it->second.blocks.size());

    restored_seqs.erase(it);
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos /*p0*/, llama_pos /*p1*/) {
    sequence_positions.erase(seq_id);
    sequence_blocks.erase(seq_id);
    // a parked restore for a sequence being torn down is dead weight holding real blocks
    discard_restored(seq_id);
    return true;
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;
    const size_t                                 n_gpu_kvs = kv_gpu_layers.size();
    const size_t                                 n_cpu_kvs = kv_cpu_layers.size();

    for (size_t il = 0; il < n_layers; ++il) {
        auto * kv_gpu = (il < n_gpu_kvs) ? kv_gpu_layers[il] : nullptr;
        if (kv_gpu) {
            breakdown[ggml_backend_buffer_get_type(kv_gpu->buffer)] = ggml_nbytes(kv_gpu);
        }
        auto * kv_cpu = (il < n_cpu_kvs) ? kv_cpu_layers[il] : nullptr;
        if (kv_cpu) {
            breakdown[ggml_backend_buffer_get_type(kv_cpu->buffer)] = ggml_nbytes(kv_cpu);
        }
    }
    return breakdown;
}

void llama_kv_cache_paged::set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min) {
    sequence_positions[seq_id].min = new_min;
}

void llama_kv_cache_paged::set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max) {
    sequence_positions[seq_id].max = new_max;
}

// llama_kv_cache_paged_context

void llama_kv_cache_paged_context::set_batch_data(const llama_paged_batch_info & info) {
    // ⚠ COPY, DO NOT ALIAS. llama.h says these arrays "are owned by the scheduler and must remain
    // valid until the next scheduler step clears them" -- so holding raw pointers means the graph's
    // set_input() can read arrays the scheduler has already reused or freed. That produces
    // INTERMITTENT corruption: correct whenever set_input happens to run before the next step,
    // garbage when it does not, which is exactly the 1-of-3 pattern multislot was showing after the
    // offsets were re-based correctly. Own the data for as long as this context exists.
    const int32_t ns = info.n_seq;
    const int32_t nt = info.n_tokens;
    const int32_t nb = info.n_blocks_per_seq;
    auto keep = [](std::vector<int32_t> & dst, const int32_t * src, size_t n) {
        if (src && n) { dst.assign(src, src + n); } else { dst.clear(); }
    };
    keep(own_write_slots,   info.write_slots,   (size_t) nt);
    keep(own_seq_ids,       info.seq_ids,       (size_t) ns);
    keep(own_block_table,   info.block_table,   (size_t) ns * (size_t) nb);
    keep(own_context_lens,  info.context_lens,  (size_t) ns);
    keep(own_batch_offsets, info.batch_offsets, (size_t) ns);
    keep(own_batch_lens,    info.batch_lens,    (size_t) ns);

    paged_write_slots   = own_write_slots.empty()   ? nullptr : own_write_slots.data();
    paged_seq_ids       = own_seq_ids.empty()       ? nullptr : own_seq_ids.data();
    paged_n_seq         = info.n_seq;
    paged_block_table   = own_block_table.empty()   ? nullptr : own_block_table.data();
    paged_context_lens  = own_context_lens.empty()  ? nullptr : own_context_lens.data();
    paged_batch_offsets = own_batch_offsets.empty() ? nullptr : own_batch_offsets.data();
    paged_batch_lens    = own_batch_lens.empty()    ? nullptr : own_batch_lens.data();
    n_tokens            = info.n_tokens;
    max_blocks          = info.n_blocks_per_seq;
    batch_size          = info.n_seq;
}

bool llama_kv_cache_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    if (++i_cur >= ubatches.size()) {
        return false;
    }
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // Nothing to do for paged KV cache, return true to allow for execution
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_cur];
}

struct ggml_tensor * llama_kv_cache_paged_context::get_k(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

struct ggml_tensor * llama_kv_cache_paged_context::get_v(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_kv_tensor(layer_idx);
}

int32_t llama_kv_cache_paged_context::get_n_tokens() const {
    return n_tokens;
}

int32_t llama_kv_cache_paged_context::get_batch_size() const {
    return batch_size;
}

int32_t llama_kv_cache_paged_context::get_max_blocks() const {
    return max_blocks;
}

int32_t * llama_kv_cache_paged_context::get_seq_ids() const {
    return paged_seq_ids;
}

int32_t llama_kv_cache_paged_context::get_n_seq() const {
    return paged_n_seq;
}

int32_t * llama_kv_cache_paged_context::get_write_slots() const {
    return paged_write_slots;
}

int32_t * llama_kv_cache_paged_context::get_block_table() const {
    return paged_block_table;
}

int32_t * llama_kv_cache_paged_context::get_context_lens() const {
    return paged_context_lens;
}

int32_t * llama_kv_cache_paged_context::get_batch_offsets() const {
    return paged_batch_offsets;
}

int32_t * llama_kv_cache_paged_context::get_batch_lens() const {
    return paged_batch_lens;
}

void llama_kv_cache_paged_context::set_n_tokens(int32_t new_n_tokens) {
    n_tokens = new_n_tokens;
}

void llama_kv_cache_paged_context::set_batch_size(int32_t new_batch_size) {
    batch_size = new_batch_size;
}

void llama_kv_cache_paged_context::set_max_blocks(int32_t new_max_blocks) {
    max_blocks = new_max_blocks;
}

// ★ SELF-DRIVE -- see the header for why this exists and how narrow it is.
bool llama_kv_cache_paged::self_drive_enabled() const {
    static const bool en = [] {
        const char * e = getenv("DS4P_PAGED_DRIVE");
        return e && atoi(e) != 0;
    }();
    return en;
}

// Release ONLY the batch info. The group's blocks must survive a decode step.
void llama_kv_cache_paged::self_drive_release_info() {
    set_paged_batch_info(nullptr);

    delete[] sd_info.write_slots;
    delete[] sd_info.block_table;
    delete[] sd_info.context_lens;
    delete[] sd_info.batch_offsets;
    delete[] sd_info.batch_lens;
    delete[] sd_info.prefill_pending;
    delete[] sd_info.seq_ids;
    sd_info = {};
}

void llama_kv_cache_paged::self_drive_end() {
    if (!sd_active) {
        return;
    }

    set_paged_batch_info(nullptr);

    delete[] sd_info.write_slots;
    delete[] sd_info.block_table;
    delete[] sd_info.context_lens;
    delete[] sd_info.batch_offsets;
    delete[] sd_info.batch_lens;
    delete[] sd_info.prefill_pending;
    delete[] sd_info.seq_ids;
    sd_info = {};

    free_blocks(sd_group);
    sd_group = {};
    sd_active = false;
}

bool llama_kv_cache_paged::self_drive_begin(int32_t n_tokens) {
    if (n_tokens <= 0) {
        return false;
    }

    // ★ PERSISTENT SEQUENCE GROUP. The first version rebuilt the group from scratch on every call
    // with n_past = 0, so each decode step saw only its own token and generation degenerated
    // (`one, two, three.` then drivel) -- caught by the hybrid decode gate, which is why that gate
    // exists.
    //
    // n_tokens > 1  => a new prompt: release the old sequence and start fresh.
    // n_tokens == 1 => a decode step: KEEP the group, grow it, and write at the next position.
    // ⚠ A genuine ONE-token prompt is indistinguishable from a decode step here. Acceptable for a
    // dev-gated single-sequence bridge; a real scheduler carries the request identity instead.
    const bool have_live  = sd_active && !sd_group.block_table.empty();
    const bool is_prefill = (n_tokens > 1) || !have_live;

    // Release only the INFO between steps; the group's blocks must survive a decode step or the
    // prefix is lost. self_drive_end() frees blocks too, so it is only correct at a real boundary.
    self_drive_release_info();

    if (is_prefill) {
        // ⚠ FREE ON A NON-EMPTY TABLE, NOT ON have_live. have_live is
        //     sd_active && !sd_group.block_table.empty()
        // so whenever sd_active is false while the table still holds ids, this skipped the free and
        // the `sd_group = {}` below STRANDED those blocks -- checked out, unreachable, never
        // returned. Measured ledger at -ngpub 8: two checkouts then a single RELEASE n=1 at the
        // prefill boundary, and by the refusal 8 blocks were out, 1 returned, 3 in the table, FOUR
        // stranded. The pool was never too small; it had been quietly drained of half its blocks.
        //
        // sd_active answers "is a self-drive batch in flight". The table answers "do I hold blocks".
        // Only the second one is relevant to whether they must be returned, and conflating them made
        // the leak conditional on an unrelated flag.
        if (!sd_group.block_table.empty()) {
            free_blocks(sd_group);
        }
        sd_group = {};
        sd_group.request_id = 0;
        sd_group.n_prompt   = (uint32_t) n_tokens;
        sd_group.n_decoded  = 0;
    }

    // Position this batch's first token: everything already written for this sequence.
    const int32_t n_past = is_prefill ? 0 : (int32_t) (sd_group.n_prompt + sd_group.n_decoded);

    if (!is_prefill) {
        sd_group.n_decoded += (uint32_t) n_tokens;
    }

    // allocate() tops the block table up to cover n_prompt + n_decoded (+ the extra it is asked
    // for). Counts are already final here, so ask for 0 and let it size from the group.
    if (!allocate(0, sd_group)) {
        // ★ THE FALLBACK IS THE BUG, NOT THE ALLOCATION FAILURE -- and only mid-sequence.
        //
        // At n_past == 0 nothing has been written yet, so handing the request to the static path is
        // a genuine graceful degrade. At n_past > 0 this sequence's KV lives in the POOL, and the
        // lines below FREE IT and then let the static path continue with a cache that never saw
        // those tokens. Measured on Ornith-9B at -ngpub 8: this fires once at n_past=48 and the
        // output turns to garbage at exactly token 48 ("...seventeen, eighteen, ten, ten."), HTTP
        // 200, no error. -ngpub 10 never fires it and is byte-identical to a large pool.
        //
        // Same line, safe in one context and silently destructive in the other, previously logged
        // at WARN with the words "static path" -- which reads as a routing decision rather than
        // data loss. The producer was correct the whole time and nothing consumed it.
        // ⚠ MEASURE THE PREMISE, DO NOT DESIGN AROUND IT. "the pool is too small" is an assumption:
        // at -ngpub 8 there are 7 usable blocks and this sequence needs 4, yet allocation failed at
        // 3. Print what the allocator actually HAS at the moment it refuses, so the fix addresses
        // the real shortage rather than the one I guessed.
        extern uint64_t ds4p_blocks_checked_out();
        extern uint64_t ds4p_blocks_released();
        LLAMA_LOG_ERROR("%s: DS4P-ALLOCFAIL n_past=%d table_blocks=%zu free_gpu=%u total_gpu=%u "
                        "usable_gpu=%u block_size=%u | checked_out=%llu released=%llu OUTSTANDING=%lld\n",
                        __func__, n_past,
                        sd_group.block_table.size(), block_manager.num_free_gpu_blocks(),
                        num_gpu_blocks, block_manager.get_usable_gpu_blocks(), block_size,
                        (unsigned long long) ds4p_blocks_checked_out(),
                        (unsigned long long) ds4p_blocks_released(),
                        (long long) (ds4p_blocks_checked_out() - ds4p_blocks_released()));
        // ★ WHO HOLDS THE REST. The ledger proved four blocks are outstanding and not in the
        // group's table; it could not say WHO has them. sequence_blocks is a SHADOW COPY of a
        // group's table kept by request_id, so a superseded generation's ids can survive there
        // after the group itself moved on. Print its size next to the group's -- if they differ,
        // the shadow is the holder and "release at the supersede boundary" has a concrete target.
        {
            const auto sb = sequence_blocks.find(sd_group.request_id);
            LLAMA_LOG_ERROR("%s: DS4P-HOLDERS group_table=%zu sequence_blocks[%d]=%zu n_seq_entries=%zu\n",
                            __func__, sd_group.block_table.size(), sd_group.request_id,
                            sb == sequence_blocks.end() ? (size_t) 0 : sb->second.size(),
                            sequence_blocks.size());
        }
        if (n_past > 0) {
            LLAMA_LOG_ERROR("%s: PAGED KV LOST. self-drive could not grow the block table for %d "
                            "token(s) at n_past=%d (table holds %zu blocks x %u = %u tokens). The "
                            "sequence's KV is in the paged pool and the static path cannot see it, "
                            "so THIS REQUEST'S OUTPUT WILL BE CORRUPT FROM TOKEN %d ONWARD. Raise "
                            "-ngpub (or lower --kv-block-size) so the pool can grow with the "
                            "sequence.\n",
                            __func__, n_tokens, n_past, sd_group.block_table.size(), block_size,
                            (uint32_t) sd_group.block_table.size() * block_size, n_past);
        } else {
            // n_past == 0: nothing written yet, the static path is a real fallback.
            LLAMA_LOG_WARN("%s: self-drive could not allocate for %d tokens at n_past=0; taking the "
                           "static path (safe: no KV written yet)\n", __func__, n_tokens);
        }
        free_blocks(sd_group);
        sd_group  = {};
        sd_active = false;
        return false;
    }

    const int32_t n_blocks = (int32_t) sd_group.block_table.size();
    if (n_blocks <= 0) {
        free_blocks(sd_group);
        sd_group  = {};
        sd_active = false;
        return false;
    }

    sd_info                  = {};
    sd_info.n_seq            = 1;
    sd_info.n_tokens         = n_tokens;
    sd_info.n_blocks_per_seq = n_blocks;
    sd_info.write_slots      = new int32_t[n_tokens];
    sd_info.block_table      = new int32_t[n_blocks];
    sd_info.context_lens     = new int32_t[1];
    sd_info.batch_offsets    = new int32_t[1];
    sd_info.batch_lens       = new int32_t[1];
    sd_info.prefill_pending  = new int32_t[1];
    sd_info.seq_ids          = new int32_t[1];

    for (int32_t b = 0; b < n_blocks; ++b) {
        sd_info.block_table[b] = (int32_t) sd_group.block_table[b];
    }

    // Same mapping as llama_paged_scheduler_impl::calculate_global_slot_index, at ABSOLUTE
    // positions -- this is what carries the prefix across decode steps.
    for (int32_t i = 0; i < n_tokens; ++i) {
        const int32_t pos = n_past + i;
        const int32_t blk = pos / (int32_t) block_size;
        const int32_t off = pos % (int32_t) block_size;
        GGML_ASSERT(blk < n_blocks && "self-drive slot OOB -- block table too short for n_past");
        sd_info.write_slots[i] = sd_info.block_table[blk] * (int32_t) block_size + off;
    }

    // The kernel walks blocks up to context_lens, so this must be the FULL context, not the batch.
    sd_info.context_lens[0]    = n_past + n_tokens;
    sd_info.batch_offsets[0]   = 0;
    sd_info.batch_lens[0]      = n_tokens;
    sd_info.prefill_pending[0] = 0;
    sd_info.seq_ids[0]         = sd_group.request_id;

    sd_active = true;
    set_paged_batch_info(&sd_info);

    {
        static int n_calls = 0;
        ++n_calls;
        if (n_calls <= 8) {
            LLAMA_LOG_INFO("%s: DS4P-PAGED-DRIVE call #%d -- %s %d tok at n_past=%d, ctx=%d, %d blocks\n",
                           __func__, n_calls, is_prefill ? "PREFILL" : "decode",
                           n_tokens, n_past, sd_info.context_lens[0], n_blocks);
        }
    }

    return true;
}
