#include "llama-io.h"

#include <stdexcept>
#include "llama-kv-cache-paged.h"

#include <algorithm>

#include "llama-impl.h"

//
// llama_kv_cache_paged
//

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
    block_bytes    = 2 * block_size * n_heads_kv * head_dim * ggml_type_size(kv_type);

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
            if (layer_backends[il] == be) n_here++;
        }

        struct ggml_init_params gp;
        gp.mem_size   = ggml_tensor_overhead() * 5 * n_here;
        gp.mem_buffer = NULL;
        gp.no_alloc   = true;

        struct ggml_context * ctx = ggml_init(gp);
        GGML_ASSERT(ctx && "failed to create paged KV context");

        for (uint32_t il = 0; il < n_layers; ++il) {
            if (layer_backends[il] != be) continue;
            kv_gpu_layers[il] =
                ggml_new_tensor_4d(ctx, type, head_dim, block_size, 2 * n_heads_kv, n_gpu_blocks);
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        GGML_ASSERT(buf && "Failed to allocate paged KV buffer on a device");
        ggml_backend_buffer_clear(buf, 0);

        LLAMA_LOG_INFO("%s:   device %s holds %zu layer(s), %.2f MiB\n", __func__,
                       ggml_backend_name(be), n_here,
                       ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);

        gpu_ctxs.push_back(ctx);
        gpu_bufs.push_back(buf);
    }

    for (uint32_t il = 0; il < n_layers; ++il) {
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
            ggml_new_tensor_4d(ctx_cpu, type, head_dim, block_size, 2 * n_heads_kv, n_cpu_blocks));
    }
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(buf_cpu, 0);
    for (uint32_t il = 0; il < n_layers; ++il) {
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
    block_bytes    = 2 * block_size * n_heads_kv * head_dim * ggml_type_size(kv_type);

    // Set up GPU context and tensor
    // Interleaved shape: [num_blocks, 2, n_heads_kv, block_size, head_dim] (5D)
    struct ggml_init_params gpu_params;
    gpu_params.mem_size   = ggml_tensor_overhead() * 5 * n_layers;
    gpu_params.mem_buffer = NULL;
    gpu_params.no_alloc   = true;

    struct ggml_context * ctx_gpu = ggml_init(gpu_params);

    for (uint32_t il = 0; il < n_layers; ++il) {
        // Since GGML_MAX_DIMS is set to 4, we flatten the layout to be 4D: [num_blocks, 2 * n_heads_kv, block_size, head_dim]
        ggml_tensor * kv_layer_gpu =
            ggml_new_tensor_4d(ctx_gpu, type, head_dim, block_size, 2 * n_heads_kv, n_gpu_blocks);
        kv_gpu_layers.push_back(kv_layer_gpu);
    }

    // Allocate on GPU backend
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    GGML_ASSERT(buf_gpu && "Failed to allocate GPU KV cache buffer");
    ggml_backend_buffer_clear(buf_gpu, 0);  // zero out the cache
    for (uint32_t il = 0; il < n_layers; ++il) {
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
        ggml_tensor * kv_layer_cpu =
            ggml_new_tensor_4d(ctx_cpu, type, head_dim, block_size, 2 * n_heads_kv, n_cpu_blocks);
        kv_cpu_layers.push_back(kv_layer_cpu);
    }

    // Allocate on the CPU backend (using pinned memory for faster PCIe transfer)
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(buf_cpu, 0);  // zero out the cache
    for (uint32_t il = 0; il < n_layers; ++il) {
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

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_requested_blocks);
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
    seq_rm(group.request_id, llama_pos{}, llama_pos{});

    sequence_blocks.erase(group.request_id);
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");

    const auto & src_layers = to_gpu ? kv_cpu_layers : kv_gpu_layers;
    const auto & dst_layers = to_gpu ? kv_gpu_layers : kv_cpu_layers;

    GGML_ASSERT(src_layers.size() == n_layers && "src layer count mismatch.");
    GGML_ASSERT(dst_layers.size() == n_layers && "src layer count mismatch.");

    // Buffer on HOST to faciliate block data transfer
    // Note: an optimization would be to use views and async copies. Beware of
    // memory overhead heurisitcs.
    std::vector<uint8_t> staging(block_bytes);

    for (uint32_t il = 0; il < n_layers; ++il) {
        struct ggml_tensor * src_main = src_layers[il];
        struct ggml_tensor * dst_main = dst_layers[il];

        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];

            // GPU and CPu blocks may differ (usually CPU < GPU)
            // We substract the diffence to calculate where the local starts before we calculate offsets
            const uint32_t src_local = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            const size_t src_offset = (size_t) src_local * block_bytes;
            const size_t dst_offset = (size_t) dst_local * block_bytes;

            // Put src tensor into HOST staging buffer
            ggml_backend_tensor_get(src_main, staging.data(), src_offset, block_bytes);
            // Put tensor from HOST staging into dst tensor
            ggml_backend_tensor_set(dst_main, staging.data(), dst_offset, block_bytes);
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

    llama_block_ids new_ids = block_manager.checkout_gpu_blocks(num_blocks);
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

    std::vector<uint8_t> staging(block_bytes);

    for (uint32_t il = 0; il < n_layers; ++il) {
        for (uint32_t i = 0; i < n_blocks; ++i) {
            const uint32_t gid = blocks[i];
            const bool     gpu = block_manager.is_gpu(gid);

            struct ggml_tensor * layer = gpu ? kv_gpu_layers[il] : kv_cpu_layers[il];
            const uint32_t local = gpu ? gid : gid - num_gpu_blocks;

            ggml_backend_tensor_get(layer, staging.data(), (size_t) local * block_bytes, block_bytes);
            io.write(staging.data(), block_bytes);
        }
    }
}

// P1-5 RESTORE -- deliberately NOT implemented, and failing loudly rather than silently.
//
// Reading the bytes back is the easy half; the hard half is that a restored sequence
// needs BLOCKS ALLOCATED and, critically, its SCHEDULER GROUP's block_table repopulated.
// The cache can allocate (block_manager.checkout_*) but it cannot reach into the
// scheduler's group records, and a sequence whose cache map and group table disagree is
// exactly the silent-wrong-reuse that P0-2's revalidate guard exists to catch.
//
// So: refuse, loudly, with the reason. A no-op here would let a restore appear to
// succeed and then serve another request's KV -- strictly worse than not restoring.
void llama_kv_cache_paged::state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) {
    // THROW, do not abort. Measured: aborting here killed the server the moment a spilled
    // entry was admitted -- and a spilled entry is ALWAYS admitted eventually, so the
    // abort turned a working server into a crashing one. llama_context::state_seq_set_data
    // wraps this in try/catch and returns 0, which the server's prompt_load treats as a
    // failed restore -> prompt_clear() -> normal recompute. That is the correct
    // degradation: LOUD (the error is logged) but not fatal, and never silently wrong.
    throw std::runtime_error(
        "paged KV state_read is not implemented: restoring a sequence requires repopulating "
        "its scheduler group's block_table, which the cache cannot do on its own. Writing "
        "state (spill) is supported; reading it back (admit) is not yet -- falling back to "
        "recompute.");
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos /*p0*/, llama_pos /*p1*/) {
    sequence_positions.erase(seq_id);
    sequence_blocks.erase(seq_id);
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
    paged_write_slots   = info.write_slots;
    paged_block_table   = info.block_table;
    paged_context_lens  = info.context_lens;
    paged_batch_offsets = info.batch_offsets;
    paged_batch_lens    = info.batch_lens;
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
