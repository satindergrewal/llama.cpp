#include "llama-block-manager.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

void llama_block_manager::init(uint32_t n_gpu, uint32_t n_cpu, float watermark) {
    LLAMA_LOG_INFO("%s: Block manager initialized: n_free_gpu_blocks=%d, n_free_cpu_blocks=%d\n", __func__, n_gpu,
                   n_cpu);
    total_num_gpu_blocks = n_gpu;
    total_num_cpu_blocks = n_cpu;

    watermark_gpu_safety_num_blocks = std::ceil(total_num_gpu_blocks * watermark);
    watermark_cpu_safety_num_blocks = std::ceil(total_num_cpu_blocks * watermark);

    gpu_registry.resize(n_gpu);
    for (uint32_t i = 0; i < n_gpu; ++i) {
        gpu_registry[i].id     = i;
        gpu_registry[i].is_gpu = true;
        free_gpu_ids.push_back(i);
    }

    cpu_registry.resize(n_cpu);
    for (uint32_t i = 0; i < n_cpu; ++i) {
        cpu_registry[i].id     = i + total_num_gpu_blocks;
        cpu_registry[i].is_gpu = false;
        free_cpu_ids.push_back(i + total_num_gpu_blocks);
    }
}

size_t llama_block_manager::n_free_gpu_blocks() const {
    return free_gpu_ids.size();
}

size_t llama_block_manager::n_free_cpu_blocks() const {
    return free_cpu_ids.size();
}

bool llama_block_manager::has_free_gpu_blocks(uint32_t num_requested_blocks) const {
    size_t curr_free_gpus = free_gpu_ids.size();
    if (curr_free_gpus < watermark_gpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_gpus - watermark_gpu_safety_num_blocks);
}

bool llama_block_manager::has_free_cpu_blocks(uint32_t num_requested_blocks) const {
    size_t curr_free_cpus = free_cpu_ids.size();
    if (curr_free_cpus < watermark_cpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_cpus - watermark_cpu_safety_num_blocks);
}

// ★ BLOCK ACCOUNTING. Measured: at a refusal the sequence held 3 blocks, the free list had 1, and
// the pool has 8 -- four unaccounted for. checkout and release are the only two doors, so counting
// both and diffing them locates the shortfall instead of nominating a suspect.
//
// ⚠ CORRECTION TO THE NAME. Commits cfcac58a and 0b7be856 call this a "block LEAK". It is NOT one,
// and the word is wrong in both messages -- which cannot be rewritten, so the correction lives here
// where anyone reading the instrumentation will see it. The timestamped ledger shows 8 blocks
// checked out and 8 released: nothing is permanently lost. What happens is a TRANSIENT OVER-HOLD --
// blocks taken during warmup and during a superseded generation stay checked out until process
// teardown, and while they are held the LIVE sequence cannot grow. The pool is not drained; it is
// occupied by the past.
//
// The distinction decides the repair. A leak means "find who forgets to free". An over-hold means
// "free EARLIER" -- the release already exists, it just fires at teardown rather than at the
// boundary where the holding became pointless. I would have built the first one.
static uint64_t ds4p_n_checked_out = 0;
static uint64_t ds4p_n_released    = 0;
uint64_t ds4p_blocks_checked_out() { return ds4p_n_checked_out; }
uint64_t ds4p_blocks_released()    { return ds4p_n_released;    }

llama_block_manager::physical_block_ids llama_block_manager::checkout_gpu_blocks(uint32_t num_blocks, const char * tag) {
    ds4p_n_checked_out += num_blocks;
    LLAMA_LOG_ERROR("DS4P-CHECKOUT %s n=%u free_before=%zu\n", tag ? tag : "?", num_blocks, free_gpu_ids.size());
    physical_block_ids new_ids = {};
    if (num_blocks > free_gpu_ids.size()) {
        return new_ids;
    }

    new_ids.insert(new_ids.end(), std::make_move_iterator(free_gpu_ids.end() - num_blocks),
                   std::make_move_iterator(free_gpu_ids.end()));
    free_gpu_ids.erase(free_gpu_ids.end() - num_blocks, free_gpu_ids.end());

    for (const uint32_t & id : new_ids) {
        gpu_registry[id].ref_count += 1;
    }
    return new_ids;
}

llama_block_manager::physical_block_ids llama_block_manager::checkout_cpu_blocks(uint32_t num_blocks) {
    physical_block_ids new_ids = {};
    if (num_blocks > free_cpu_ids.size()) {
        return new_ids;
    }

    new_ids.insert(new_ids.end(), std::make_move_iterator(free_cpu_ids.end() - num_blocks),
                   std::make_move_iterator(free_cpu_ids.end()));
    free_cpu_ids.erase(free_cpu_ids.end() - num_blocks, free_cpu_ids.end());

    for (const uint32_t & id : new_ids) {
        cpu_registry[id - total_num_gpu_blocks].ref_count += 1;
    }
    return new_ids;
}

void llama_block_manager::share_blocks(const physical_block_ids & shared_blocks) {
    for (const uint32_t & id : shared_blocks) {
        if (is_gpu(id)) {
            gpu_registry[id].ref_count += 1;
        } else {
            cpu_registry[id - total_num_gpu_blocks].ref_count += 1;
        }
    }
}

uint32_t llama_block_manager::get_ref_count(uint32_t block) const {
    return is_gpu(block) ? gpu_registry[block].ref_count
                         : cpu_registry[block - total_num_gpu_blocks].ref_count;
}

void llama_block_manager::release_gpu_blocks(const physical_block_ids & freed_blocks_ids) {
    ds4p_n_released += freed_blocks_ids.size();
    LLAMA_LOG_ERROR("DS4P-RELEASE n=%zu\n", freed_blocks_ids.size());
    // ★ DS4P_BLOCK_AUDIT -- detector for the #1(c) hypothesis: a block released while ALREADY free
    // is pushed onto free_gpu_ids a SECOND time, so two live sequences can check out the SAME
    // physical block and overwrite each other's KV. Impossible on a cold server (nothing has ever
    // been released) and possible the moment a request finishes -- which is exactly the measured
    // warm/cold split: WARM 7/12 corrupt, COLD 0/12.
    //
    // ⚠ DETECTOR, NOT A FIX, and it must be able to print nothing. If a run corrupts and NO line
    // below fires, the hypothesis is refuted and gets recorded as refuted.
    const bool audit = getenv("DS4P_BLOCK_AUDIT") != nullptr;
    for (const uint32_t & id : freed_blocks_ids) {
        if (audit) {
            if (gpu_registry[id].ref_count <= 0) {
                LLAMA_LOG_ERROR("DS4P-DOUBLE-FREE block=%u refcount_was=%d\n", id, gpu_registry[id].ref_count);
            }
            if (std::find(free_gpu_ids.begin(), free_gpu_ids.end(), id) != free_gpu_ids.end()) {
                LLAMA_LOG_ERROR("DS4P-DUP-FREELIST block=%u already queued as free\n", id);
            }
        }
        gpu_registry[id].ref_count -= 1;
        if (gpu_registry[id].ref_count <= 0) {
            gpu_registry[id].ref_count = 0;
            free_gpu_ids.push_back(id);
        }
    }
    if (audit) {
        // Whole-list scan. The invariant is that a physical block appears in the free list AT MOST
        // ONCE; a duplicate means the next two checkouts can be handed the same block.
        std::vector<uint32_t> sorted(free_gpu_ids.begin(), free_gpu_ids.end());
        std::sort(sorted.begin(), sorted.end());
        const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
        if (dup != sorted.end()) {
            LLAMA_LOG_ERROR("DS4P-FREELIST-CORRUPT duplicate block=%u free_list_size=%zu\n",
                            *dup, free_gpu_ids.size());
        }
    }
}

void llama_block_manager::release_cpu_blocks(const physical_block_ids & freed_blocks_ids) {
    for (const uint32_t & id : freed_blocks_ids) {
        cpu_registry[id - total_num_gpu_blocks].ref_count -= 1;
        if (cpu_registry[id - total_num_gpu_blocks].ref_count <= 0) {
            cpu_registry[id - total_num_gpu_blocks].ref_count = 0;
            free_cpu_ids.push_back(id);
        }
    }
}

bool llama_block_manager::is_gpu(uint32_t block_id) const {
    return block_id < total_num_gpu_blocks;
}
