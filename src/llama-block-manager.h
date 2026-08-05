#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class llama_block_manager {
    struct physical_block {
        uint32_t id        = 0;
        uint32_t ref_count = 0;
        bool     is_gpu    = false;
    };

    using physical_block_bool = std::vector<physical_block>;
    using physical_block_ids  = std::vector<uint32_t>;

    physical_block_bool gpu_registry;
    physical_block_bool cpu_registry;

    physical_block_ids free_gpu_ids;  // [0 to total_num_gpu_blocks - 1]
    physical_block_ids free_cpu_ids;  // [total_num_gpu_blocks, total_num_gpu_blocks + total_num_cpu_blocks]

    uint32_t watermark_gpu_safety_num_blocks;
    uint32_t watermark_cpu_safety_num_blocks;

    uint32_t total_num_gpu_blocks;
    uint32_t total_num_cpu_blocks;

  public:
    void init(uint32_t n_gpu, uint32_t n_cpu, float watermark);

    size_t n_free_gpu_blocks() const;
    size_t n_free_cpu_blocks() const;

    bool has_free_gpu_blocks(uint32_t num_requested_blocks) const;
    bool has_free_cpu_blocks(uint32_t num_requested_blocks) const;

    physical_block_ids checkout_gpu_blocks(uint32_t num_blocks);
    physical_block_ids checkout_cpu_blocks(uint32_t num_blocks);

    // P1-6 COW: take an extra reference on blocks now shared by another sequence. The
    // release path already decrements and only frees at zero, so sharing is symmetric.
    void share_blocks(const physical_block_ids & shared_blocks);

    uint32_t get_ref_count(uint32_t block) const;

    // Blocks a request can ACTUALLY obtain: total minus the watermark safety reserve.
    // ⚠ The distinction is not academic. With -ngpub 8 and the default 0.05 watermark,
    // ceil(8*0.05) = 1 block is permanently reserved, so usable capacity is 7 blocks -- and a
    // sequence that reaches exactly 7 blocks of context can never obtain an 8th, forever. A guard
    // written against the TOTAL pool does not fire, because the sequence never exceeds the total.
    // How many GPU blocks are actually on the free list right now. Diagnostic: "the pool is too
    // small" and "the pool will not hand out what it has" look identical from the caller.
    uint32_t num_free_gpu_blocks() const { return (uint32_t) free_gpu_ids.size(); }

    uint32_t get_usable_gpu_blocks() const {
        return total_num_gpu_blocks > watermark_gpu_safety_num_blocks
             ? total_num_gpu_blocks - watermark_gpu_safety_num_blocks : 0;
    }

    void release_gpu_blocks(const physical_block_ids & freed_blocks);
    void release_cpu_blocks(const physical_block_ids & freed_blocks);

    bool is_gpu(uint32_t block) const;
};
