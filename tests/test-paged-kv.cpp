#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-paged-scheduler-impl.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#define TEST(name) static void name()
#define RUN(name)                                   \
    do {                                            \
        fprintf(stderr, "  running %-40s ", #name); \
        fflush(stderr);                             \
        name();                                     \
        fprintf(stderr, "OK\n");                    \
    } while (0)

#define EXPECT_EQ(a, b)                                                                                          \
    do {                                                                                                         \
        auto _a = (a);                                                                                           \
        auto _b = (b);                                                                                           \
        if (_a != _b) {                                                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: expected %s == %s, got %lld vs %lld\n", __FILE__, __LINE__, #a, #b, \
                    (long long) _a, (long long) _b);                                                             \
        }                                                                                                        \
    } while (0)

#define EXPECT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

// Testing block_manager main functionality

TEST(test_block_manager_leak_simple) {
    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);

    auto gpu_ids = block_manager.checkout_gpu_blocks(10);
    EXPECT_EQ(gpu_ids.size(), 10u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (n_gpu_blocks - 10u));

    auto cpu_ids = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(cpu_ids.size(), 5u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), (n_cpu_blocks - 5u));

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

// Testing that we always return to full (stress test)
TEST(test_block_manager_leak_repeated) {
    const uint32_t block_size   = 16;
    const uint32_t n_gpu_blocks = 64;
    const uint32_t n_cpu_blocks = 32;
    const float    watermark    = 0.0f;
    const int      n_iter       = 1000;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    for (int iter = 0; iter < n_iter; ++iter) {
        const uint32_t n   = (iter % block_size) + 1;
        auto           ids = block_manager.checkout_gpu_blocks(n);
        EXPECT_EQ(ids.size(), (size_t) n);
        block_manager.release_gpu_blocks(ids);
        EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    }
}

// Attempting to check-out more blocks than available. It should return empty.
TEST(test_block_manager_checkout_too_many) {
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto ids = block_manager.checkout_gpu_blocks(9);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);

    auto ids2 = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(ids2.size(), 0u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

TEST(test_block_manager_watermark) {
    // watermark=0.2 means 2 blocks reserved as safety (always consider max gpu blocks)
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.2f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    // 10 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(9));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(10));

    auto ids = block_manager.checkout_gpu_blocks(5);
    EXPECT_EQ(ids.size(), 5u);
    // 5 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(3));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(4));

    block_manager.release_gpu_blocks(ids);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
}

TEST(test_block_manager_watermark_zero) {
    // watermark=0 means the entire pool is requestable.
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(10));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(11));
}

TEST(test_block_manager_gpu_cpu_disjoint) {
    // Just making sure CPU and GPU blocks are disjoint
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto gpu_ids = block_manager.checkout_gpu_blocks(8);
    auto cpu_ids = block_manager.checkout_cpu_blocks(4);

    for (auto id : gpu_ids) {
        EXPECT_TRUE(block_manager.is_gpu(id));
    }
    for (auto id : cpu_ids) {
        EXPECT_FALSE(block_manager.is_gpu(id));
    }

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);
}

// Testing llama_kv_cache_paged book-keeping

// kv_cache_paged with arbitrary shape (we only care about the bookkeeping)
static llama_kv_cache_paged make_kv() {
    return llama_kv_cache_paged(
        /*head_dim=*/64,
        /*n_heads_kv=*/4,
        /*block_size=*/16,
        /*n_layers=*/2,
        /*n_ubatch=*/32,
        /*n_seq_max=*/8);
}

TEST(test_seq_pos_default_unknown) {
    auto kv = make_kv();
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
    EXPECT_EQ(kv.seq_pos_min(42), -1);
}

TEST(test_seq_pos_set_and_get) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);
    EXPECT_EQ(kv.seq_pos_max(0), 17);

    // Independent per seq_id.
    kv.set_seq_min_pos(1, 100);
    EXPECT_EQ(kv.seq_pos_min(1), 100);
    EXPECT_EQ(kv.seq_pos_min(0), 5);  // unchanged
}

TEST(test_seq_pos_overwrite) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(0, 9);
    EXPECT_EQ(kv.seq_pos_min(0), 9);
}

TEST(test_seq_pos_seq_rm_removes) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);

    kv.seq_rm(0, 0, 0);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
}

TEST(test_seq_pos_clear_removes_all) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(1, 7);
    kv.set_seq_min_pos(2, 9);

    kv.clear(/*data=*/false);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_min(1), -1);
    EXPECT_EQ(kv.seq_pos_min(2), -1);
}

TEST(test_free_blocks_releases_to_pool) {
    // Initialize a KV cache with a real CPU backend (used as both "GPU" and CPU).
    // This is fine for testing
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    auto kv = make_kv();
    kv.init(/*backend_gpu=*/backend,
            /*backend_cpu=*/backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, watermark);

    // allocate() pulls from the GPU block pool. After releasing, the count
    // must return to the initial value.
    const uint32_t n_gpu_initial = kv.get_num_gpu_blocks();
    EXPECT_EQ(n_gpu_initial, n_gpu_blocks);

    llama_sequence_group group;
    group.request_id = 0;
    group.n_prompt   = 32;  // 2 blocks
    group.n_decoded  = 0;

    bool success = kv.allocate(/*num_tokens=*/0, group);
    EXPECT_TRUE(success);
    EXPECT_EQ(group.block_table.size(), 2u);

    // Allocating a second sequence further reduces the pool.
    llama_sequence_group group2;
    group2.request_id = 1;
    group2.n_prompt   = 48;  // 3 blocks
    group2.n_decoded  = 0;

    success = kv.allocate(0, group2);
    EXPECT_TRUE(success);
    EXPECT_EQ(group2.block_table.size(), 3u);

    // Free the first sequence: we release 2 blocks
    kv.free_blocks(group);
    EXPECT_EQ(group.block_table.size(), 0u);

    llama_sequence_group group3;
    group3.request_id = 2;
    group3.n_prompt   = 32;  // 2 blocks
    group3.n_decoded  = 0;
    success           = kv.allocate(0, group3);
    EXPECT_TRUE(success);
    EXPECT_EQ(group3.block_table.size(), 2u);

    // All blocks released
    kv.free_blocks(group2);
    kv.free_blocks(group3);

    llama_sequence_group group_full;
    group_full.request_id = 99;
    group_full.n_prompt   = n_gpu_blocks * 16;  // exactly n_gpu_blocks worth
    group_full.n_decoded  = 0;
    success               = kv.allocate(0, group_full);
    EXPECT_TRUE(success);
    EXPECT_EQ(group_full.block_table.size(), 16u);
    kv.free_blocks(group_full);

    ggml_backend_free(backend);
}

// Testing scheduler

// For easy testing and clean-up.
// KV cache paged + scheduler (must free CPU backend)
struct paged_test_fixture {
    ggml_backend_t                              backend = nullptr;
    std::unique_ptr<llama_kv_cache_paged>       kv;
    std::unique_ptr<llama_paged_scheduler_impl> sched;

    ~paged_test_fixture() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    // rule of 5
    paged_test_fixture()                                       = default;
    paged_test_fixture(paged_test_fixture &&)                  = default;
    paged_test_fixture & operator=(paged_test_fixture &&)      = default;
    paged_test_fixture(const paged_test_fixture &)             = delete;
    paged_test_fixture & operator=(const paged_test_fixture &) = delete;
};

static paged_test_fixture make_fixture(uint32_t n_ctx           = 128,
                                       uint32_t block_size      = 16,
                                       uint32_t n_batch         = 64,
                                       uint32_t n_gpu_blocks    = 4,
                                       uint32_t n_cpu_blocks    = 2,
                                       uint32_t n_seq_max_batch = 0) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        /*block_size=*/block_size,
        /*n_layers=*/2u,
        /*n_ubatch=*/n_batch,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(n_ctx, block_size, n_batch, fixture.kv.get(), n_seq_max_batch));
    return fixture;
}

static llama_sequence_group make_group(int32_t request_id, uint32_t n_prompt) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = request_id;  // control ordering based on request_id
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

TEST(test_scheduler_no_deadlock_on_empty) {
    // No requests queued means no deadlock
    auto                   fixture = make_fixture();
    llama_batch            batch   = {};
    llama_scheduler_status status  = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 0);
}

TEST(test_scheduler_deadlock_oversize_waiting_request) {
    // There are 2 blocks, the waiting request needs 3.
    // Attempt to process prefill for this requets will fail and the request will remain in waiting (deadlock).
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1);
    auto group   = make_group(0, /*n_prompts=*/32);

    bool queued = fixture.sched->queue_request(group);
    EXPECT_TRUE(queued);

    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);

    // Next steps will continue remained deadlocked
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);
}

TEST(test_scheduler_rejects_oversized_prompt) {
    auto fixture = make_fixture(/*n_ctx=*/64, /*block_size=*/16, /*n_batch=*/128,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    bool queued = fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64));
    EXPECT_FALSE(queued);

    queued = fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/128));
    EXPECT_FALSE(queued);

    // A request just under the limit is accepted.
    queued = fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/63));
    EXPECT_TRUE(queued);
}

// Drive one prefill chunk so n_past crosses a block boundary. The share scan
// skips sources with n_past==0; without this the independent-share abort is
// invisible (every other scheduler test in this file queues and never updates).
static void prefill_one_chunk(paged_test_fixture & fixture, llama_batch & batch) {
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_TRUE(info->n_seq >= 1);
    std::vector<llama_token> toks((size_t) info->n_seq, /*dummy=*/1);
    std::vector<int8_t>      stop((size_t) info->n_seq, 0);
    fixture.sched->update(batch, toks, stop.data(), nullptr);
}

TEST(test_prefix_share_keeps_full_prompt) {
    // Independent-share used to let fork_blocks clobber n_prompt to the inherited
    // span. populate_batch_from then hit remaining_prompt==0 and aborted.
    // Measured 2026-08-14: "admitted SHARED: 64 of 64 ... 0 left to prefill".
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/80)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    const llama_sequence_group * a = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(a != nullptr);
    EXPECT_TRUE(a->n_past >= 64);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/80)));
    const llama_sequence_group * b = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(b != nullptr);
    EXPECT_TRUE(b->n_prompt == 80u);
    EXPECT_TRUE(b->logical_seq.size() == 80u);
    EXPECT_TRUE(b->n_past == 64u);
    EXPECT_TRUE(!b->block_table.empty());

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
}

TEST(test_fork_does_not_reshare) {
    // queue_forked_request already fork_blocks + restores n_prompt, then calls
    // queue_request. Scanning again would clobber n_prompt and leak a refcount.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/80)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0)->n_past >= 64);

    EXPECT_TRUE(fixture.sched->queue_forked_request(make_group(/*id=*/1, /*n_prompt=*/80), /*parent=*/0));
    const llama_sequence_group * b = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(b != nullptr);
    EXPECT_TRUE(b->n_prompt == 80u);
    EXPECT_TRUE(b->logical_seq.size() == 80u);
    EXPECT_TRUE(b->n_past == 64u);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
}

TEST(test_pool_full_children_wait_master_stays) {
    // vLLM PREEMPTED + prepend: master + children that TOGETHER exceed the
    // pool must not 500 / take_terminated. Children wait or swap (unref
    // tails). The highest-ref_cnt master prefix stays. When a child leaves,
    // a waiter resumes.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/4, /*n_cpu_blocks=*/4,
                                /*n_seq_max_batch=*/0);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past >= 32);

    // 40 = 32 inherited (2 shared blocks) + 8 unique. allocate() wants 1
    // extra block per child. 2 shared + 3 unique = 5 > 4 GPU blocks.
    // Do NOT use n_prompt==32: inherit would set n_past==n_prompt and
    // populate_batch_from asserts remaining_prompt==0.
    for (int id = 1; id <= 3; ++id) {
        EXPECT_TRUE(fixture.sched->queue_forked_request(make_group(id, /*n_prompt=*/40),
                                                       /*parent=*/0));
        EXPECT_TRUE(fixture.sched->get_group_from_id(id) != nullptr);
    }

    int n_parked_seen = 0;
    for (int step = 0; step < 8; ++step) {
        const llama_scheduler_status st = fixture.sched->step(batch);
        EXPECT_TRUE(fixture.sched->terminated_ids.empty());
        master = fixture.sched->get_group_from_id(0);
        EXPECT_TRUE(master != nullptr);
        EXPECT_TRUE(master->status != llama_sequence_group_status::FINISHED);

        int n_parked = 0;
        for (int id = 0; id <= 3; ++id) {
            const llama_sequence_group * g = fixture.sched->get_group_from_id(id);
            EXPECT_TRUE(g != nullptr);
            EXPECT_TRUE(g->status != llama_sequence_group_status::FINISHED);
            if (g->status == llama_sequence_group_status::WAITING ||
                g->status == llama_sequence_group_status::SWAPPED) {
                n_parked++;
            }
        }
        if (n_parked > 0) {
            n_parked_seen = n_parked;
        }

        if (st == llama_scheduler_status::OK && batch.n_tokens > 0) {
            const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
            EXPECT_TRUE(info != nullptr);
            std::vector<llama_token> toks((size_t) info->n_seq, /*dummy=*/1);
            std::vector<int8_t>      stop((size_t) info->n_seq, 0);
            fixture.sched->update(batch, toks, stop.data(), nullptr);
        }
        EXPECT_TRUE(st != llama_scheduler_status::DEADLOCK);
    }

    EXPECT_TRUE(fixture.sched->terminated_ids.empty());
    master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->status == llama_sequence_group_status::RUNNING ||
                master->status == llama_sequence_group_status::WAITING);
    EXPECT_TRUE(n_parked_seen >= 1);

    // Free a child's unique tail. A parked sibling must be able to resume --
    // the pool was only momentarily full.
    int victim = -1;
    for (int id = 1; id <= 3; ++id) {
        const llama_sequence_group * g = fixture.sched->get_group_from_id(id);
        if (g && g->status == llama_sequence_group_status::RUNNING) {
            victim = id;
            break;
        }
    }
    if (victim < 0) {
        for (int id = 1; id <= 3; ++id) {
            if (fixture.sched->get_group_from_id(id)) {
                victim = id;
                break;
            }
        }
    }
    EXPECT_TRUE(victim > 0);
    EXPECT_TRUE(fixture.sched->abort_request(victim));
    EXPECT_TRUE(fixture.sched->get_group_from_id(victim) == nullptr);

    EXPECT_TRUE(fixture.sched->step(batch) != llama_scheduler_status::DEADLOCK);
    EXPECT_TRUE(fixture.sched->terminated_ids.empty());
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) != nullptr);
    bool sibling_live = false;
    for (int id = 1; id <= 3; ++id) {
        if (id == victim) {
            continue;
        }
        if (fixture.sched->get_group_from_id(id) != nullptr) {
            sibling_live = true;
        }
    }
    EXPECT_TRUE(sibling_live);
}

TEST(test_batch_width_cap_does_not_reject) {
    // Cut 1: n_seq_max_batch is BATCH WIDTH, not an admission ceiling.
    // Three requests that all fit the pool must all queue. Each step emits
    // at most one sequence. After the first finishes, the next is in the
    // batch. A gate that still needs -np N as the concurrency ceiling
    // does not close this.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8,
                                /*n_seq_max_batch=*/1);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/16)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/16)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/16)));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) != nullptr);
    EXPECT_TRUE(fixture.sched->get_group_from_id(1) != nullptr);
    EXPECT_TRUE(fixture.sched->get_group_from_id(2) != nullptr);

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_EQ(info->n_seq, 1);

    // Do not use stop_flags here: a mid-prefill chunk ignores them and
    // the first step of a 16-token prompt is a prefill. abort() is the
    // admission-width probe -- group 0 leaves, 1 and 2 stay queued.
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) == nullptr);
    EXPECT_TRUE(fixture.sched->get_group_from_id(1) != nullptr);
    EXPECT_TRUE(fixture.sched->get_group_from_id(2) != nullptr);

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_TRUE(fixture.sched->get_group_from_id(1) != nullptr ||
                fixture.sched->get_group_from_id(2) != nullptr);
}

int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "test-paged-kv: block_manager\n");
    RUN(test_block_manager_leak_simple);
    RUN(test_block_manager_leak_repeated);
    RUN(test_block_manager_checkout_too_many);
    RUN(test_block_manager_watermark);
    RUN(test_block_manager_watermark_zero);
    RUN(test_block_manager_gpu_cpu_disjoint);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged seq_pos\n");
    RUN(test_seq_pos_default_unknown);
    RUN(test_seq_pos_set_and_get);
    RUN(test_seq_pos_overwrite);
    RUN(test_seq_pos_seq_rm_removes);
    RUN(test_seq_pos_clear_removes_all);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged free_blocks\n");
    RUN(test_free_blocks_releases_to_pool);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged scheduler\n");
    RUN(test_scheduler_no_deadlock_on_empty);
    RUN(test_scheduler_deadlock_oversize_waiting_request);
    RUN(test_scheduler_rejects_oversized_prompt);
    RUN(test_prefix_share_keeps_full_prompt);
    RUN(test_fork_does_not_reshare);
    RUN(test_pool_full_children_wait_master_stays);
    RUN(test_batch_width_cap_does_not_reject);

    fprintf(stderr, "test-paged-kv: ALL PASSED\n");
    return 0;
}
