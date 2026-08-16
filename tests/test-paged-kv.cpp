#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-paged.h"
#include "llama-paged-scheduler-impl.h"
#include "llama-cparams.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

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
    const uint32_t n_usable = kv.get_usable_gpu_blocks();
    EXPECT_TRUE(n_usable > 0 && n_usable <= n_gpu_blocks);
    group_full.n_prompt   = n_usable * 16;
    group_full.n_decoded  = 0;
    success               = kv.allocate(0, group_full);
    EXPECT_TRUE(success);
    EXPECT_EQ(group_full.block_table.size(), (size_t) n_usable);
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
                                       uint32_t n_seq_max_batch = 0,
                                       float    watermark       = 0.0f) {
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
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, watermark);

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

TEST(test_init_zero_gpu_blocks_throws) {
    // ⚠ DESIGNED REFUSE, NOT GGML_ASSERT. common_fit_paged_kv_blocks sets
    // n_gpu_blocks=0 when the request will not fit; init used to abort 134.
    // A throw / false here is the contract; abort is the bug.
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    llama_kv_cache_paged kv(/*head_dim=*/64, /*n_heads_kv=*/4, /*block_size=*/16,
                            /*n_layers=*/2, /*n_ubatch=*/64, /*n_seq_max=*/4);
    bool threw = false;
    try {
        kv.init(backend, backend, GGML_TYPE_F16, /*n_gpu_blocks=*/0, /*n_cpu_blocks=*/1, 0.0f);
    } catch (const std::exception & e) {
        threw = true;
        const char * msg = e.what();
        EXPECT_TRUE(std::strstr(msg, "n_gpu_blocks=0") != nullptr);
        EXPECT_TRUE(std::strstr(msg, "Largest n_ctx that fits") != nullptr);
    }
    EXPECT_TRUE(threw);
    ggml_backend_free(backend);
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
    // pool must not 500 / take_terminated. Children wait, swap, or admit
    // unique on CPU (mixed table). The highest-ref_cnt master prefix stays.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/6, /*n_cpu_blocks=*/4,
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
    // Mixed-table children may ADMIT unique on CPU instead of parking.
    if (n_parked_seen < 1) {
        int n_child_running = 0;
        for (int id = 1; id <= 3; ++id) {
            const llama_sequence_group * g = fixture.sched->get_group_from_id(id);
            if (g && g->status == llama_sequence_group_status::RUNNING) {
                n_child_running++;
            }
        }
        EXPECT_TRUE(n_child_running >= 1);
    }

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

TEST(test_finished_prefix_survives_for_children) {
    // Independent-share used to scan only running ∪ waiting. After the
    // master finished, later children missed and full-prefilled. finish()
    // now parks the full-block prefix; children that arrive after abort
    // must admit SHARED (n_past covers the prefix), not cold from 0.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past >= 32);

    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) == nullptr);

    // Same prefix + short unique tail so remaining_prompt > 0 after inherit.
    for (int id = 1; id <= 2; ++id) {
        EXPECT_TRUE(fixture.sched->queue_request(make_group(id, /*n_prompt=*/40)));
        const llama_sequence_group * c = fixture.sched->get_group_from_id(id);
        EXPECT_TRUE(c != nullptr);
        EXPECT_TRUE(c->n_prompt == 40u);
        EXPECT_TRUE(c->logical_seq.size() == 40u);
        EXPECT_TRUE(c->n_past == 32u);
        EXPECT_TRUE(!c->block_table.empty());
    }

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
}

TEST(test_session_fork_live_and_parked) {
    // Harness: bind a master, fork a child while it is live, then after
    // the master finishes fork another child from the parked prefix.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0)->n_past >= 32);

    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->has_session("master"));

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    const llama_sequence_group * live_child = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(live_child != nullptr);
    EXPECT_TRUE(live_child->n_prompt == 40u);
    EXPECT_TRUE(live_child->n_past == 32u);
    EXPECT_TRUE(!live_child->block_table.empty());

    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) == nullptr);
    // Name follows the parked hold, not the reused slot id.
    EXPECT_TRUE(fixture.sched->has_session("master"));
    EXPECT_TRUE(fixture.sched->n_held_prefixes() >= 1u);

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/2, /*n_prompt=*/40),
                                                         "master"));
    const llama_sequence_group * parked_child = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(parked_child != nullptr);
    EXPECT_TRUE(parked_child->n_prompt == 40u);
    EXPECT_TRUE(parked_child->n_past == 32u);
    EXPECT_TRUE(!parked_child->block_table.empty());
}

TEST(test_named_session_grow_same_prefix) {
    // Cheap bookkeeping only -- not the Qwen 8k HTTP proof.
    // First request parks a named prefix. Second queue_request with the
    // SAME prompt (named-session grow) must inherit some blocks, keep a
    // prompt remainder (not a zero-remainder prefill), and step without
    // asserting. n_past must increase.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0)->n_past >= 32);
    EXPECT_TRUE(fixture.sched->bind_session("grow", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) == nullptr);
    EXPECT_TRUE(fixture.sched->has_session("grow"));
    EXPECT_TRUE(fixture.sched->n_held_prefixes() >= 1u);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/32)));
    EXPECT_TRUE(fixture.sched->bind_session("grow", /*request_id=*/1));
    const llama_sequence_group * g = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(g != nullptr);
    EXPECT_TRUE(g->n_prompt == 32u);
    EXPECT_TRUE(g->n_past > 0);
    EXPECT_TRUE(g->n_past < g->n_prompt);
    EXPECT_TRUE(!g->block_table.empty());
    const uint32_t past_before = g->n_past;

    prefill_one_chunk(fixture, batch);
    g = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(g != nullptr);
    EXPECT_TRUE(g->n_past > past_before);
}

TEST(test_named_master_not_eviction_victim) {
    // Named/held session prefix must not be shortened to admit a child.
    // Victim selection prefers unref child tails / non-session holds.
    // Unique suffix may move to CPU or the child may ADMIT mixed-table.
    // After room frees, a later child inherits the original N blocks, not N-k.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/7, /*n_cpu_blocks=*/2);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past >= 64);

    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->has_session("master"));
    EXPECT_TRUE(fixture.sched->n_held_prefixes() >= 1u);

    const int32_t hold_id = fixture.sched->session_request_id("master");
    const size_t  N       = fixture.sched->held_prefix_n_blocks(hold_id);
    EXPECT_TRUE(N == 4u);

    // Occupy the leftover slack block with a *named* dummy so evict()
    // cannot take an unref tail and must consider the parked master.
    llama_sequence_group dummy = make_group(/*id=*/10, /*n_prompt=*/8);
    dummy.logical_seq.assign(8, /*token=*/2);
    EXPECT_TRUE(fixture.sched->queue_request(std::move(dummy)));
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->bind_session("other", /*request_id=*/10));
    EXPECT_TRUE(fixture.sched->get_group_from_id(10) != nullptr);

    // Child shares 40 tokens -> inherits 32 (2 of N). Named hold still
    // has a unique suffix. Pool is full. evict_held_prefix must NOT
    // shorten the session; the child waits.
    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    const llama_sequence_group * child1 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(child1 != nullptr);
    EXPECT_TRUE(child1->n_past == 32u);

    EXPECT_TRUE(fixture.sched->step(batch) != llama_scheduler_status::DEADLOCK);
    EXPECT_TRUE(fixture.sched->terminated_ids.empty());
    EXPECT_TRUE(fixture.sched->held_prefix_n_blocks(hold_id) == N);
    child1 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(child1 != nullptr);
    EXPECT_TRUE(child1->status == llama_sequence_group_status::WAITING ||
                child1->status == llama_sequence_group_status::RUNNING);
    EXPECT_TRUE(child1->status != llama_sequence_group_status::SWAPPED);
    EXPECT_TRUE(fixture.sched->has_session("master"));

    // Later child wants the full prefix. Must inherit N blocks / 64
    // tokens, not N-k (the shortened 32 the old victim path left).
    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/2, /*n_prompt=*/72),
                                                         "master"));
    const llama_sequence_group * child2 = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(child2 != nullptr);
    EXPECT_TRUE(child2->n_past == (uint32_t) (N * 16));
    EXPECT_TRUE(fixture.sched->held_prefix_n_blocks(hold_id) == N);

    // Room frees: dummy leaves. Master stay N; child 2 still inherited N.
    EXPECT_TRUE(fixture.sched->abort_request(10));
    EXPECT_TRUE(fixture.sched->step(batch) != llama_scheduler_status::DEADLOCK);
    EXPECT_TRUE(fixture.sched->terminated_ids.empty());
    EXPECT_TRUE(fixture.sched->held_prefix_n_blocks(hold_id) == N);
    child2 = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(child2 != nullptr);
    EXPECT_TRUE(child2->n_past == (uint32_t) (N * 16));
    EXPECT_TRUE(fixture.sched->has_session("master"));
}

TEST(test_named_master_full_gpu_children_wait_not_cpu_swap) {
    // stories15M pool-full analog: 8 GPU + 8 CPU, block 16.
    // Named parked master holds 7 GPU blocks (112 tokens). Children
    // inherit 4 (64 tokens) and need unique. GPU is prefix-full
    // (1 leftover < 2 unique). allocate() admits unique on CPU
    // (mixed table) without rewriting master prefix GPU ids and
    // without close_session / whole-table swap. 2 of 10 GPU are reserved scratch so remap works when the pool is full.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/10, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/112)));
    llama_batch batch = {};
    for (int i = 0; i < 4; ++i) {
        const llama_sequence_group * m = fixture.sched->get_group_from_id(0);
        if (!m || m->n_past >= 112) {
            break;
        }
        prefill_one_chunk(fixture, batch);
    }
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past >= 112);

    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->has_session("master"));
    EXPECT_TRUE(fixture.sched->n_held_prefixes() >= 1u);
    const int32_t hold_id = fixture.sched->session_request_id("master");
    const size_t  N       = fixture.sched->held_prefix_n_blocks(hold_id);
    EXPECT_TRUE(N == 7u);

    for (int id = 1; id <= 3; ++id) {
        llama_sequence_group child = make_group(id, /*n_prompt=*/80);
        // Tail must differ or LCP is 80 (all dummy 1s) and inherit is 5
        // blocks -- then 1 free GPU is enough and the wait path is missed.
        for (size_t i = 64; i < child.logical_seq.size(); ++i) {
            child.logical_seq[i] = 2;
        }
        EXPECT_TRUE(fixture.sched->queue_forked_from_session(std::move(child), "master"));
        const llama_sequence_group * c = fixture.sched->get_group_from_id(id);
        EXPECT_TRUE(c != nullptr);
        EXPECT_TRUE(c->n_past == 64u);
    }

    const llama_block_ids * hold_bl = fixture.kv->get_sequence_blocks(hold_id);
    EXPECT_TRUE(hold_bl != nullptr && hold_bl->size() >= 4);
    std::vector<uint32_t> prefix_before(hold_bl->begin(), hold_bl->begin() + 4);
    for (uint32_t id : prefix_before) {
        EXPECT_TRUE(fixture.kv->is_gpu_block(id));
    }

    int n_running = 0;
    for (int step = 0; step < 12; ++step) {
        EXPECT_TRUE(fixture.sched->step(batch) != llama_scheduler_status::DEADLOCK);
        EXPECT_TRUE(fixture.sched->terminated_ids.empty());
        EXPECT_TRUE(fixture.sched->held_prefix_n_blocks(hold_id) == N);
        n_running = 0;
        for (int id = 1; id <= 3; ++id) {
            const llama_sequence_group * c = fixture.sched->get_group_from_id(id);
            EXPECT_TRUE(c != nullptr);
            EXPECT_TRUE(c->status != llama_sequence_group_status::SWAPPED);
            if (c->status == llama_sequence_group_status::RUNNING) {
                n_running++;
            }
        }
        if (n_running >= 1) {
            break;
        }
        if (batch.n_tokens > 0) {
            const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
            if (info && info->n_seq >= 1) {
                std::vector<llama_token> toks((size_t) info->n_seq, 1);
                std::vector<int8_t>      stop((size_t) info->n_seq, 0);
                fixture.sched->update(batch, toks, stop.data(), nullptr);
            }
        }
    }
    EXPECT_TRUE(n_running >= 1);
    EXPECT_TRUE(fixture.sched->has_session("master"));

    hold_bl = fixture.kv->get_sequence_blocks(hold_id);
    EXPECT_TRUE(hold_bl != nullptr && hold_bl->size() >= 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE((*hold_bl)[i] == prefix_before[i]);
    }
}

TEST(test_mixed_table_child_admits_when_gpu_full) {
    // Child inherits 64 tokens / 4 GPU blocks and needs unique. GPU is prefix-full after the parked master (usable 8 after 2 scratch). Child must ADMIT without
    // close_session. Unique suffix lives on CPU (mixed stored table).
    // Shared prefix GPU ids stay. Decode remap makes the batch table
    // all-GPU so get_kv_tensor stays GPU-only.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/10, /*n_cpu_blocks=*/8,
                                /*n_seq_max_batch=*/0, /*watermark=*/0.2f);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/112)));
    llama_batch batch = {};
    for (int i = 0; i < 4; ++i) {
        const llama_sequence_group * m = fixture.sched->get_group_from_id(0);
        if (!m || m->n_past >= 112) {
            break;
        }
        prefill_one_chunk(fixture, batch);
    }
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past >= 112);

    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->has_session("master"));
    const int32_t hold_id = fixture.sched->session_request_id("master");
    const size_t  N       = fixture.sched->held_prefix_n_blocks(hold_id);
    EXPECT_TRUE(N == 7u);

    llama_sequence_group child = make_group(/*id=*/1, /*n_prompt=*/80);
    for (size_t i = 64; i < child.logical_seq.size(); ++i) {
        child.logical_seq[i] = 2;
    }
    EXPECT_TRUE(fixture.sched->queue_forked_from_session(std::move(child), "master"));
    const llama_sequence_group * c0 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(c0 != nullptr);
    EXPECT_TRUE(c0->n_past == 64u);
    EXPECT_TRUE(c0->block_table.size() == 4u);

    const llama_block_ids * hold_bl = fixture.kv->get_sequence_blocks(hold_id);
    EXPECT_TRUE(hold_bl != nullptr && hold_bl->size() >= 4);
    std::vector<uint32_t> prefix_before(hold_bl->begin(), hold_bl->begin() + 4);
    for (uint32_t id : prefix_before) {
        EXPECT_TRUE(fixture.kv->is_gpu_block(id));
    }

    bool admitted = false;
    for (int step = 0; step < 12; ++step) {
        const llama_scheduler_status st = fixture.sched->step(batch);
        EXPECT_TRUE(st != llama_scheduler_status::DEADLOCK);
        EXPECT_TRUE(fixture.sched->terminated_ids.empty());
        EXPECT_TRUE(fixture.sched->held_prefix_n_blocks(hold_id) == N);

        const llama_sequence_group * c = fixture.sched->get_group_from_id(1);
        EXPECT_TRUE(c != nullptr);
        EXPECT_TRUE(c->status != llama_sequence_group_status::SWAPPED);
        if (c->status == llama_sequence_group_status::RUNNING) {
            admitted = true;
            const uint32_t n_gpu = fixture.kv->get_num_gpu_blocks();
            bool has_cpu = false;
            bool has_gpu = false;
            for (uint32_t id : c->block_table) {
                if (id >= n_gpu) {
                    has_cpu = true;
                } else {
                    has_gpu = true;
                }
            }
            EXPECT_TRUE(has_gpu);
            EXPECT_TRUE(has_cpu || fixture.kv->count_cpu_unique(*c) > 0);

            const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
            if (info && info->block_table && info->n_seq >= 1) {
                for (int i = 0; i < info->n_seq * info->n_blocks_per_seq; ++i) {
                    const int32_t bid = info->block_table[i];
                    if (bid >= 0) {
                        EXPECT_TRUE((uint32_t) bid < n_gpu);
                    }
                }
            }
            break;
        }
        if (batch.n_tokens > 0) {
            const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
            if (info && info->n_seq >= 1) {
                std::vector<llama_token> toks((size_t) info->n_seq, 1);
                std::vector<int8_t>      stop((size_t) info->n_seq, 0);
                fixture.sched->update(batch, toks, stop.data(), nullptr);
            }
        }
    }
    EXPECT_TRUE(admitted);
    EXPECT_TRUE(fixture.sched->has_session("master"));
    // Do NOT require close_session for the child to admit.

    hold_bl = fixture.kv->get_sequence_blocks(hold_id);
    EXPECT_TRUE(hold_bl != nullptr && hold_bl->size() >= 4);
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE((*hold_bl)[i] == prefix_before[i]);
    }
    const llama_sequence_group * c = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(c != nullptr);
    EXPECT_TRUE(c->block_table.size() >= 4);
    for (size_t i = 0; i < 4 && i < c->block_table.size(); ++i) {
        EXPECT_TRUE(c->block_table[i] == prefix_before[i]);
    }
}

 TEST(test_mixed_remap_fail_once_does_not_spin) {
    // 4 GPU, watermark 0. Master prompt 48 fits (allocate wants 49 -> 4 blocks).
    // Park keeps 3 full GPU blocks. Child inherits 48 and needs 2 unique;
    // 1 leftover GPU is not enough so unique is CPU. Remap need=2 > scratch=1
    // -- cannot checkout without touching the prefix. Fail that child ONCE.
    // A second step must not grow terminated_ids.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/4, /*n_cpu_blocks=*/4);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/48)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past >= 48);

    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->has_session("master"));
    const int32_t hold_id = fixture.sched->session_request_id("master");
    EXPECT_TRUE(fixture.sched->held_prefix_n_blocks(hold_id) == 3u);

    llama_sequence_group child = make_group(/*id=*/1, /*n_prompt=*/64);
    for (size_t i = 48; i < child.logical_seq.size(); ++i) {
        child.logical_seq[i] = 2;
    }
    EXPECT_TRUE(fixture.sched->queue_forked_from_session(std::move(child), "master"));
    const llama_sequence_group * c0 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(c0 != nullptr);
    EXPECT_TRUE(c0->n_past == 48u);
    EXPECT_TRUE(c0->block_table.size() == 3u);
    EXPECT_TRUE(fixture.sched->terminated_ids.empty());

    EXPECT_TRUE(fixture.sched->step(batch) != llama_scheduler_status::DEADLOCK);
    EXPECT_TRUE(fixture.sched->terminated_ids.size() == 1u);
    EXPECT_TRUE(fixture.sched->terminated_ids[0] == 1);
    const llama_sequence_group * c1 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(c1 == nullptr || c1->status == llama_sequence_group_status::FINISHED);

    const size_t n_term = fixture.sched->terminated_ids.size();
    EXPECT_TRUE(fixture.sched->step(batch) != llama_scheduler_status::DEADLOCK);
    EXPECT_TRUE(fixture.sched->terminated_ids.size() == n_term);
}
 

TEST(test_session_close_keeps_child_refs) {
    // close_session drops the session's extra hold refs. Children that
    // inherited the prefix keep theirs. A later named fork misses.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->bind_session("master", 0));
    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->n_held_prefixes() >= 1u);

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/2, /*n_prompt=*/40),
                                                         "master"));
    const llama_sequence_group * c1 = fixture.sched->get_group_from_id(1);
    const llama_sequence_group * c2 = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(c1 != nullptr && c1->n_past == 32u && !c1->block_table.empty());
    EXPECT_TRUE(c2 != nullptr && c2->n_past == 32u && !c2->block_table.empty());

    EXPECT_TRUE(fixture.sched->close_session("master"));
    EXPECT_FALSE(fixture.sched->has_session("master"));
    EXPECT_FALSE(fixture.sched->close_session("master")); // second close is a miss
    EXPECT_EQ(fixture.sched->n_held_prefixes(), 0u);

    // Children still hold the inherited prefix.
    c1 = fixture.sched->get_group_from_id(1);
    c2 = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(c1 != nullptr && c1->n_past == 32u && !c1->block_table.empty());
    EXPECT_TRUE(c2 != nullptr && c2->n_past == 32u && !c2->block_table.empty());

    // Named fork no longer resolves. Must fail loud -- not admit as cold/APC.
    EXPECT_FALSE(fixture.sched->queue_forked_from_session(make_group(/*id=*/3, /*n_prompt=*/40),
                                                          "master"));
    EXPECT_TRUE(fixture.sched->get_group_from_id(3) == nullptr);
    EXPECT_FALSE(fixture.sched->has_session("master"));
}

TEST(test_session_omitted_is_noop) {
    // No session_id: bind/close miss, queue is the old path.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);
    EXPECT_FALSE(fixture.sched->bind_session("", 0));
    EXPECT_FALSE(fixture.sched->has_session("nope"));
    EXPECT_FALSE(fixture.sched->close_session("nope"));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) != nullptr);
}

TEST(test_session_survives_short_prefix) {
    // Named session must stay resolvable after finish even when n_past
    // is below one block. Losing the NAME is the hole; children may
    // inherit 0 full blocks and prefill the short prefix.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/13)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    const llama_sequence_group * master = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(master != nullptr);
    EXPECT_TRUE(master->n_past > 0);
    EXPECT_TRUE(master->n_past < 16);

    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));
    EXPECT_TRUE(fixture.sched->has_session("master"));

    EXPECT_TRUE(fixture.sched->abort_request(0));
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) == nullptr);
    EXPECT_TRUE(fixture.sched->has_session("master"));

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/13),
                                                         "master"));
    EXPECT_TRUE(fixture.sched->get_group_from_id(1) != nullptr);
    EXPECT_TRUE(fixture.sched->has_session("master"));
}

TEST(test_fork_predicate_hybrid_rollback) {
    // Dense always. Hybrid+rollback (DSV4) and hybrid with no RS also.
    // Only non-rewindable hybrid with RS must refuse.
    EXPECT_TRUE(llama_paged_fork_allowed(/*hybrid=*/false, /*rollback=*/false, /*rs=*/false));
    EXPECT_TRUE(llama_paged_fork_allowed(/*hybrid=*/false, /*rollback=*/false, /*rs=*/true));
    EXPECT_TRUE(llama_paged_fork_allowed(/*hybrid=*/true,  /*rollback=*/true,  /*rs=*/true));
    EXPECT_TRUE(llama_paged_fork_allowed(/*hybrid=*/true,  /*rollback=*/false, /*rs=*/false));
    EXPECT_FALSE(llama_paged_fork_allowed(/*hybrid=*/true, /*rollback=*/false, /*rs=*/true));
}

TEST(test_session_fork_hybrid_rollback_inherits) {
    // DSV4-class: is_hybrid but supports_rs_rollback. /fork must take
    // fork_blocks, not degrade to queue_request (which APC may then share).
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);
    fixture.sched->set_hybrid(true);
    fixture.sched->set_has_recurrent_state(true);
    fixture.sched->set_supports_rs_rollback(true);
    EXPECT_TRUE(fixture.sched->can_fork());

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0)->n_past >= 32);
    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    EXPECT_TRUE(fixture.sched->last_fork_used_blocks());
    const llama_sequence_group * child = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(child != nullptr);
    EXPECT_TRUE(child->n_prompt == 40u);
    EXPECT_TRUE(child->n_past == 32u);
    EXPECT_TRUE(!child->block_table.empty());
}

TEST(test_session_fork_hybrid_norewind_refuses) {
    // True non-rewindable hybrid (SSM): refuse loud. Do not silent-cold
    // via queue_request / APC share.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);
    fixture.sched->set_hybrid(true);
    fixture.sched->set_has_recurrent_state(true);
    fixture.sched->set_supports_rs_rollback(false);
    EXPECT_FALSE(fixture.sched->can_fork());

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->bind_session("master", 0));

    EXPECT_FALSE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                          "master"));
    EXPECT_FALSE(fixture.sched->last_fork_used_blocks());
    EXPECT_TRUE(fixture.sched->get_group_from_id(1) == nullptr);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) != nullptr);
    EXPECT_TRUE(fixture.sched->has_session("master"));
}

TEST(test_session_fork_hybrid_no_rs_inherits) {
    // Hybrid wrapper with no recurrent state (pure SWA / MSA): nothing to
    // rewind, so /fork still takes fork_blocks.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);
    fixture.sched->set_hybrid(true);
    fixture.sched->set_has_recurrent_state(false);
    fixture.sched->set_supports_rs_rollback(false);
    EXPECT_TRUE(fixture.sched->can_fork());

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->bind_session("master", 0));

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    EXPECT_TRUE(fixture.sched->last_fork_used_blocks());
    const llama_sequence_group * child = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(child != nullptr);
    EXPECT_TRUE(child->n_past == 32u);
    EXPECT_TRUE(!child->block_table.empty());
}

TEST(test_unknown_session_does_not_admit_cold) {
    // Unknown parent_session_id must fail loud. Falling back to queue_request
    // would admit a cold/APC request and lie that the fork happened.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) != nullptr);

    EXPECT_FALSE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                          "no-such-session"));
    EXPECT_TRUE(fixture.sched->get_group_from_id(1) == nullptr);
    EXPECT_TRUE(fixture.sched->get_group_from_id(0) != nullptr);
    EXPECT_FALSE(fixture.sched->has_session("no-such-session"));
}

TEST(test_named_fork_n_past_is_http_cache_n) {
    // HTTP timings.cache_n is slot.stats.n_prompt_cached. The server copies
    // llama_paged_scheduler_get_seq_state().n_past immediately after a named
    // /fork. That number is whole physical blocks inherited by reference --
    // not the token LCP, not APC, not prompt-cache warm. A child that
    // inherited 32 tokens must expose n_past=32 here or cache_n stays 0
    // and the fork looks fake.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    EXPECT_TRUE(fixture.sched->last_fork_used_blocks());
    const llama_sequence_group * child = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(child != nullptr);
    EXPECT_EQ(child->n_past, 32u);
    EXPECT_TRUE(child->n_past > 0);
    EXPECT_TRUE((child->n_past % 16u) == 0);
    EXPECT_TRUE(!child->block_table.empty());
}

TEST(test_hybrid_rs_few_live_cells) {
    // 4 cells, not 256. Sequential check-in reuses a freed cell.
    // Two overlapping children need a prefix hold + two tails.
    EXPECT_EQ(llama_hybrid_rs_size(/*n_seq_max=*/1, /*kv_paged=*/false), 1u);
    EXPECT_EQ(llama_hybrid_rs_size(/*n_seq_max=*/1, /*kv_paged=*/true), 4u);
    EXPECT_EQ(llama_hybrid_rs_size(/*n_seq_max=*/2, /*kv_paged=*/true), 4u);
    EXPECT_EQ(llama_hybrid_rs_size(/*n_seq_max=*/8, /*kv_paged=*/true), 8u);
    EXPECT_TRUE(LLAMA_HYBRID_RS_CELLS_PAGED == 4);
    EXPECT_TRUE(LLAMA_HYBRID_RS_CELLS_PAGED < LLAMA_MAX_SEQ);
}

TEST(test_dsv4_bookkeeping_id_space) {
    // DSV4 cache cannot be constructed without weights. The helper is the
    // contract: n_seq_max is batch width; after grow_paged_slot, ids 0 and 1
    // must be legal. Id space is LLAMA_MAX_SEQ (256), not n_seq_max.
    EXPECT_TRUE(llama_dsv4_seq_id_ok(0));
    EXPECT_TRUE(llama_dsv4_seq_id_ok(1));
    EXPECT_TRUE(llama_dsv4_seq_id_ok((llama_seq_id) (LLAMA_MAX_SEQ - 1)));
    EXPECT_FALSE(llama_dsv4_seq_id_ok(-1));
    EXPECT_FALSE(llama_dsv4_seq_id_ok((llama_seq_id) LLAMA_MAX_SEQ));
    EXPECT_EQ(llama_dsv4_seq_id_max(), (uint32_t) LLAMA_MAX_SEQ);
}

TEST(test_two_named_children_batch_width_one) {
    // Product: two concurrent children of one named session. Decode stays
    // single-width (n_seq_max_batch=1). Both inherit the prefix by reference.
    // Bookkeeping ids 1 and 2 must be accepted -- the DSV4 wrapper used to
    // GGML_ASSERT seq_id < n_seq_max (1) on the second child.
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8,
                                /*n_seq_max_batch=*/1);
    fixture.sched->set_hybrid(true);
    fixture.sched->set_has_recurrent_state(true);
    fixture.sched->set_supports_rs_rollback(true);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    llama_batch batch = {};
    prefill_one_chunk(fixture, batch);
    EXPECT_TRUE(fixture.sched->bind_session("master", /*request_id=*/0));

    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/1, /*n_prompt=*/40),
                                                         "master"));
    EXPECT_TRUE(fixture.sched->queue_forked_from_session(make_group(/*id=*/2, /*n_prompt=*/40),
                                                         "master"));
    const llama_sequence_group * c1 = fixture.sched->get_group_from_id(1);
    const llama_sequence_group * c2 = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(c1 != nullptr && c1->n_past == 32u && !c1->block_table.empty());
    EXPECT_TRUE(c2 != nullptr && c2->n_past == 32u && !c2->block_table.empty());
    EXPECT_TRUE(fixture.sched->last_fork_used_blocks());

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const llama_paged_batch_info * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_EQ(info->n_seq, 1);
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

TEST(test_hybrid_decode_attaches_paged_ctx) {
    // The "hybrid DECODE gate pending" construction log was stale. Decode
    // already takes ggml_paged_attn (-> ggml_metal_op_paged_attn) when the
    // pool exists and the scheduler has set batch info. The hybrid wrapper
    // calls init_batch_with_ubatches on that pool -- this test is that
    // attach, without 27B weights.
    //
    // Qwen3.8 interval-4: (il+1) % 4 == 0 is full attention; the rest are
    // recurrent and hold no paged KV. Warmup STATIC on 3,7,11 is the
    // reserve graph (no batch info yet), not decode fallback.

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    const uint32_t n_layers   = 8;
    const uint32_t block_size = 16;
    llama_kv_cache_paged kv(/*head_dim=*/64, /*n_heads_kv=*/4, block_size, n_layers,
                            /*n_ubatch=*/64, /*n_seq_max=*/8);

    std::vector<uint8_t> has_kv(n_layers, 0);
    for (uint32_t il = 0; il < n_layers; ++il) {
        // same formula as qwen35.cpp: is_recr = (il+1) % 4 != 0
        has_kv[il] = ((il + 1) % 4 == 0) ? 1 : 0;
    }
    kv.set_layer_filter(std::move(has_kv));
    kv.init(backend, backend, GGML_TYPE_F16, /*n_gpu_blocks=*/16, /*n_cpu_blocks=*/4, 0.0f);

    // Recurrent layers have no tensor. Interval-4 attn layers 3 and 7 do.
    EXPECT_TRUE(kv.get_kv_tensor(0) == nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(1) == nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(2) == nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(3) != nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(4) == nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(5) == nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(6) == nullptr);
    EXPECT_TRUE(kv.get_kv_tensor(7) != nullptr);

    // Warmup/reserve: no scheduler batch info -> hybrid wrapper cannot
    // attach a paged child. That is the STATIC print on 3,7,11.
    EXPECT_FALSE(kv.has_paged_batch_info());
    {
        llama_ubatch ub = {};
        ub.n_tokens     = 1;
        ub.n_seqs       = 1;
        auto ctx        = kv.init_batch_with_ubatches({ub});
        EXPECT_TRUE(ctx != nullptr);
        EXPECT_TRUE(ctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS);
    }

    // Scheduler-driven decode: batch info present -> paged child exists
    // and get_k is live on the interval-4 attention layers. This is the
    // path qwen35.cpp feeds to build_attn_paged_or_null.
    llama_paged_scheduler_impl sched(/*n_ctx=*/256, block_size, /*n_batch=*/64, &kv,
                                     /*n_seq_max_batch=*/1);
    EXPECT_TRUE(sched.queue_request(make_group(/*id=*/0, /*n_prompt=*/16)));
    llama_batch batch = {};
    EXPECT_TRUE(sched.step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(kv.has_paged_batch_info());

    llama_ubatch ub = {};
    ub.n_tokens     = 1;
    ub.n_seqs       = 1;
    auto ctx        = kv.init_batch_with_ubatches({ub});
    EXPECT_TRUE(ctx != nullptr);
    EXPECT_TRUE(ctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    const auto * paged = ctx->get_attn_paged();
    EXPECT_TRUE(paged != nullptr);
    EXPECT_TRUE(paged->get_k(3) != nullptr);
    EXPECT_TRUE(paged->get_k(7) != nullptr);
    EXPECT_TRUE(paged->get_k(0) == nullptr);
    EXPECT_TRUE(paged->get_k(1) == nullptr);
    EXPECT_TRUE(paged->get_k(2) == nullptr);

    ggml_backend_free(backend);
}

int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "test-paged-kv: hybrid RS few live cells\n");
    RUN(test_hybrid_rs_few_live_cells);
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
    RUN(test_init_zero_gpu_blocks_throws);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged scheduler\n");
    RUN(test_scheduler_no_deadlock_on_empty);
    RUN(test_scheduler_deadlock_oversize_waiting_request);
    RUN(test_scheduler_rejects_oversized_prompt);
    RUN(test_prefix_share_keeps_full_prompt);
    RUN(test_fork_does_not_reshare);
    RUN(test_pool_full_children_wait_master_stays);
    RUN(test_finished_prefix_survives_for_children);
    RUN(test_session_fork_live_and_parked);
    RUN(test_named_session_grow_same_prefix);
    RUN(test_named_master_not_eviction_victim);
    RUN(test_named_master_full_gpu_children_wait_not_cpu_swap);
    RUN(test_mixed_table_child_admits_when_gpu_full);
    RUN(test_mixed_remap_fail_once_does_not_spin);
    RUN(test_session_close_keeps_child_refs);
    RUN(test_session_omitted_is_noop);
    RUN(test_session_survives_short_prefix);
    RUN(test_fork_predicate_hybrid_rollback);
    RUN(test_session_fork_hybrid_rollback_inherits);
    RUN(test_session_fork_hybrid_norewind_refuses);
    RUN(test_session_fork_hybrid_no_rs_inherits);
    RUN(test_unknown_session_does_not_admit_cold);
    RUN(test_named_fork_n_past_is_http_cache_n);
    RUN(test_dsv4_bookkeeping_id_space);
    RUN(test_two_named_children_batch_width_one);
    RUN(test_batch_width_cap_does_not_reject);
    RUN(test_hybrid_decode_attaches_paged_ctx);

    fprintf(stderr, "test-paged-kv: ALL PASSED\n");
    return 0;
}
