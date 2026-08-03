#include "server-kv-bank.h"

#include "server-task.h"
#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

// v0 file format (little-endian):
//   magic   "KVBK"                       4 B
//   version u32 = 1                      4 B
//   id_len  u32, identity bytes          (P0-2 binding_identity, "" = unsealed entry)
//   n_tok   u64                          (prompt token count; token ids follow in v1+)
//   main_sz u64, main bytes              (llama_state_seq payload, target)
//   drft_sz u64, drft bytes              (draft payload, may be 0)
// File name: kvbk-<fnv1a64 of identity+sizes+n_tok>.kv (content-addressing proper lands
// with the probe side; this name is collision-safe enough for the spill witness).

server_kv_bank & server_kv_bank::instance() {
    static server_kv_bank bank;
    return bank;
}

server_kv_bank::server_kv_bank() {
    const char * s = getenv("DS4P_KV_BANK");
    if (s != nullptr && s[0] != '\0') {
        dir = s;
    }
}

static uint64_t fnv1a64(const uint8_t * p, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

void server_kv_bank::spill(const server_prompt_cache_state & entry) {
    if (!active()) {
        return;
    }

    const std::string & ident   = entry.binding_identity;
    const uint64_t      n_tok   = (uint64_t) entry.prompt.tokens.size();
    const uint64_t      main_sz = (uint64_t) entry.data.main.size();
    const uint64_t      drft_sz = (uint64_t) entry.data.drft.size();

    if (main_sz == 0) {
        SRV_WRN("%s", " - kv-bank: entry has no main state, not spilling\n");
        return;
    }

    uint64_t h = fnv1a64((const uint8_t *) ident.data(), ident.size());
    h = fnv1a64((const uint8_t *) &n_tok,   sizeof(n_tok),   h);
    h = fnv1a64((const uint8_t *) &main_sz, sizeof(main_sz), h);
    if (main_sz >= sizeof(uint64_t)) {
        // fold in a slice of the payload so equal-shape prompts do not collide
        h = fnv1a64(entry.data.main.data(), (size_t) (main_sz < 4096 ? main_sz : 4096), h);
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/kvbk-%016llx.kv", dir.c_str(), (unsigned long long) h);

    FILE * f = fopen(path, "wb");
    if (f == nullptr) {
        SRV_WRN(" - kv-bank: cannot open %s for spill, dropping\n", path);
        return;
    }

    bool ok = true;
    const uint32_t ver    = 1;
    const uint32_t id_len = (uint32_t) ident.size();

    ok = ok && fwrite("KVBK", 1, 4, f) == 4;
    ok = ok && fwrite(&ver, sizeof(ver), 1, f) == 1;
    ok = ok && fwrite(&id_len, sizeof(id_len), 1, f) == 1;
    ok = ok && (id_len == 0 || fwrite(ident.data(), 1, id_len, f) == id_len);
    ok = ok && fwrite(&n_tok, sizeof(n_tok), 1, f) == 1;
    ok = ok && fwrite(&main_sz, sizeof(main_sz), 1, f) == 1;
    ok = ok && fwrite(entry.data.main.data(), 1, (size_t) main_sz, f) == (size_t) main_sz;
    ok = ok && fwrite(&drft_sz, sizeof(drft_sz), 1, f) == 1;
    ok = ok && (drft_sz == 0 || fwrite(entry.data.drft.data(), 1, (size_t) drft_sz, f) == (size_t) drft_sz);

    if (fclose(f) != 0) {
        ok = false;
    }

    if (!ok) {
        SRV_WRN(" - kv-bank: short write on %s, removing\n", path);
        remove(path);
        return;
    }

    n_spilled++;
    SRV_INF(" - kv-bank: spilled evicted entry -> %s (%.3f MiB, n_tok = %llu, total spills = %llu)\n",
            path, (main_sz + drft_sz) / (1024.0 * 1024.0),
            (unsigned long long) n_tok, (unsigned long long) n_spilled);
}
