#include "server-kv-bank.h"

#include "server-task.h"
#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>

// v0 file format (little-endian):
//   magic   "KVBK"                       4 B
//   version u32 = 1                      4 B
//   id_len  u32, identity bytes          (P0-2 binding_identity, "" = unsealed entry)
//   n_tok   u64, token ids (i32 each)    (prompt tokens -- the probe's LCP key)
//   main_sz u64, main bytes              (llama_state_seq payload, target)
//   drft_sz u64, drft bytes              (draft payload, may be 0)
// File name: kvbk-<fnv1a64 of identity+sizes+n_tok>.kv (content-addressing proper lands
// with the probe side; this name is collision-safe enough for the spill witness).

void server_kv_bank::configure(const std::string & dir_in, int32_t cap_mib) {
    if (dir_in == "-") {
        // --no-kv-bank: force-disable regardless of env
        dir.clear();
        return;
    }
    if (!dir_in.empty()) {
        dir = dir_in;
    }
    if (cap_mib > 0) {
        cap_bytes = (uint64_t) cap_mib * 1024ull * 1024ull;
    }
    if (active()) {
        SRV_INF(" - kv-bank: enabled, dir = %s, cap = %s\n", dir.c_str(),
                cap_bytes ? (std::to_string(cap_bytes / (1024*1024)) + " MiB").c_str() : "uncapped");
    }
}

server_kv_bank & server_kv_bank::instance() {
    static server_kv_bank bank;
    return bank;
}

server_kv_bank::server_kv_bank() {
    const char * s = getenv("DS4P_KV_BANK");
    if (s != nullptr && s[0] != '\0') {
        dir = s;
    }
    const char * cap = getenv("DS4P_KV_BANK_CAP_MIB");
    if (cap != nullptr) {
        cap_bytes = (uint64_t) atoll(cap) * 1024ull * 1024ull;
    }
}

// size-capped LRU by mtime: after each spill, drop the least-recently-touched files until
// the bank fits the cap. mtime is refreshed on admit (utimes-free: rewrite is overkill, a
// read does not bump mtime -- admit calls touch() below). Best-effort like everything here.
void server_kv_bank::enforce_cap() {
    if (cap_bytes == 0) {
        return; // uncapped
    }

    struct file_info { std::string path; time_t mtime; uint64_t size; };
    std::vector<file_info> files;
    uint64_t total = 0;

    DIR * d = opendir(dir.c_str());
    if (d == nullptr) return;
    for (dirent * e = readdir(d); e != nullptr; e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() < 4 || name.compare(name.size() - 3, 3, ".kv") != 0) continue;
        const std::string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) != 0) continue;
        files.push_back({path, st.st_mtime, (uint64_t) st.st_size});
        total += (uint64_t) st.st_size;
    }
    closedir(d);

    if (total <= cap_bytes) return;

    std::sort(files.begin(), files.end(),
              [](const file_info & a, const file_info & b) { return a.mtime < b.mtime; });

    for (const auto & fi : files) {
        if (total <= cap_bytes) break;
        if (remove(fi.path.c_str()) == 0) {
            total -= fi.size;
            SRV_INF(" - kv-bank: cap %.0f MiB exceeded, evicted %s (%.3f MiB)\n",
                    cap_bytes / (1024.0 * 1024.0), fi.path.c_str(), fi.size / (1024.0 * 1024.0));
        }
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
    {
        const llama_tokens toks = entry.prompt.tokens.get_text_tokens();
        ok = ok && (uint64_t) toks.size() == n_tok;
        ok = ok && (n_tok == 0 || fwrite(toks.data(), sizeof(llama_token), (size_t) n_tok, f) == (size_t) n_tok);
    }
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

    enforce_cap();
}

// ---- probe/admit (increment 2) ----

// read one bank file's header + tokens; leaves fp positioned at main_sz on success
static bool bank_read_head(FILE * f, std::string & ident, std::vector<llama_token> & toks) {
    char magic[4];
    uint32_t ver = 0, id_len = 0;
    uint64_t n_tok = 0;

    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "KVBK", 4) != 0) return false;
    if (fread(&ver, sizeof(ver), 1, f) != 1 || ver != 1)             return false;
    if (fread(&id_len, sizeof(id_len), 1, f) != 1)                   return false;

    ident.resize(id_len);
    if (id_len && fread(&ident[0], 1, id_len, f) != id_len)          return false;
    if (fread(&n_tok, sizeof(n_tok), 1, f) != 1)                     return false;
    if (n_tok > (1ull << 32))                                        return false; // sanity

    toks.resize((size_t) n_tok);
    if (n_tok && fread(toks.data(), sizeof(llama_token), (size_t) n_tok, f) != (size_t) n_tok) return false;

    return true;
}

bool server_kv_bank::probe(const server_tokens & tokens_new, const std::string & identity_cur,
                           server_prompt_cache_state & out) {
    if (!active()) {
        return false;
    }

    DIR * d = opendir(dir.c_str());
    if (d == nullptr) {
        return false;
    }

    // pick the file whose stored tokens share the longest prefix with the incoming prompt,
    // subject to the salvage floor (>= 1/8 of the new prompt, per the design)
    std::string best_path;
    size_t      best_lcp = 0;

    const llama_tokens new_toks = tokens_new.get_text_tokens();
    const size_t floor_lcp = new_toks.size() / 8;

    for (dirent * e = readdir(d); e != nullptr; e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() < 4 || name.compare(name.size() - 3, 3, ".kv") != 0) {
            continue;
        }

        const std::string path = dir + "/" + name;
        FILE * f = fopen(path.c_str(), "rb");
        if (f == nullptr) continue;

        std::string ident;
        std::vector<llama_token> toks;
        const bool head_ok = bank_read_head(f, ident, toks);
        fclose(f);

        if (!head_ok) continue;
        if (!ident.empty() && !identity_cur.empty() && ident != identity_cur) {
            continue; // never feed a state from a different model/build
        }

        size_t lcp = 0;
        while (lcp < toks.size() && lcp < new_toks.size() && toks[lcp] == new_toks[lcp]) {
            lcp++;
        }

        if (lcp > best_lcp) {
            best_lcp  = lcp;
            best_path = path;
        }
    }
    closedir(d);

    if (best_path.empty() || best_lcp < floor_lcp || best_lcp == 0) {
        n_probe_miss++;
        return false;
    }

    // ★ ECONOMICS GATE -- the difference between a cache and a liability.
    // Measured: on qwen3-4b IQ4_KT an unconditional admit made the warm request 1.64x
    // SLOWER than the cold one, because reading 1.59 GiB of KV back costs roughly what
    // recomputing 11K tokens costs on a 4B model at 4 bits. The SAME restore is a large win
    // on a 600B model, where prefill per token is orders of magnitude dearer and the KV per
    // token is not. So this compares two MEASURED rates rather than assuming either.
    //
    // Until both rates are known the bank admits anyway and learns from the result: one
    // possibly-slow request buys the measurement that protects every later one.
    // DS4P_BANK_FORCE=1 skips the gate, for A/B arms that need the admit to happen.
    {
        const char * force  = getenv("DS4P_BANK_FORCE");
        const bool   forced = force != nullptr && atoi(force) != 0;

        struct stat sb;
        if (!forced && ewma_read_mib_per_ms > 0.0 && ewma_prefill_ms_per_tok > 0.0 &&
            stat(best_path.c_str(), &sb) == 0) {
            const double restore_ms = ((double) sb.st_size / (1024.0 * 1024.0)) / ewma_read_mib_per_ms;
            const double prefill_ms = (double) best_lcp * ewma_prefill_ms_per_tok;

            if (restore_ms >= prefill_ms) {
                n_uneconomic++;
                SRV_INF(" - kv-bank: DECLINING %s -- restoring costs ~%.0f ms but prefilling those "
                        "%zu tokens costs ~%.0f ms (declines = %llu)\n",
                        best_path.c_str(), restore_ms, (size_t) best_lcp, prefill_ms,
                        (unsigned long long) n_uneconomic);
                return false;
            }
        }
    }

    const int64_t t_read_start = ggml_time_us();

    // rebuild the entry
    FILE * f = fopen(best_path.c_str(), "rb");
    if (f == nullptr) { n_probe_miss++; return false; }

    std::string ident;
    std::vector<llama_token> toks;
    if (!bank_read_head(f, ident, toks)) { fclose(f); n_probe_miss++; return false; }

    uint64_t main_sz = 0, drft_sz = 0;
    bool ok = fread(&main_sz, sizeof(main_sz), 1, f) == 1;
    if (ok) {
        out.data.main.resize((size_t) main_sz);
        ok = main_sz == 0 || fread(out.data.main.data(), 1, (size_t) main_sz, f) == (size_t) main_sz;
    }
    if (ok && fread(&drft_sz, sizeof(drft_sz), 1, f) == 1 && drft_sz > 0) {
        out.data.drft.resize((size_t) drft_sz);
        ok = fread(out.data.drft.data(), 1, (size_t) drft_sz, f) == (size_t) drft_sz;
    }
    fclose(f);

    if (!ok) {
        SRV_WRN(" - kv-bank: short read on %s, ignoring\n", best_path.c_str());
        out.data.main.clear();
        out.data.drft.clear();
        n_probe_miss++;
        return false;
    }

    out.prompt.tokens.insert(toks); // default-constructed; copy-assign is deleted by design

    // identity was checked against identity_cur above; the v1 file carries no payload
    // hashes, so admit UNSEALED (empty identity = reval skipped) rather than sealed with
    // zero hashes, which P0-2's hash check would rightly fail
    out.binding_identity.clear();

    // LRU touch: reads do not bump mtime, so refresh it explicitly
    {
        struct stat st;
        if (stat(best_path.c_str(), &st) == 0) {
            utimes(best_path.c_str(), nullptr);
        }
    }

    // measured read bandwidth feeds the gate above -- the estimate a later request is
    // declined on is this request's own observation, not a constant anybody chose
    {
        const double ms  = (ggml_time_us() - t_read_start) / 1000.0;
        const double mib = (double) (out.data.main.size() + out.data.drft.size()) / (1024.0 * 1024.0);
        if (ms > 0.0 && mib > 0.0) {
            const double rate = mib / ms;
            ewma_read_mib_per_ms = ewma_read_mib_per_ms > 0.0
                ? 0.7 * ewma_read_mib_per_ms + 0.3 * rate
                : rate;
        }
    }

    n_admitted++;
    SRV_INF(" - kv-bank: admitted %s (lcp = %zu of %zu new tokens, %.3f MiB, admits = %llu)\n",
            best_path.c_str(), best_lcp, new_toks.size(),
            (main_sz + drft_sz) / (1024.0 * 1024.0), (unsigned long long) n_admitted);
    return true;
}

// P1-5 ECONOMICS: the prefill side of the comparison. Only genuinely cold prefills are
// admissible evidence -- see the header for why a warm request must never be counted here.
void server_kv_bank::note_prefill(size_t n_tokens, double ms) {
    // a handful of tokens is dominated by fixed per-request overhead and says nothing about
    // the per-token rate the gate needs
    if (n_tokens < 256 || ms <= 0.0) {
        return;
    }

    const double rate = ms / (double) n_tokens;

    ewma_prefill_ms_per_tok = ewma_prefill_ms_per_tok > 0.0
        ? 0.7 * ewma_prefill_ms_per_tok + 0.3 * rate
        : rate;
}
