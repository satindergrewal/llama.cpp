// llama-dspark-capture: run a (possibly quantized) GGUF target over pre-tokenized
// sequences and dump the per-layer hidden states DeepSpec-style drafter training
// consumes (target_hidden_states at the configured extract layers + post-final-norm
// last hidden states), as bf16 raw blocks plus a JSONL offset sidecar.
//
// This enables training DSpark/DFlash heads matched to the exact deployed artifact
// (quantized GGUF) instead of the full-precision HF checkpoint.
//
// Layer id convention: --layers takes llama.cpp extract ids, i.e. the layer whose
// INPUT is captured. HF-side DeepSpec captures the OUTPUT of decoder layer L, which
// equals the input of layer L+1 here — same +1 mapping the GGUF head converter uses
// for {arch}.target_layers.
//
// Input token file (little-endian):
//   magic u32 'DSCT' (0x54435344), version u32 = 1, n_samples u32
//   per sample: sample_id u64, seq_len u32, ids i32[seq_len], loss_mask u8[seq_len]
// (loss_mask is passed through untouched; it is not needed for the forward but is
//  kept in one file so the tokenizer step remains the single source of truth.)
//
// Output: <out>/capture.bin (bf16 blocks) and <out>/offsets.jsonl with per-sample
// byte offsets, consumed by the DeepSpec-side assembler that writes the canonical
// target-cache shards/index/manifest.

#include "llama.h"
#include "../../src/llama-ext.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    if ((x & 0x7fffffff) > 0x7f800000) { // NaN: keep quiet NaN
        return (uint16_t) ((x >> 16) | 0x0040);
    }
    x += 0x7fff + ((x >> 16) & 1); // round to nearest even
    return (uint16_t) (x >> 16);
}

struct sample_t {
    uint64_t id;
    std::vector<int32_t> ids;
    std::vector<uint8_t> loss_mask;
};

static bool read_samples(const char * path, std::vector<sample_t> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return false;
    }
    uint32_t magic = 0, version = 0, n = 0;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x54435344u ||
        fread(&version, 4, 1, f) != 1 || version != 1u ||
        fread(&n, 4, 1, f) != 1) {
        fprintf(stderr, "error: bad token file header in %s\n", path);
        fclose(f);
        return false;
    }
    out.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t id = 0; uint32_t len = 0;
        if (fread(&id, 8, 1, f) != 1 || fread(&len, 4, 1, f) != 1) {
            fprintf(stderr, "error: truncated sample header at %u\n", i);
            fclose(f);
            return false;
        }
        out[i].id = id;
        out[i].ids.resize(len);
        out[i].loss_mask.resize(len);
        if (fread(out[i].ids.data(), 4, len, f) != len ||
            fread(out[i].loss_mask.data(), 1, len, f) != len) {
            fprintf(stderr, "error: truncated sample %u\n", i);
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * in_path    = nullptr;
    const char * out_dir    = nullptr;
    std::vector<uint32_t> layers; // llama.cpp extract ids (HF layer id + 1)
    int n_gpu_layers = 999;
    int n_ctx        = 2048;
    int n_ubatch     = 512;
    bool flash_attn  = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", flag); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")        model_path = need("-m");
        else if (a == "--in")      in_path    = need("--in");
        else if (a == "--out")     out_dir    = need("--out");
        else if (a == "-ngl")      n_gpu_layers = atoi(need("-ngl"));
        else if (a == "-c")        n_ctx      = atoi(need("-c"));
        else if (a == "-ub")       n_ubatch   = atoi(need("-ub"));
        else if (a == "--no-fa")   flash_attn = false;
        else if (a == "--layers") {
            std::string s = need("--layers");
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                layers.push_back((uint32_t) atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        } else {
            fprintf(stderr, "usage: %s -m model.gguf --in tokens.bin --out dir --layers 2,7,12,17,22 [-ngl N] [-c N] [-ub N] [--no-fa]\n", argv[0]);
            return 1;
        }
    }
    if (!model_path || !in_path || !out_dir || layers.empty()) {
        fprintf(stderr, "error: -m, --in, --out and --layers are required\n");
        return 1;
    }

    std::vector<sample_t> samples;
    if (!read_samples(in_path, samples)) {
        return 1;
    }
    fprintf(stderr, "%s: %zu samples loaded from %s\n", __func__, samples.size(), in_path);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load %s\n", model_path);
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = n_ctx;
    cparams.n_batch      = n_ctx;   // whole sequence in one llama_decode call
    cparams.n_ubatch     = n_ubatch;
    cparams.embeddings   = true;    // post-final-norm hidden states per token
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    for (uint32_t lid : layers) {
        llama_set_embeddings_layer_inp(ctx, lid, true);
    }

    std::string cap_path = std::string(out_dir) + "/capture.bin";
    std::string off_path = std::string(out_dir) + "/offsets.jsonl";
    FILE * fcap = fopen(cap_path.c_str(), "wb");
    FILE * foff = fopen(off_path.c_str(), "w");
    if (!fcap || !foff) {
        fprintf(stderr, "error: cannot open outputs in %s\n", out_dir);
        return 1;
    }

    const uint32_t n_cap = (uint32_t) layers.size();
    std::vector<uint16_t> row_bf16;
    uint64_t written = 0;

    for (size_t si = 0; si < samples.size(); ++si) {
        const auto & s = samples[si];
        const uint32_t len = (uint32_t) s.ids.size();
        if (len == 0 || (int) len > n_ctx) {
            fprintf(stderr, "warn: sample %" PRIu64 " len %u out of range, skipped\n", s.id, len);
            fprintf(foff, "{\"sample_id\":%" PRIu64 ",\"skipped\":true}\n", s.id);
            continue;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(len, 0, 1);
        batch.n_tokens = len;
        for (uint32_t i = 0; i < len; ++i) {
            batch.token[i]     = s.ids[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1; // every token is an output row (embeddings for all positions)
        }

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            fprintf(stderr, "error: llama_decode failed (%d) on sample %" PRIu64 "\n", rc, s.id);
            fprintf(foff, "{\"sample_id\":%" PRIu64 ",\"skipped\":true}\n", s.id);
            continue;
        }

        // target_hidden_states: per token, concat of captured layers ([len, n_cap*n_embd])
        const uint64_t hidden_offset = (uint64_t) ftell(fcap);
        std::vector<const float *> caps(n_cap);
        for (uint32_t k = 0; k < n_cap; ++k) {
            caps[k] = llama_get_embeddings_layer_inp(ctx, layers[k]);
            if (!caps[k]) {
                fprintf(stderr, "error: null capture for layer %u\n", layers[k]);
                return 1;
            }
        }
        row_bf16.resize((size_t) n_cap * n_embd);
        for (uint32_t t = 0; t < len; ++t) {
            for (uint32_t k = 0; k < n_cap; ++k) {
                const float * src = caps[k] + (size_t) t * n_embd;
                uint16_t * dst = row_bf16.data() + (size_t) k * n_embd;
                for (int j = 0; j < n_embd; ++j) {
                    dst[j] = f32_to_bf16(src[j]);
                }
            }
            fwrite(row_bf16.data(), 2, row_bf16.size(), fcap);
        }

        // target_last_hidden_states: post-final-norm per token ([len, n_embd])
        const uint64_t last_offset = (uint64_t) ftell(fcap);
        row_bf16.resize(n_embd);
        for (uint32_t t = 0; t < len; ++t) {
            const float * emb = llama_get_embeddings_ith(ctx, (int32_t) t);
            if (!emb) {
                fprintf(stderr, "error: null embeddings row %u on sample %" PRIu64 "\n", t, s.id);
                return 1;
            }
            for (int j = 0; j < n_embd; ++j) {
                row_bf16[j] = f32_to_bf16(emb[j]);
            }
            fwrite(row_bf16.data(), 2, row_bf16.size(), fcap);
        }

        fprintf(foff, "{\"sample_id\":%" PRIu64 ",\"seq_len\":%u,\"hidden_offset\":%" PRIu64 ",\"last_offset\":%" PRIu64 "}\n",
                s.id, len, hidden_offset, last_offset);
        ++written;
        if (written % 100 == 0 || si + 1 == samples.size()) {
            fprintf(stderr, "%s: %" PRIu64 "/%zu samples captured\n", __func__, written, samples.size());
            fflush(foff);
        }
    }

    fclose(fcap);
    fclose(foff);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "done: %" PRIu64 " samples -> %s\n", written, cap_path.c_str());
    return 0;
}
