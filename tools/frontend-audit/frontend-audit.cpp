// llama-frontend-audit
//
// Reproduces, off-line and CPU-only, the exact prompt-construction path that
// llama-server uses for /v1/chat/completions:
//
//   request body -> oaicompat_chat_params_parse() -> data["prompt"]
//                -> tokenize_input_prompts(vocab, nullptr, prompt, true, true)
//
// The model is loaded with vocab_only=true so this costs no VRAM and does not
// touch the tensor data.  Output is JSON: rendered prompt string plus the exact
// token ids and per-token pieces, so an external reference renderer/tokenizer
// can be diffed token-for-token.

#include "common.h"
#include "log.h"
#include "llama.h"
#include "chat.h"

#include "server-common.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
        "usage: %s --model <gguf> --corpus <json> --out <json>\n"
        "          [--no-jinja] [--reasoning-format none|auto|deepseek]\n"
        "          [--chat-template-file <file>] [--no-parse-special]\n"
        "          [--no-add-special] [--template-kwarg k=v]...\n"
        "\n"
        "  --corpus  JSON array of { \"name\": ..., \"body\": <oai chat body> }\n"
        "            or of raw oai chat bodies.\n", prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string corpus_path;
    std::string out_path;
    std::string tmpl_file;
    bool use_jinja      = true;
    bool parse_special  = true;
    bool add_special    = true;
    bool emit_pieces    = true;
    common_reasoning_format rformat = COMMON_REASONING_FORMAT_NONE;
    std::map<std::string, std::string> tmpl_kwargs;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](void) -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return std::string(argv[++i]);
        };
        if      (a == "--model")              model_path  = next();
        else if (a == "--corpus")             corpus_path = next();
        else if (a == "--out")                out_path    = next();
        else if (a == "--chat-template-file") tmpl_file   = next();
        else if (a == "--no-jinja")           use_jinja   = false;
        else if (a == "--no-parse-special")   parse_special = false;
        else if (a == "--no-add-special")     add_special = false;
        else if (a == "--no-pieces")          emit_pieces = false;
        else if (a == "--reasoning-format") {
            std::string v = next();
            if      (v == "none")     rformat = COMMON_REASONING_FORMAT_NONE;
            else if (v == "auto")     rformat = COMMON_REASONING_FORMAT_AUTO;
            else { fprintf(stderr, "unknown reasoning-format %s\n", v.c_str()); return 1; }
        }
        else if (a == "--template-kwarg") {
            std::string kv = next();
            auto p = kv.find(61);
            if (p == std::string::npos) { usage(argv[0]); return 1; }
            tmpl_kwargs[kv.substr(0, p)] = kv.substr(p + 1);
        }
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || corpus_path.empty() || out_path.empty()) { usage(argv[0]); return 1; }

    common_log_set_verbosity_thold(-1); // quiet

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string tmpl_override;
    if (!tmpl_file.empty()) {
        std::ifstream f(tmpl_file);
        std::stringstream ss; ss << f.rdbuf();
        tmpl_override = ss.str();
    }

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, tmpl_override);

    server_chat_params chat_params;
    chat_params.use_jinja            = use_jinja;
    chat_params.prefill_assistant    = true;
    chat_params.reasoning_format     = rformat;
    chat_params.chat_template_kwargs = tmpl_kwargs;
    chat_params.tmpls                = std::move(tmpls);
    chat_params.allow_image          = false;
    chat_params.allow_audio          = false;
    chat_params.allow_video          = false;
    chat_params.enable_thinking      = true;
    chat_params.reasoning_budget     = -1;
    chat_params.force_pure_content   = false;

    std::ifstream cf(corpus_path);
    if (!cf) { fprintf(stderr, "cannot open corpus\n"); return 1; }
    json corpus = json::parse(cf);

    json results = json::array();

    for (size_t i = 0; i < corpus.size(); i++) {
        json entry = corpus[i];
        std::string name = "case_" + std::to_string(i);
        json body;
        if (entry.contains("body")) {
            name = entry.value("name", name);
            body = entry.at("body");
        } else {
            body = entry;
        }

        json rec;
        rec["name"] = name;
        try {
            std::vector<raw_buffer> files;
            json data = oaicompat_chat_params_parse(body, chat_params, files);
            std::string prompt = data.at("prompt").get<std::string>();
            rec["prompt"] = prompt;

            llama_tokens ids;
            if (data.contains("prompt_segments")) {
                rec["segments"] = data.at("prompt_segments");
                ids = tokenize_input_segments(vocab, data.at("prompt_segments"), add_special).get_text_tokens();
            } else {
                auto toks = tokenize_input_prompts(vocab, nullptr, data.at("prompt"), add_special, parse_special);
                ids = toks[0].get_text_tokens();
            }
            rec["tokens"] = ids;
            if (emit_pieces) {
                json pieces = json::array();
                for (auto t : ids) {
                    // byte-level BPE pieces are not always valid UTF-8; escape to a
                    // JSON-safe ASCII form so the audit output can always be written
                    std::string raw = common_token_to_piece(vocab, t, true);
                    std::string esc;
                    for (unsigned char c : raw) {
                        if (c >= 0x20 && c < 0x7f && c != 0x5c) { esc += (char) c; }
                        else { char buf[8]; snprintf(buf, sizeof(buf), "\\x%02X", c); esc += buf; }
                    }
                    pieces.push_back(esc);
                }
                rec["pieces"] = pieces;
            }
            rec["n_tokens"] = (int) ids.size();
            rec["ok"] = true;
        } catch (const std::exception & e) {
            rec["ok"] = false;
            rec["error"] = std::string(e.what());
        }
        results.push_back(rec);
    }

    json out;
    out["model"]          = model_path;
    out["use_jinja"]      = use_jinja;
    out["parse_special"]  = parse_special;
    out["typed_segments"] = typed_segments_enabled();
    out["add_special"]    = add_special;
    out["template_src"]   = common_chat_templates_source(chat_params.tmpls.get(), "");
    out["add_bos_token"]  = llama_vocab_get_add_bos(vocab);
    out["add_eos_token"]  = llama_vocab_get_add_eos(vocab);
    out["bos_id"]         = llama_vocab_bos(vocab);
    out["eos_id"]         = llama_vocab_eos(vocab);
    out["results"]        = results;

    std::ofstream of(out_path);
    of << out.dump(2) << std::endl;
    of.close();

    fprintf(stderr, "wrote %s (%zu cases)\n", out_path.c_str(), results.size());

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
