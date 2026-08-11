#pragma once

// ★ THE champion capability predicate -- ONE copy.
//
// This contract used to live in four places that had to move in lockstep: the metal-side
// hd_ok refusal chain, the vec pipeline builder's implicit geometry, the graph-side
// champ_geometry (llama-graph.cpp), and the relaxation/abort predicate beside the
// staged-tile bound. Admitting D=512 in one copy and not another produced a "champion"
// A/B in which the champion never dispatched -- admission wider than service, silently
// (2026-08-12). The head-dim set and the core geometry test now live HERE; the remaining
// site-specific conditions (n_seq from the dispatch, sinks, partials plumbing) stay at
// their sites because only those sites can see them, and each says so.
//
// Adding a head dim: instantiate the metal templates (champ + champ_vec), verify the smem
// arithmetic (28,672 B at D=512 against the 32,768 budget -- the validated formula), THEN
// widen this list. The list is the LAST thing to move, never the first.

static inline bool ggml_paged_champ_head_dim_ok(int head_dim) {
    return head_dim == 64 || head_dim == 96 || head_dim == 128 ||
           head_dim == 192 || head_dim == 256 || head_dim == 512;
}

// Core serve geometry shared by the graph-side admission and the metal-side refusal.
// kv_is_f16: every champion instantiation is f16 K/V today (q8_0 needs the staging term
// and nsg <= 4 at D=512 -- not instantiated).
static inline bool ggml_paged_champ_geometry_ok(int block_size, int head_dim, bool kv_is_f16) {
    return block_size == 64 && kv_is_f16 && ggml_paged_champ_head_dim_ok(head_dim);
}
