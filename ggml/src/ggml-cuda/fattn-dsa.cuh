#pragma once

#include "common.cuh"

// Sparse ("DSA") flash attention: computes attention over only the top-k K/V rows
// selected by the lightning indexer, by gathering those rows into compact buffers,
// instead of running a full-size KQ and masking the unselected columns away.
//
// The CUDA kernels behind this entry point are transplanted from ik_llama.cpp,
//   ggml/src/ggml-cuda/dsa_attn.cu -- Copyright (C) 2024 Iwan Kawrakow, MIT license.

// Returns false when the tensor configuration is not handled by the gathered path.
bool ggml_cuda_flash_attn_ext_dsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_dsa_supported(int device, const ggml_tensor * dst);
