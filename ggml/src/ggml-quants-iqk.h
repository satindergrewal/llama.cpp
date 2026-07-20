#pragma once

// IQK quantization types (IQ4_K, IQ5_KS, IQ6_K and the IQ*_KT trellis family)
//
// These quantization formats were designed and first implemented by
// Iwan Kawrakow (@ikawrakow) in ik_llama.cpp (https://github.com/ikawrakow/ik_llama.cpp).
// The block layouts, type numbers and codebooks are kept bit-exact with the
// original implementation so that GGUF files are interchangeable between the
// two projects.
//
// The IQ*_KT types and IQ5_KS store a per-row header (currently a single f32
// scale, see ggml_type_traits.row_meta_size) in front of the block data.
// A row therefore is: [row header][block 0][block 1]...

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they are used by the CPU backend

// Quantization (reference; also used as from_float for the CPU backend)
GGML_API void quantize_row_iq4_k_ref (const float * GGML_RESTRICT x, block_iq4_k  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq5_ks_ref(const float * GGML_RESTRICT x, block_iq5_ks * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq6_k_ref (const float * GGML_RESTRICT x, block_iq6_k  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, block_iq1_kt * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, block_iq3_kt * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq4_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq5_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq6_k (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq1_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_iq4_k (const block_iq4_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq5_ks(const block_iq5_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq6_k (const block_iq6_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_kt(const block_iq1_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_kt(const block_iq2_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_kt(const block_iq3_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_kt(const block_iq4_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization with importance matrix (called by ggml_quantize_chunk)
GGML_API size_t quantize_iq4_k (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq5_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq6_k (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// Reference dot products against Q8_K (dequantize + multiply; correctness path,
// not performance - optimized kernels are added separately)
GGML_API void ggml_vec_dot_iq4_k_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq5_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq6_k_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq1_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq2_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq3_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void ggml_vec_dot_iq4_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
