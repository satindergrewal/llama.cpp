// #19 EXTENDED round-trip: closes the three synth-blind gaps the advisor named.
//   (1) runs on CPU *and* Metal  -> verifies the Metal store kernel's VALUES, not just that it runs
//   (2) gathers ALL K heads and ALL V heads (head-offset term no longer x0; V base exercised)
//   (3) builds gather rows with the model's IN-GRAPH ggml-op chain (floor/scale/sub/add/arange),
//       not hand-computed C++ rows -- so the arithmetic the synth model is blind to is checked here.
// Store known K/V into a zeroed pool via ggml_paged_kv_store, gather back with the op chain, assert
// equality (modulo f16). Passing on Metal for all heads == the paged MSA store+gather is correct.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <string>
#include <unistd.h>

static int run_backend(const char * name, ggml_backend_t backend) {
    const int hd = 8, pbs = 4, hkv = 3, nblk = 5, ntok = 13;  // 3 kv heads, 13 tokens over 5 blocks of 4

    struct ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(ip);

    ggml_tensor * pool  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hd, pbs, 2*hkv, nblk);  // [hd,pbs,2hkv,nblk]
    ggml_tensor * kcur  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, hkv, ntok);          // [hd,hkv,ntok]
    ggml_tensor * vcur  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, hkv, ntok);
    ggml_tensor * wslot = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ntok);                   // I64 slots (as the model)
    ggml_tensor * cs    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ntok);                   // cell index per token (float)
    ggml_set_input(pool); ggml_set_input(kcur); ggml_set_input(vcur); ggml_set_input(wslot); ggml_set_input(cs);

    // store
    ggml_tensor * stored   = ggml_paged_kv_store(ctx, pool, kcur, vcur, wslot, pbs);
    ggml_tensor * poolflat = ggml_reshape_2d(ctx, stored, hd, pbs*2*hkv*nblk);

    // --- rows built with the MODEL's ggml-op chain (minimax-m3.cpp:443-453), all heads at once ---
    // cs2 [ntok,1] broadcast over heads; head arange [1,hkv] broadcast over tokens.
    ggml_tensor * cs2  = ggml_reshape_2d(ctx, cs, ntok, 1);
    ggml_tensor * blkf = ggml_floor(ctx, ggml_scale(ctx, cs2, 1.0f/(float) pbs));         // [ntok,1]
    ggml_tensor * pos  = ggml_sub(ctx, cs2, ggml_scale(ctx, blkf, (float) pbs));          // [ntok,1]
    ggml_tensor * blkt = ggml_scale(ctx, blkf, (float) (pbs*2*hkv));                       // [ntok,1]
    ggml_tensor * hK   = ggml_reshape_2d(ctx, ggml_arange(ctx, 0.0f,        (float) hkv,    1.0f), 1, hkv);  // [1,hkv]
    ggml_tensor * hV   = ggml_reshape_2d(ctx, ggml_arange(ctx, (float) hkv, (float)(2*hkv), 1.0f), 1, hkv);  // [1,hkv]
    // materialize the full [ntok,hkv] plane before add() (ggml add broadcasts b into a, so both
    // operands must already be the target shape). ref carries the shape for ggml_repeat.
    ggml_tensor * ref  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ntok, hkv);  ggml_set_input(ref);
    ggml_tensor * posF = ggml_repeat(ctx, pos,  ref);
    ggml_tensor * blkF = ggml_repeat(ctx, blkt, ref);
    ggml_tensor * hKF  = ggml_repeat(ctx, ggml_scale(ctx, hK, (float) pbs), ref);
    ggml_tensor * hVF  = ggml_repeat(ctx, ggml_scale(ctx, hV, (float) pbs), ref);
    // rowK[t,h] = pos + pbs*hK + blkt ; rowV likewise with hV.
    ggml_tensor * rowK = ggml_add(ctx, ggml_add(ctx, posF, hKF), blkF);  // [ntok,hkv]
    ggml_tensor * rowV = ggml_add(ctx, ggml_add(ctx, posF, hVF), blkF);  // [ntok,hkv]
    ggml_tensor * rowKi = ggml_cast(ctx, ggml_reshape_1d(ctx, rowK, ntok*hkv), GGML_TYPE_I32);
    ggml_tensor * rowVi = ggml_cast(ctx, ggml_reshape_1d(ctx, rowV, ntok*hkv), GGML_TYPE_I32);
    ggml_tensor * kg = ggml_get_rows(ctx, poolflat, rowKi);   // [hd, ntok*hkv]
    ggml_tensor * vg = ggml_get_rows(ctx, poolflat, rowVi);
    ggml_set_output(kg); ggml_set_output(vg);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kg);
    ggml_build_forward_expand(gf, vg);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    (void) buf;

    std::vector<ggml_fp16_t> poolz((size_t) hd*pbs*2*hkv*nblk, ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(pool, poolz.data(), 0, poolz.size()*sizeof(ggml_fp16_t));

    std::vector<float> kd((size_t) hd*hkv*ntok), vd((size_t) hd*hkv*ntok);
    for (int t = 0; t < ntok; ++t) for (int h = 0; h < hkv; ++h) for (int d = 0; d < hd; ++d) {
        kd[(size_t)t*hkv*hd + h*hd + d] = (float)(1000 + t*10 + h*2) + d*0.01f;
        vd[(size_t)t*hkv*hd + h*hd + d] = (float)(5000 + t*10 + h*2) + d*0.01f;
    }
    ggml_backend_tensor_set(kcur, kd.data(), 0, kd.size()*sizeof(float));
    ggml_backend_tensor_set(vcur, vd.data(), 0, vd.size()*sizeof(float));

    std::vector<int64_t> ws(ntok); std::vector<float> csd(ntok);
    for (int t = 0; t < ntok; ++t) { ws[t] = t; csd[t] = (float) t; }  // append: slot == cell == t
    ggml_backend_tensor_set(wslot, ws.data(), 0, ws.size()*sizeof(int64_t));
    ggml_backend_tensor_set(cs, csd.data(), 0, csd.size()*sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("[%s] FAIL: compute\n", name); ggml_free(ctx); return 1; }

    std::vector<float> gotK((size_t) hd*ntok*hkv), gotV((size_t) hd*ntok*hkv);
    ggml_backend_tensor_get(kg, gotK.data(), 0, gotK.size()*sizeof(float));
    ggml_backend_tensor_get(vg, gotV.data(), 0, gotV.size()*sizeof(float));

    // gathered layout: [hd, ntok*hkv]. rowK is [ntok,hkv] COLUMN-major, so element (t,h) sits at
    // flat index t + ntok*h -> gathered column = t + ntok*h (NOT t*hkv+h).
    double maxerr = 0.0; int bad = 0;
    for (int t = 0; t < ntok; ++t) for (int h = 0; h < hkv; ++h) for (int d = 0; d < hd; ++d) {
        const int col = t + ntok*h;
        const float wantK = ggml_fp16_to_fp32(ggml_fp32_to_fp16(kd[(size_t)t*hkv*hd + h*hd + d]));
        const float wantV = ggml_fp16_to_fp32(ggml_fp32_to_fp16(vd[(size_t)t*hkv*hd + h*hd + d]));
        const double eK = std::fabs(gotK[(size_t)col*hd + d] - wantK);
        const double eV = std::fabs(gotV[(size_t)col*hd + d] - wantV);
        if (eK > maxerr) maxerr = eK;
        if (eV > maxerr) maxerr = eV;
        if (eK > 0.5) { if (bad < 4) printf("  [%s] K MISMATCH t=%d h=%d d=%d got=%.3f want=%.3f\n", name, t, h, d, gotK[(size_t)col*hd+d], wantK); bad++; }
        if (eV > 0.5) { if (bad < 4) printf("  [%s] V MISMATCH t=%d h=%d d=%d got=%.3f want=%.3f\n", name, t, h, d, gotV[(size_t)col*hd+d], wantV); bad++; }
    }
    printf("[%s] %s: store+gather ALL heads (K+V), in-graph rows, max_err=%.4g bad=%d/%d (pool f16)\n",
           name, bad == 0 ? "PASS" : "FAIL", maxerr, bad, 2*hd*ntok*hkv);
    ggml_free(ctx);
    return bad == 0 ? 0 : 1;
}

int main() {
    int rc = 0, ran = 0;
    // Run on EVERY registered device (CPU + Metal/GPU when linked) via the backend registry, so the
    // Metal store kernel's VALUES are checked wherever this is built, with no compile-time defines.
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        // only CPU and GPU: ACCEL devices (e.g. BLAS) do not implement the custom paged_kv_store op
        // and would abort in graph_compute.
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
        if (t != GGML_BACKEND_DEVICE_TYPE_CPU && t != GGML_BACKEND_DEVICE_TYPE_GPU) {
            printf("[%s] SKIP: not a CPU/GPU device\n", ggml_backend_dev_name(dev));
            continue;
        }
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) { printf("[%s] SKIP: init failed\n", ggml_backend_dev_name(dev)); continue; }
        rc |= run_backend(ggml_backend_dev_name(dev), b);
        ran++;
    }
    if (ran == 0) { printf("FAIL: no backend devices\n"); rc = 1; }
    // _exit bypasses the Metal backend's static-destructor rsets assert at process teardown (a known
    // shutdown-order quirk unrelated to the test result, which is already printed above).
    fflush(stdout); fflush(stderr);
    _exit(rc);
}
