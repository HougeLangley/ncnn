// gemm_repro.cpp -- minimal reproducer for a RISC-V ncnn crash:
//
//   A bare (non-Net) Gemm layer driven with an fp32 input, on a ZVFH-capable
//   RISC-V CPU with the default options (use_fp16_storage = true), packs BT_data
//   as fp16 in create_pipeline() but then reads it as fp32 in forward()
//   -> out-of-bounds read -> SIGSEGV.
//
// Build (SpacemiT vendor GCC 17; the rpath is required because the vendor
// toolchain's libgomp provides GOMP_6.0.2 which the system libgomp lacks):
//
//   /opt/riscv-spacemit/bin/riscv64-linux-gnu-g++ -O3 -g -rdynamic -std=c++17 \
//     -fopenmp -march=rv64gcv_zfh_zvfh -D__fp16=_Float16 \
//     -I ncnn/src -I ncnn/build/src \
//     gemm_repro.cpp ncnn/build/src/libncnn.a \
//     -L/opt/riscv-spacemit/lib -Wl,-rpath,/opt/riscv-spacemit/lib -fopenmp \
//     -o gemm_repro
//
// Run on the A100 cluster (VLEN=1024):
//   ai-run taskset -c 8-15 ./gemm_repro 1 80000 1024 8      # SIGSEGV
//   NO_FP16_STORAGE=1 ai-run taskset -c 8-15 ./gemm_repro 1 80000 1024 8   # OK

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <ctime>

#include "layer.h"
#include "mat.h"
#include "option.h"
#include "paramdict.h"
#include "modelbin.h"
#include "cpu.h"

static void segv_handler(int sig)
{
    void* bt[64];
    int n = backtrace(bt, 64);
    fprintf(stderr, "\n*** SIGSEGV (signal %d) -- backtrace (%d frames) ***\n", sig, n);
    backtrace_symbols_fd(bt, n, 2);
    _exit(139);
}

int main(int argc, char** argv)
{
    signal(SIGSEGV, segv_handler);

    const int M = argc > 1 ? atoi(argv[1]) : 1;
    const int N = argc > 2 ? atoi(argv[2]) : 80000;
    const int K = argc > 3 ? atoi(argv[3]) : 1024;
    const int threads = argc > 4 ? atoi(argv[4]) : 8;

    ncnn::Option opt;
    opt.num_threads = threads;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = getenv("NO_FP16_STORAGE") ? false : true; // <-- the switch
    opt.use_fp16_arithmetic = true;                                  // ncnn default
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_vulkan_compute = false;

    // NAIVE=1 uses ncnn's own reference implementation as ground truth.
    const bool naive = getenv("NAIVE") != nullptr;
    ncnn::Layer* layer = naive ? ncnn::create_layer_naive("Gemm") : ncnn::create_layer("Gemm"); // -> Gemm_riscv via runtime dispatch
    if (!layer)
    {
        fprintf(stderr, "create_layer failed\n");
        return 2;
    }
    fprintf(stderr, "cpu: riscv_v=%d zfh=%d zvfh=%d | Gemm support_fp16_storage=%d | opt.use_fp16_storage=%d\n",
            ncnn::cpu_support_riscv_v(), ncnn::cpu_support_riscv_zfh(), ncnn::cpu_support_riscv_zvfh(),
            (int)layer->support_fp16_storage, (int)opt.use_fp16_storage);

    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 0.f); // beta = 0 -> no C term
    pd.set(2, 0);   // transA = 0
    pd.set(3, 1);   // transB = 1
    pd.set(4, 0);   // constantA = 0
    pd.set(5, 1);   // constantB = 1
    pd.set(6, 0);   // constantC = 0
    pd.set(7, 0);   // constantM
    pd.set(8, N);   // constantN
    pd.set(9, K);   // constantK
    pd.set(10, 0);  // constant_broadcast_type_C
    pd.set(11, 0);  // output_N1M
    pd.set(12, 0);  // output_elempack
    pd.set(13, 0);  // output_elemtype
    pd.set(14, 0);  // output_transpose
    if (layer->load_param(pd) != 0)
    {
        fprintf(stderr, "load_param failed\n");
        return 2;
    }

    // transB=1: Gemm::load_model does mb.load(constantK, constantN), so B_data
    // must be Mat(w=K, h=N) -- each row j holds the K weights of output j.
    ncnn::Mat B(K, N);
    if (B.empty())
    {
        fprintf(stderr, "weight allocation failed\n");
        return 2;
    }
    for (size_t i = 0; i < (size_t)N * K; i++) ((float*)B)[i] = 0.001f;

    ncnn::Mat weights[2];
    weights[0] = B;
    weights[1] = ncnn::Mat(); // terminator
    ncnn::ModelBinFromMatArray mb(weights);
    if (layer->load_model(mb) != 0)
    {
        fprintf(stderr, "load_model failed\n");
        return 2;
    }
    fprintf(stderr, "create_pipeline (packs BT_data) ...\n");
    if (layer->create_pipeline(opt) != 0)
    {
        fprintf(stderr, "create_pipeline failed\n");
        return 2;
    }

    ncnn::Mat in(M, K); // fp32 input, as a bare layer would receive
    for (size_t i = 0; i < (size_t)M * K; i++) ((float*)in)[i] = 0.01f;

    // ---- in-program reference: out[j] = sum_k A[0][k] * B[j][k] (transB=1) ----
    std::vector<float> ref(N, 0.f);
    {
        const float* a0 = in;              // in is [w=K, h=M]
        const float* b = B;                // B is [w=N, h=K]
        for (int j = 0; j < N; j++)
        {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)a0[k] * (double)b[(size_t)j * K + k];
            ref[j] = (float)acc;
        }
    }

    std::vector<ncnn::Mat> bottoms(1), tops(1);
    ncnn::Mat in16;
    if (getenv("FP16_IN"))   // simulate what Net does: cast the input to fp16
    {
        ncnn::cast_float32_to_float16(in, in16, opt);
        bottoms[0] = in16;
    }
    else
    {
        bottoms[0] = in;
    }
    fprintf(stderr, "calling forward() with an fp32 input (elembits=32) ...\n");
    // benchmark mode: time the steady-state forward() calls
    const char* benchenv = getenv("BENCH");
    if (benchenv)
    {
        const int bn = atoi(benchenv);
        double best = 1e30;
        for (int r = 0; r < bn; r++)
        {
            std::vector<ncnn::Mat> bt(1), tp(1);
            bt[0] = bottoms[0];
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            int rr = layer->forward(bt, tp, opt);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
            if (rr != 0) { fprintf(stderr, "BENCH_ERR ret=%d\n", rr); break; }
            if (ms < best) best = ms;
        }
        double fl = 2.0 * (double)M * (double)N * (double)K;
        printf("BENCH M=%d N=%d K=%d fp16in=%d  ms=%.4f  GFLOPs=%.2f\n",
               M, N, K, getenv("FP16_IN") ? 1 : 0, best, fl / (best / 1e3) / 1e9);
        layer->destroy_pipeline(opt);
        delete layer;
        return 0;
    }

    int fret = layer->forward(bottoms, tops, opt);
    fprintf(stderr, "forward() returned %d\n", fret);
    if (fret != 0 || tops[0].empty())
    {
        // the layer refused the call (expected with the fp16/fp32 mismatch guard):
        // there is no output to inspect, and this is a clean error, not a crash.
        fprintf(stderr, "  -> clean error return, no output produced (no crash)\n");
        layer->destroy_pipeline(opt);
        delete layer;
        return 0;
    }
    // checksum of the output: with the fp16/fp32 packing mismatch the fp32 path
    // reads garbage weights, which may not fault but silently produces wrong
    // numbers. Comparing this against NO_FP16_STORAGE=1 detects that case.
    {
        const ncnn::Mat& out = tops[0];
        double sum = 0.0;
        const float* q = out;
        size_t n = (size_t)out.total();
        for (size_t i = 0; i < n; i++) sum += (double)q[i] * (double)(i % 7 + 1);
        const float* q0 = out;
        // compare the first min(total,N) elements against the in-program reference
        int cmp = 0, bad = 0;
        double maxrel = 0.0;
        for (size_t i = 0; i < out.total() && (int)i < N; i++)
        {
            cmp++;
            double r = (double)ref[i];
            double d = (r != 0.0) ? fabs((double)q0[i] - r) / fabs(r) : fabs((double)q0[i]);
            if (d > maxrel) maxrel = d;
            if (d > 1e-3) bad++;
        }
        fprintf(stderr, "forward OK  dims=%d w=%d h=%d c=%d elempack=%d elembits=%d total=%d\n",
                out.dims, out.w, out.h, out.c, out.elempack, out.elembits(), (int)out.total());
        fprintf(stderr, "  checksum=%.10g   ref[0]=%.10g  out[0]=%.10g\n", sum, ref[0], q0[0]);
        fprintf(stderr, "  vs in-program reference: compared=%d  mismatched(>1e-3 rel)=%d  max_rel_err=%.6g\n",
                cmp, bad, maxrel);
        const char* dump = getenv("DUMP");
        if (dump)
        {
            FILE* f = fopen(dump, "wb");
            if (f) { fwrite(q0, sizeof(float), out.total(), f); fclose(f); }
            fprintf(stderr, "  dumped %d floats to %s\n", (int)out.total(), dump);
        }
        fprintf(stderr, "            first 4 out values = %.10g %.10g %.10g %.10g   (analytic value = %.10g)\n",
                q0[0], q0[1], q0[2], q0[3], 0.01f * 0.001f * (float)K);
    }

    layer->destroy_pipeline(opt);
    delete layer;
    return 0;
}
