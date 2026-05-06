/*
 * optimized_lora.cu  –  LoRA forward: Y = W*X + A*(B^T*X)
 *
 * W, X : d x d  float32   (d in [3584, 4608])
 * A, B : d x r  float32   (r = 16)
 *
 * Optimisations
 * ─────────────
 * 1. cuBLAS with TF32 tensor-core math (CUBLAS_TF32_TENSOR_OP_MATH)
 *    and cublasGemmEx / CUBLAS_COMPUTE_32F_FAST_TF32 for the large GEMM.
 * 2. Two non-blocking CUDA streams:
 *      stream 0 : Y   = W * X            (dominant, d×d GEMM)
 *      stream 1 : BtX = B^T * X          (small, r×d GEMM, overlapped)
 *    After stream-0 signals an event, stream 1 accumulates Y += A * BtX.
 * 3. Persistent cuBLAS handles – created once, reused on every call.
 *
 * Row-major cuBLAS trick
 * ──────────────────────
 * For row-major C(m,n) = opA(A) * opB(B), we call cuBLAS as:
 *   cublasSgemm(h, opB, opA, n, m, k, α, B, ldB, A, ldA, β, C, ldC)
 * (swap A<->B, swap opA<->opB, swap m<->n).
 */

#include <torch/extension.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Persistent state: two streams + two cuBLAS handles + one CUDA event
// ---------------------------------------------------------------------------
struct State {
    cublasHandle_t h0 = nullptr;   // stream 0 (W*X)
    cublasHandle_t h1 = nullptr;   // stream 1 (LoRA branch)
    cudaStream_t   s0 = nullptr;
    cudaStream_t   s1 = nullptr;
    cudaEvent_t    ev = nullptr;   // signals "Y = W*X is done" to stream 1

    void init() {
        if (h0) return;
        cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

        cublasCreate(&h0);
        cublasCreate(&h1);
        cublasSetMathMode(h0, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetMathMode(h1, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetStream(h0, s0);
        cublasSetStream(h1, s1);
    }
};
static State g_state;

// ---------------------------------------------------------------------------
// Row-major GEMM helper:  C(m,n) = alpha * opA(A) * opB(B)  +  beta * C
// All matrices stored in row-major order.
// ---------------------------------------------------------------------------
static inline void rmm(cublasHandle_t h,
                        cublasOperation_t opA, cublasOperation_t opB,
                        int m, int n, int k,
                        float alpha,
                        const float* A, int ldA,
                        const float* B, int ldB,
                        float beta,
                        float* C, int ldC)
{
    // Swap A<->B, swap opA<->opB, swap m<->n  (standard col-major trick)
    cublasSgemm(h,
                opB, opA,
                n, m, k,
                &alpha,
                B, ldB,
                A, ldA,
                &beta,
                C, ldC);
}

// ---------------------------------------------------------------------------
// Same, using cublasGemmEx for explicit CUBLAS_COMPUTE_32F_FAST_TF32
// ---------------------------------------------------------------------------
static inline void rmm_ex(cublasHandle_t h,
                           cublasOperation_t opA, cublasOperation_t opB,
                           int m, int n, int k,
                           float alpha,
                           const float* A, int ldA,
                           const float* B, int ldB,
                           float beta,
                           float* C, int ldC)
{
    cublasGemmEx(h,
                 opB, opA,
                 n, m, k,
                 &alpha,
                 B, CUDA_R_32F, ldB,
                 A, CUDA_R_32F, ldA,
                 &beta,
                 C, CUDA_R_32F, ldC,
                 CUBLAS_COMPUTE_32F_FAST_TF32,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---------------------------------------------------------------------------
// forward:  Y = W * X  +  A * (B^T * X)
// ---------------------------------------------------------------------------
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B)
{
    TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
                A.is_contiguous() && B.is_contiguous(),
                "All inputs must be contiguous");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "Expected float32");

    g_state.init();

    const int d = (int)W.size(0);
    const int r = (int)A.size(1);   // 16

    auto opts = W.options();
    auto Y   = torch::empty({d, d}, opts);   // output
    auto BtX = torch::empty({r, d}, opts);   // intermediate B^T * X

    const float* Wp   = W.data_ptr<float>();
    const float* Xp   = X.data_ptr<float>();
    const float* Ap   = A.data_ptr<float>();
    const float* Bp   = B.data_ptr<float>();
    float*       Yp   = Y.data_ptr<float>();
    float*       BtXp = BtX.data_ptr<float>();

    // -- Stream 0: Y = W * X  (large d x d GEMM, TF32 tensor cores) ----------
    rmm_ex(g_state.h0,
           CUBLAS_OP_N, CUBLAS_OP_N,
           d, d, d,
           1.0f, Wp, d, Xp, d,
           0.0f, Yp, d);

    // Record event on stream 0 so stream 1 can wait for Y to be ready
    cudaEventRecord(g_state.ev, g_state.s0);

    // -- Stream 1: BtX = B^T * X  (small r x d GEMM) -------------------------
    //    B is d x r row-major; use opA=T to treat it as B^T (r x d)
    rmm(g_state.h1,
        CUBLAS_OP_T, CUBLAS_OP_N,   // opA=T on B(d x r), opB=N on X(d x d)
        r, d, d,
        1.0f, Bp, r, Xp, d,
        0.0f, BtXp, d);

    // Wait for stream 0 to finish writing Y before accumulating
    cudaStreamWaitEvent(g_state.s1, g_state.ev, 0);

    // -- Stream 1 (cont.): Y += A * BtX  (d x d GEMM with k=r=16) ------------
    rmm(g_state.h1,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d, d, r,
        1.0f, Ap, r, BtXp, d,
        1.0f, Yp, d);   // beta=1 accumulates into Y

    // Wait for all work to complete before returning the tensor
    cudaStreamSynchronize(g_state.s1);
    // (stream 0 is already done since stream 1 waited for its event above)

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
          "LoRA forward: Y = W*X + A*(B^T*X) [cuBLAS TF32, dual-stream]");
}
