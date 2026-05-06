#!/usr/bin/env python3
"""
agent.py – LoRA CUDA Optimization Agent
========================================
Iteratively generates, compiles, tests, benchmarks, and improves CUDA
implementations of:

    Y = W * X  +  A * (B^T * X)

where W, X in R^{d x d}, A, B in R^{d x r}, d in [3584, 4608], r = 16.

Usage:
    python agent.py [--time-budget 1500]  [--model gpt-4o]

Environment variables:
    OPENAI_API_KEY   – required when using OpenAI backend
    OPENAI_MODEL     – optional override (default: gpt-4o)

The agent always keeps the best correct implementation in ./optimized_lora.cu.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
import textwrap
import time
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR       = Path(__file__).resolve().parent
OPTIMIZED_PATH = WORK_DIR / "optimized_lora.cu"
BASELINE_PATH  = WORK_DIR / "optimized_lora.cu"   # same file – always valid

DEFAULT_TIME_BUDGET = 25 * 60   # 25 min (leaves 5-min buffer out of 30)
TEST_SIZES = [3584, 4096, 4608]
R = 16

# ─────────────────────────────────────────────────────────────────────────────
# Reference implementation
# ─────────────────────────────────────────────────────────────────────────────
def reference_impl(W, X, A, B):
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def make_tensors(d, r=16, seed=42):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(d, d, generator=g).cuda().float()
    X = torch.randn(d, d, generator=g).cuda().float()
    A = torch.randn(d, r, generator=g).cuda().float()
    B = torch.randn(d, r, generator=g).cuda().float()
    return W, X, A, B


# ─────────────────────────────────────────────────────────────────────────────
# Compilation
# ─────────────────────────────────────────────────────────────────────────────
def compile_module(cu_path: Path, build_dir: Path):
    """Return (module, error_string).  module is None on failure.
    Each unique source gets its own module name to avoid JIT-cache collisions.
    """
    from torch.utils.cpp_extension import load
    code_hash = hashlib.md5(cu_path.read_bytes()).hexdigest()[:8]
    mod_name  = f"lora_ext_{code_hash}"
    try:
        mod = load(
            name=mod_name,
            sources=[str(cu_path)],
            build_dir=str(build_dir),
            verbose=False,
            extra_cuda_cflags=["-O3", "-arch=sm_86"],
            with_cuda=True,
        )
        return mod, None
    except Exception as exc:
        return None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Correctness & benchmark
# ─────────────────────────────────────────────────────────────────────────────
def check_correctness(module, d: int = 4096) -> tuple[bool, str]:
    W, X, A, B = make_tensors(d)
    try:
        with torch.no_grad():
            y_student = module.forward(W, X, A, B)
            y_ref     = reference_impl(W, X, A, B)
        ok = torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4)
        if not ok:
            diff = (y_student - y_ref).float()
            return False, f"max_err={diff.abs().max().item():.3e}"
        return True, ""
    except Exception as exc:
        return False, f"runtime error: {exc}"


def benchmark_module(module, d: int = 4096, warmup: int = 10, iters: int = 50) -> float:
    W, X, A, B = make_tensors(d)
    for _ in range(warmup):
        module.forward(W, X, A, B)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); module.forward(W, X, A, B); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def benchmark_torch(d: int = 4096, warmup: int = 10, iters: int = 50) -> float:
    W, X, A, B = make_tensors(d)
    for _ in range(warmup):
        reference_impl(W, X, A, B)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); reference_impl(W, X, A, B); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# ─────────────────────────────────────────────────────────────────────────────
# Full candidate evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_candidate(cu_code: str, build_dir: Path, d: int = 4096):
    """
    Returns dict:
      { passed: bool, time_ms: float|None, error: str|None }
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    cu_path = build_dir / "candidate.cu"
    cu_path.write_text(cu_code)

    module, err = compile_module(cu_path, build_dir)
    if module is None:
        return {"passed": False, "time_ms": None, "error": f"compile: {err[:600]}"}

    ok, msg = check_correctness(module, d=d)
    if not ok:
        return {"passed": False, "time_ms": None, "error": f"correctness: {msg}"}

    try:
        t = benchmark_module(module, d=d)
    except Exception as exc:
        return {"passed": False, "time_ms": None, "error": f"benchmark: {exc}"}

    return {"passed": True, "time_ms": t, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# LLM back-end (OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
class LLMBackend:
    def __init__(self, api_key: str | None, model: str = "gpt-4o"):
        self.model = model
        self.client = None
        if not api_key:
            print("[agent] No OPENAI_API_KEY found – LLM generation disabled.")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"[agent] OpenAI backend ready (model={model})")
        except ImportError:
            print("[agent] openai package not installed – LLM disabled.")

    @property
    def available(self):
        return self.client is not None

    def generate(self, prompt: str, max_tokens: int = 4096) -> str | None:
        if not self.available:
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            print(f"[agent] LLM error: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_CONTEXT = textwrap.dedent("""\
    You are a GPU performance engineer optimising a CUDA kernel for the operator:

        Y = W * X  +  A * (B^T * X)

    Shapes:
        W, X : d x d  float32   d in [3584, 4608]
        A, B : d x r  float32   r = 16

    Target GPU: NVIDIA GeForce RTX 3090 (Ampere sm_86).
    Toolchain: CUDA 12.4, PyTorch 2.3, GCC 11.4.

    REQUIREMENTS (non-negotiable):
    1. Single .cu file, self-contained, includes <torch/extension.h>.
    2. Exports exactly:
           torch::Tensor forward(torch::Tensor W, torch::Tensor X,
                                 torch::Tensor A, torch::Tensor B);
       via PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, ...); }
    3. Results must pass torch.allclose(Y_student, Y_ref, rtol=1e-4, atol=1e-4)
       against the PyTorch reference:
           Y_ref = W @ X + A @ (B.T.contiguous() @ X)
    4. Works for any d in [3584, 4608] – do NOT hard-code d.

    OUTPUT: Return ONLY the complete CUDA source code (no prose, no markdown fences).
""")


def build_prompt(best_code: str, best_ms: float | None,
                 history: list[dict], last_error: str | None,
                 iteration: int) -> str:
    history_lines = []
    for h in history[-8:]:   # keep last 8 entries
        status = "PASS" if h["passed"] else "FAIL"
        t_str  = f"{h['time_ms']:.3f}ms" if h.get("time_ms") else "N/A"
        e_str  = h.get("error") or ""
        history_lines.append(f"  iter {h['iter']:3d}: {status}  time={t_str}  {e_str[:120]}")

    best_str = f"{best_ms:.3f}ms" if best_ms is not None else "N/A"

    prompt = SYSTEM_CONTEXT + f"""
---- CURRENT BEST (time = {best_str}, iteration {iteration}) ----
{best_code}
---- RECENT HISTORY ----
{chr(10).join(history_lines) or "  (none yet)"}
"""
    if last_error:
        prompt += f"\n---- LAST ERROR (please fix) ----\n{last_error[:800]}\n"

    prompt += textwrap.dedent(f"""
    ---- YOUR TASK ----
    Iteration {iteration}: produce an IMPROVED CUDA implementation.

    Key optimisation directions to consider:
    * cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 for the d×d W*X GEMM
    * Two non-blocking CUDA streams: one for W*X, one for B^T*X; accumulate
      Y += A*(B^T*X) on stream-1 after a CUDA-event sync from stream-0
    * Persistent cuBLAS handles (static, initialised once)
    * Row-major GEMM trick: C(m,n)=A*B in row-major
      => cublasSgemm(h, opB, opA, n,m,k, alpha, B,ldB, A,ldA, beta, C,ldC)
    * Optionally, a custom CUDA kernel for the final element-wise add / fusion
    * Explore CUTLASS, Triton (if available), or hand-written tiled GEMM
    * CUDA Graphs to amortise kernel-launch overhead

    RETURN ONLY THE COMPLETE .cu SOURCE CODE.
    """)
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Predefined hand-crafted variants (fallback when LLM unavailable)
# ─────────────────────────────────────────────────────────────────────────────
def load_predefined_variants() -> list[str]:
    """
    Return a list of alternative CUDA implementations to try when no LLM
    is available.  Each string is a complete .cu source.
    """
    variants = []

    # Variant 1: sequential cuBLAS, no streams (simpler, sometimes faster
    #            due to less synchronisation overhead on small workloads)
    variants.append(textwrap.dedent("""\
        #include <torch/extension.h>
        #include <cublas_v2.h>
        #include <cuda_runtime.h>

        static cublasHandle_t g_handle = nullptr;

        static void ensure_handle() {
            if (g_handle) return;
            cublasCreate(&g_handle);
            cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        }

        static inline void rmm(cublasHandle_t h,
                                cublasOperation_t opA, cublasOperation_t opB,
                                int m, int n, int k, float alpha,
                                const float* A, int ldA,
                                const float* B, int ldB,
                                float beta, float* C, int ldC) {
            cublasSgemm(h, opB, opA, n, m, k, &alpha,
                        B, ldB, A, ldA, &beta, C, ldC);
        }

        torch::Tensor forward(torch::Tensor W, torch::Tensor X,
                              torch::Tensor A, torch::Tensor B) {
            TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda());
            TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
                        A.is_contiguous() && B.is_contiguous());
            ensure_handle();

            const int d = (int)W.size(0);
            const int r = (int)A.size(1);

            auto Y   = torch::empty({d, d}, W.options());
            auto BtX = torch::empty({r, d}, W.options());

            const float *Wp=W.data_ptr<float>(), *Xp=X.data_ptr<float>();
            const float *Ap=A.data_ptr<float>(), *Bp=B.data_ptr<float>();
            float *Yp=Y.data_ptr<float>(), *BtXp=BtX.data_ptr<float>();

            // Y = W * X
            rmm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d, d, d,
                1.0f, Wp, d, Xp, d, 0.0f, Yp, d);
            // BtX = B^T * X
            rmm(g_handle, CUBLAS_OP_T, CUBLAS_OP_N, r, d, d,
                1.0f, Bp, r, Xp, d, 0.0f, BtXp, d);
            // Y += A * BtX
            rmm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d, d, r,
                1.0f, Ap, r, BtXp, d, 1.0f, Yp, d);

            return Y;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &forward, "LoRA forward (sequential cuBLAS TF32)");
        }
    """))

    # Variant 2: W' = W + A*B^T first, then Y = W'*X  (single large GEMM)
    variants.append(textwrap.dedent("""\
        #include <torch/extension.h>
        #include <cublas_v2.h>
        #include <cuda_runtime.h>

        static cublasHandle_t g_handle = nullptr;

        static void ensure_handle() {
            if (g_handle) return;
            cublasCreate(&g_handle);
            cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
        }

        static inline void rmm_ex(cublasHandle_t h,
                                   cublasOperation_t opA, cublasOperation_t opB,
                                   int m, int n, int k, float alpha,
                                   const float* A, int ldA,
                                   const float* B, int ldB,
                                   float beta, float* C, int ldC) {
            cublasGemmEx(h, opB, opA, n, m, k, &alpha,
                         B, CUDA_R_32F, ldB, A, CUDA_R_32F, ldA, &beta,
                         C, CUDA_R_32F, ldC,
                         CUBLAS_COMPUTE_32F_FAST_TF32,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        torch::Tensor forward(torch::Tensor W, torch::Tensor X,
                              torch::Tensor A, torch::Tensor B) {
            TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda());
            TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
                        A.is_contiguous() && B.is_contiguous());
            ensure_handle();

            const int d = (int)W.size(0);
            const int r = (int)A.size(1);

            // Wprime = W + A * B^T  (d x d)
            // Then Y = Wprime * X
            auto Wprime = W.clone();   // copy W
            float *Wp2 = Wprime.data_ptr<float>();
            const float *Ap=A.data_ptr<float>(), *Bp=B.data_ptr<float>();
            const float *Xp=X.data_ptr<float>();

            // Wprime += A * B^T
            // row-major: Wprime(d,d) = A(d,r) * B^T(r,d)  with opA=N, opB=T on B(d,r)
            rmm_ex(g_handle, CUBLAS_OP_N, CUBLAS_OP_T, d, d, r,
                   1.0f, Ap, r, Bp, r, 1.0f, Wp2, d);

            // Y = Wprime * X
            auto Y = torch::empty({d, d}, W.options());
            float *Yp = Y.data_ptr<float>();
            rmm_ex(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, d, d, d,
                   1.0f, Wp2, d, Xp, d, 0.0f, Yp, d);

            return Y;
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &forward, "LoRA forward (W+ABT first, then GEMM)");
        }
    """))

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# Extract CUDA code from LLM response (strips markdown fences if present)
# ─────────────────────────────────────────────────────────────────────────────
def extract_code(text: str) -> str:
    # Try to find a fenced code block
    m = re.search(r"```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fall back: return the whole text (might already be raw code)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, time_budget: int, llm: LLMBackend, eval_d: int = 4096):
        self.time_budget = time_budget
        self.llm         = llm
        self.eval_d      = eval_d

        self.best_time: float | None = None
        self.best_code: str | None   = None
        self.history: list[dict]     = []
        self.base_build = Path(tempfile.mkdtemp(prefix="lora_agent_"))

        # Benchmark the PyTorch reference once (for logging)
        self.torch_ms: float | None = None

    # ── Save best to disk ────────────────────────────────────────────────────
    def _save_best(self):
        if self.best_code is not None:
            OPTIMIZED_PATH.write_text(self.best_code)
            print(f"  [agent] Saved best ({self.best_time:.3f}ms) → {OPTIMIZED_PATH}")

    # ── Initialise with the current optimized_lora.cu as baseline ────────────
    def bootstrap(self):
        baseline_code = OPTIMIZED_PATH.read_text()
        print("[agent] Evaluating baseline …")
        build_dir = self.base_build / "baseline"
        result = evaluate_candidate(baseline_code, build_dir, d=self.eval_d)
        self._record(0, baseline_code, result)
        if result["passed"]:
            self.best_time = result["time_ms"]
            self.best_code = baseline_code
            self.torch_ms  = benchmark_torch(d=self.eval_d)
            speedup = self.torch_ms / self.best_time
            print(f"  baseline  : {self.best_time:.3f}ms  "
                  f"(torch={self.torch_ms:.3f}ms  speedup={speedup:.2f}x)")
        else:
            print(f"  baseline failed: {result['error']}")

    # ── Record a trial ───────────────────────────────────────────────────────
    def _record(self, iteration: int, code: str, result: dict):
        entry = {
            "iter":    iteration,
            "passed":  result["passed"],
            "time_ms": result.get("time_ms"),
            "error":   result.get("error"),
        }
        self.history.append(entry)
        status = "PASS" if result["passed"] else "FAIL"
        t_str  = f"{result['time_ms']:.3f}ms" if result.get("time_ms") else "N/A"
        print(f"  iter {iteration:3d}: {status}  {t_str}  {result.get('error') or ''}")

    # ── Main loop ────────────────────────────────────────────────────────────
    def run(self):
        self.bootstrap()

        predefined = load_predefined_variants()
        pre_idx    = 0
        last_error: str | None = None
        start_time = time.time()
        iteration  = 0

        while True:
            elapsed   = time.time() - start_time
            remaining = self.time_budget - elapsed
            if remaining < 60:
                print(f"\n[agent] Time budget nearly exhausted ({remaining:.0f}s left). Stopping.")
                break

            iteration += 1
            print(f"\n[agent] === Iteration {iteration}  "
                  f"({remaining:.0f}s remaining)  "
                  f"best={self.best_time:.3f}ms ==="
                  if self.best_time else
                  f"\n[agent] === Iteration {iteration}  ({remaining:.0f}s remaining) ===")

            # ── Generate new candidate ───────────────────────────────────────
            new_code: str | None = None

            if self.llm.available:
                prompt   = build_prompt(
                    self.best_code or "",
                    self.best_time,
                    self.history,
                    last_error,
                    iteration,
                )
                response = self.llm.generate(prompt)
                if response:
                    new_code = extract_code(response)

            # Fall back to predefined variants if LLM unavailable or failed
            if new_code is None:
                if pre_idx < len(predefined):
                    new_code = predefined[pre_idx]
                    pre_idx += 1
                    print(f"  [agent] Using predefined variant #{pre_idx}")
                else:
                    print("  [agent] No more variants to try. Sleeping 30s …")
                    time.sleep(30)
                    continue

            # ── Evaluate ─────────────────────────────────────────────────────
            build_dir = self.base_build / f"iter_{iteration}"
            result    = evaluate_candidate(new_code, build_dir, d=self.eval_d)
            self._record(iteration, new_code, result)

            if result["passed"]:
                last_error = None
                if self.best_time is None or result["time_ms"] < self.best_time:
                    self.best_time = result["time_ms"]
                    self.best_code = new_code
                    self._save_best()
                    if self.torch_ms is None:
                        self.torch_ms = benchmark_torch(d=self.eval_d)
                    speedup = self.torch_ms / self.best_time
                    print(f"  *** NEW BEST: {self.best_time:.3f}ms  "
                          f"speedup={speedup:.2f}x ***")
            else:
                last_error = result.get("error", "unknown error")

        # ── Final report ─────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"[agent] Finished.  Best time: "
              f"{self.best_time:.3f}ms" if self.best_time else "[agent] No valid candidate found.")
        if self.torch_ms and self.best_time:
            print(f"[agent] PyTorch ref: {self.torch_ms:.3f}ms  "
                  f"Speedup: {self.torch_ms/self.best_time:.2f}x")
        print(f"[agent] Final kernel saved to: {OPTIMIZED_PATH}")
        print("=" * 60)

        # Save history for inspection
        hist_path = WORK_DIR / "agent_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"[agent] History → {hist_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LoRA CUDA Optimization Agent")
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET,
                        help="Total time budget in seconds (default: 1500 = 25 min)")
    parser.add_argument("--model", type=str,
                        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                        help="OpenAI model name (default: gpt-4o)")
    parser.add_argument("--d", type=int, default=4096,
                        help="Matrix dimension to use for local benchmarking")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    llm     = LLMBackend(api_key=api_key, model=args.model)

    agent = Agent(
        time_budget=args.time_budget,
        llm=llm,
        eval_d=args.d,
    )
    agent.run()


if __name__ == "__main__":
    main()
