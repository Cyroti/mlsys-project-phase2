#!/usr/bin/env python3
"""
Local harness – mirrors the official evaluation harness.

Usage:
    python harness.py <path/to/optimized_lora.cu> [--d 4096]

Returns:
    JSON dict with: passed, max_abs_err, rel_l2_err, student_ms, torch_ms, speedup
"""

import sys
import os
import json
import tempfile
import argparse
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Reference implementation  (matches the official harness)
# ---------------------------------------------------------------------------
def reference_impl(W, X, A, B):
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


# ---------------------------------------------------------------------------
# Build a CUDA extension from a .cu file
# ---------------------------------------------------------------------------
def build_module(cu_path: str, build_dir: str):
    from torch.utils.cpp_extension import load
    module = load(
        name="optimized_lora_ext",
        sources=[cu_path],
        build_dir=build_dir,
        verbose=False,
        extra_cuda_cflags=["-O3", "-arch=sm_86"],
        with_cuda=True,
    )
    return module


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
def check_correctness(y, y_ref):
    diff = (y - y_ref).float()
    max_abs_err = diff.abs().max().item()
    rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
    passed = torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)
    return passed, max_abs_err, rel_l2_err


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(fn, W, X, A, B, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn(W, X, A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = fn(W, X, A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Generate random test tensors
# ---------------------------------------------------------------------------
def make_tensors(d, r=16, seed=42, device="cuda"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(d, d, generator=g).cuda().float()
    X = torch.randn(d, d, generator=g).cuda().float()
    A = torch.randn(d, r, generator=g).cuda().float()
    B = torch.randn(d, r, generator=g).cuda().float()
    return W, X, A, B


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
def evaluate(cu_path: str, d: int = 4096, build_dir: str | None = None) -> dict:
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="lora_build_")

    try:
        module = build_module(cu_path, build_dir)
    except Exception as e:
        return {
            "passed": False,
            "error": f"compile: {e}",
            "max_abs_err": None,
            "rel_l2_err": None,
            "student_ms": None,
            "torch_ms": None,
            "speedup": 0.0,
        }

    W, X, A, B = make_tensors(d)

    try:
        with torch.no_grad():
            y_student = module.forward(W, X, A, B)
            y_ref = reference_impl(W, X, A, B)
    except Exception as e:
        return {
            "passed": False,
            "error": f"runtime: {e}",
            "max_abs_err": None,
            "rel_l2_err": None,
            "student_ms": None,
            "torch_ms": None,
            "speedup": 0.0,
        }

    passed, max_abs_err, rel_l2_err = check_correctness(y_student, y_ref)

    if passed:
        student_ms = benchmark(module.forward, W, X, A, B)
        torch_ms = benchmark(reference_impl, W, X, A, B)
        speedup = torch_ms / student_ms
    else:
        student_ms = None
        torch_ms = None
        speedup = 0.0

    return {
        "passed": passed,
        "max_abs_err": max_abs_err,
        "rel_l2_err": rel_l2_err,
        "student_ms": student_ms,
        "torch_ms": torch_ms,
        "speedup": speedup,
        "d": d,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a LoRA CUDA kernel")
    parser.add_argument("cu_path", help="Path to optimized_lora.cu")
    parser.add_argument("--d", type=int, default=4096, help="Matrix dimension")
    parser.add_argument("--build-dir", default=None, help="Build directory")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    result = evaluate(args.cu_path, d=args.d, build_dir=args.build_dir)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"d              = {result.get('d')}")
        print(f"correct        = {result['passed']}")
        if result.get("error"):
            print(f"error          = {result['error']}")
        print(f"max_abs_err    = {result['max_abs_err']}")
        print(f"rel_l2_err     = {result['rel_l2_err']}")
        print(f"student_ms     = {result['student_ms']}")
        print(f"torch_ms       = {result['torch_ms']}")
        print(f"speedup        = {result['speedup']}")
