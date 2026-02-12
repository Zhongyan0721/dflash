"""Benchmark vLLM serving with and without DFlash speculative decoding.

Usage:
    # 1. Start the vLLM server (with or without --speculative-config)
    # 2. Run this script:
    python benchmark_vllm.py --url http://localhost:8000 --num-prompts 5
"""

import argparse
import json
import time
import requests


PROMPTS = [
    "How many positive whole-number divisors does 196 have?",
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the main causes of climate change?",
    "Summarize the plot of Romeo and Juliet.",
    "Describe the process of photosynthesis step by step.",
    "What is the difference between machine learning and deep learning?",
    "Write a short poem about the ocean.",
]


def send_request(url: str, prompt: str, max_tokens: int = 256) -> dict:
    """Send a completion request and return timing + token info."""
    payload = {
        "model": "Qwen/Qwen3-4B",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.perf_counter()
    resp = requests.post(f"{url}/v1/completions", json=payload, timeout=300)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data["usage"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "elapsed_s": elapsed,
        "tokens_per_s": usage["completion_tokens"] / elapsed if elapsed > 0 else 0,
        "text_preview": data["choices"][0]["text"][:80],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup requests")
    args = parser.parse_args()

    prompts = PROMPTS[: args.num_prompts]

    # Warmup
    print(f"=== Warmup ({args.warmup} request(s)) ===")
    for i in range(args.warmup):
        r = send_request(args.url, "Hello", max_tokens=16)
        print(f"  warmup {i}: {r['elapsed_s']:.2f}s")

    # Benchmark
    print(f"\n=== Benchmark ({len(prompts)} prompts, max_tokens={args.max_tokens}) ===")
    results = []
    for i, prompt in enumerate(prompts):
        r = send_request(args.url, prompt, max_tokens=args.max_tokens)
        results.append(r)
        print(
            f"  [{i+1}/{len(prompts)}] "
            f"prompt_tokens={r['prompt_tokens']:>4}, "
            f"completion_tokens={r['completion_tokens']:>4}, "
            f"elapsed={r['elapsed_s']:>6.2f}s, "
            f"tok/s={r['tokens_per_s']:>6.1f}"
        )

    # Summary
    total_gen_tokens = sum(r["completion_tokens"] for r in results)
    total_time = sum(r["elapsed_s"] for r in results)
    avg_tok_s = total_gen_tokens / total_time if total_time > 0 else 0
    avg_latency = total_time / len(results)

    print(f"\n=== Summary ===")
    print(f"  Total generation tokens : {total_gen_tokens}")
    print(f"  Total wall-clock time   : {total_time:.2f}s")
    print(f"  Avg tokens/s            : {avg_tok_s:.1f}")
    print(f"  Avg latency per request : {avg_latency:.2f}s")


if __name__ == "__main__":
    main()
