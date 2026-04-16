#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible API server (vLLM, SGLang, etc.).

Sends synthetic workloads with varying input/output lengths and measures
throughput, TTFT, decode speed, and end-to-end latency.

Usage:
    ./bench.py spark7:8080
    ./bench.py spark7:8080 --model "Intel/Step-3.5-Flash-int4-mixed-AutoRound"
    ./bench.py spark7:8080 --input-lens 128 512 2048 --output-lens 128 512
    ./bench.py spark7:8080 --concurrency 4 --num-prompts 20
    ./bench.py spark7:8080 --save results.json
"""

import argparse
import json
import random
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import requests

VOCAB = string.ascii_lowercase + " "


def random_prompt(token_count: int) -> str:
    """Generate a prompt that approximates `token_count` tokens (~4 chars/tok)."""
    chars = token_count * 4
    return "".join(random.choice(VOCAB) for _ in range(chars))


def detect_model(base_url: str) -> str:
    resp = requests.get(f"{base_url}/v1/models", timeout=10)
    resp.raise_for_status()
    models = resp.json()["data"]
    if not models:
        sys.exit("No models found on server")
    return models[0]["id"]


@dataclass
class RequestResult:
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    success: bool = False
    error: str = ""


def send_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool = True,
    expected_input_len: int = 0,
) -> RequestResult:
    result = RequestResult()
    result.input_tokens = expected_input_len
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }

    t0 = time.perf_counter()
    first_token_time = None

    try:
        if stream:
            resp = requests.post(
                f"{base_url}/v1/completions",
                json={**payload, "stream": True, "stream_options": {"include_usage": True}},
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
            output_tokens = 0
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                chunk = json.loads(data)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                usage = chunk.get("usage")
                if usage and "completion_tokens" in usage:
                    output_tokens = usage["completion_tokens"]
                    result.input_tokens = usage.get("prompt_tokens", 0)
                else:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        output_tokens += 1
            result.output_tokens = output_tokens
        else:
            resp = requests.post(
                f"{base_url}/v1/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            first_token_time = time.perf_counter()
            data = resp.json()
            usage = data["usage"]
            result.input_tokens = usage["prompt_tokens"]
            result.output_tokens = usage["completion_tokens"]

        t1 = time.perf_counter()
        result.total_ms = (t1 - t0) * 1000
        result.ttft_ms = (first_token_time - t0) * 1000 if first_token_time else result.total_ms
        result.success = True

    except Exception as e:
        result.total_ms = (time.perf_counter() - t0) * 1000
        result.error = str(e)

    return result


def percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def run_workload(
    base_url: str,
    model: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    concurrency: int,
    stream: bool,
) -> dict:
    prompts = [random_prompt(input_len) for _ in range(num_prompts)]
    results: list[RequestResult] = []

    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(send_request, base_url, model, p, output_len, stream, input_len)
            for p in prompts
        ]
        for f in as_completed(futures):
            results.append(f.result())

    t_elapsed = time.perf_counter() - t_start

    ok = [r for r in results if r.success]
    failed = len(results) - len(ok)

    if not ok:
        return {
            "input_len": input_len,
            "output_len": output_len,
            "num_prompts": num_prompts,
            "concurrency": concurrency,
            "completed": 0,
            "failed": failed,
            "error": results[0].error if results else "unknown",
        }

    ttfts = [r.ttft_ms for r in ok]
    totals = [r.total_ms for r in ok]
    total_out = sum(r.output_tokens for r in ok)
    total_in = sum(r.input_tokens for r in ok)
    decode_tps = [
        r.output_tokens / ((r.total_ms - r.ttft_ms) / 1000)
        for r in ok
        if r.total_ms > r.ttft_ms and r.output_tokens > 0
    ]
    prefill_tps = [
        r.input_tokens / (r.ttft_ms / 1000)
        for r in ok
        if r.ttft_ms > 0 and r.input_tokens > 0
    ]

    return {
        "input_len": input_len,
        "output_len": output_len,
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "completed": len(ok),
        "failed": failed,
        "duration_s": round(t_elapsed, 2),
        "throughput_req_s": round(len(ok) / t_elapsed, 2),
        "throughput_out_tok_s": round(total_out / t_elapsed, 1),
        "throughput_total_tok_s": round((total_in + total_out) / t_elapsed, 1),
        "ttft_mean_ms": round(sum(ttfts) / len(ttfts), 1),
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p99_ms": round(percentile(ttfts, 99), 1),
        "e2e_mean_ms": round(sum(totals) / len(totals), 1),
        "e2e_p50_ms": round(percentile(totals, 50), 1),
        "e2e_p99_ms": round(percentile(totals, 99), 1),
        "decode_tok_s_mean": round(sum(decode_tps) / len(decode_tps), 1) if decode_tps else 0,
        "prefill_tok_s_mean": round(sum(prefill_tps) / len(prefill_tps), 1) if prefill_tps else 0,
    }


def fmt_table(rows: list[dict]) -> str:
    cols = [
        ("in", "input_len", "5"),
        ("out", "output_len", "5"),
        ("n", "completed", "3"),
        ("req/s", "throughput_req_s", "6.2f"),
        ("out tok/s", "throughput_out_tok_s", "9.1f"),
        ("prefill t/s", "prefill_tok_s_mean", "11.1f"),
        ("decode t/s", "decode_tok_s_mean", "10.1f"),
        ("TTFT p50", "ttft_p50_ms", "9.0f"),
        ("TTFT p99", "ttft_p99_ms", "9.0f"),
        ("E2E p50", "e2e_p50_ms", "8.0f"),
        ("E2E p99", "e2e_p99_ms", "8.0f"),
    ]
    header = " | ".join(f"{name:>{fmt[-1] if fmt[-1].isdigit() else fmt.split('.')[0].rstrip('0123456789') or '8'}s}"
                        if True else "" for name, _, fmt in cols)
    # simpler approach
    widths = [max(len(name), 8) for name, _, _ in cols]
    header = " | ".join(f"{name:>{w}}" for (name, _, _), w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    lines = [header, sep]
    for row in rows:
        if "error" in row:
            lines.append(f"  in={row['input_len']} out={row['output_len']}: FAILED — {row['error'][:60]}")
            continue
        cells = []
        for (_, key, fmt), w in zip(cols, widths):
            v = row.get(key, 0)
            cells.append(f"{v:>{w}{fmt[-1] if not fmt[-1].isdigit() else ''}}" if isinstance(v, float)
                         else f"{v:>{w}}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible API server")
    p.add_argument("server", help="host:port (e.g. spark7:8080)")
    p.add_argument("--model", help="Model name (auto-detected if omitted)")
    p.add_argument("--input-lens", type=int, nargs="+", default=[128, 512, 2048],
                   help="Input token lengths to test (default: 128 512 2048)")
    p.add_argument("--output-lens", type=int, nargs="+", default=[128, 512],
                   help="Output token lengths to test (default: 128 512)")
    p.add_argument("--num-prompts", type=int, default=10,
                   help="Number of prompts per workload (default: 10)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Concurrent requests (default: 1)")
    p.add_argument("--warmup", type=int, default=2,
                   help="Warmup requests before measuring (default: 2)")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable streaming (no TTFT measurement)")
    p.add_argument("--save", metavar="FILE", help="Save results to JSON file")
    args = p.parse_args()

    base_url = f"http://{args.server}"

    print(f"Server:      {base_url}")
    try:
        model = args.model or detect_model(base_url)
    except Exception as e:
        sys.exit(f"Cannot reach server: {e}")
    print(f"Model:       {model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Prompts:     {args.num_prompts} per workload")
    print(f"Streaming:   {not args.no_stream}")
    print()

    if args.warmup > 0:
        print(f"Warming up ({args.warmup} requests)...")
        for _ in range(args.warmup):
            send_request(base_url, model, random_prompt(32), 16, stream=not args.no_stream, expected_input_len=32)
        print()

    all_results = []

    total = len(args.input_lens) * len(args.output_lens)
    idx = 0
    for input_len in args.input_lens:
        for output_len in args.output_lens:
            idx += 1
            print(f"[{idx}/{total}] in={input_len} out={output_len} n={args.num_prompts} c={args.concurrency} ...",
                  end="", flush=True)
            result = run_workload(
                base_url, model, input_len, output_len,
                args.num_prompts, args.concurrency,
                stream=not args.no_stream,
            )
            all_results.append(result)
            if "error" in result:
                print(f" FAILED: {result['error'][:60]}")
            else:
                print(f" {result['throughput_out_tok_s']} tok/s  TTFT={result['ttft_p50_ms']:.0f}ms  decode={result['decode_tok_s_mean']:.0f}t/s")

    print()
    print("=" * 100)
    print(f"Results: {model} @ {base_url}")
    print(f"         concurrency={args.concurrency}  prompts={args.num_prompts}")
    print("=" * 100)
    print()

    # Simple table output
    hdr = f"{'in':>6} {'out':>6} {'n':>4} {'req/s':>7} {'out t/s':>9} {'prefill t/s':>12} {'decode t/s':>11} {'TTFT p50':>10} {'TTFT p99':>10} {'E2E p50':>10} {'E2E p99':>10}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in all_results:
        if "error" in r:
            print(f"{r['input_len']:>6} {r['output_len']:>6}  FAILED: {r['error'][:60]}")
        else:
            print(f"{r['input_len']:>6} {r['output_len']:>6} {r['completed']:>4} "
                  f"{r['throughput_req_s']:>7.2f} {r['throughput_out_tok_s']:>9.1f} "
                  f"{r['prefill_tok_s_mean']:>12.1f} {r['decode_tok_s_mean']:>11.1f} "
                  f"{r['ttft_p50_ms']:>10.0f} {r['ttft_p99_ms']:>10.0f} "
                  f"{r['e2e_p50_ms']:>10.0f} {r['e2e_p99_ms']:>10.0f}")
    print(sep)
    print()

    if args.save:
        out = {
            "server": base_url,
            "model": model,
            "concurrency": args.concurrency,
            "num_prompts": args.num_prompts,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "results": all_results,
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
