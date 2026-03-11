"""
bench_decode.py — 单独测量 decode 阶段吞吐量

用法：
  python bench_decode.py --batch-size 8 --fixed_gen_tokens 256

原理：
  LLMEngine.step() 返回的 num_tokens：
    > 0  → prefill 阶段（值 = 本轮处理的 token 总数）
    < 0  → decode  阶段（值 = -batch_size，即每步生成 bs 个 token）
  
  我们只统计 decode 阶段的时间和 token 数。
"""

import argparse
import os
import time
from random import randint, seed
from time import perf_counter

from nanovllm import LLM, SamplingParams
from nanovllm.utils.secure import set_security_config


def parse_args():
    parser = argparse.ArgumentParser(description="Decode-only throughput benchmark")
    parser.add_argument("--fixed_gen_tokens", type=int, default=256,
                        help="Number of tokens each request must generate.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of prompts/sequences per run.")
    parser.add_argument("--dist-url", type=str, default=None)
    parser.add_argument("--dist-port", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    seed(0)
    num_seqs = args.batch_size
    max_input_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    # 安全配置（与 bench.py 一致）
    switch = True
    set_security_config(
        enable_softmax_encrypt=switch,
        enable_linear_noise=switch,
        encrypt_on_cpu=switch,
        decrypt_on_cpu=switch,
        tee_strict_mode=switch,
        noise_pool_size=16,
        noise_scale=0.05,
        seed=1234,
    )

    llm_kwargs = {"enforce_eager": True, "max_model_len": 4096}
    if args.dist_url and args.dist_port:
        raise ValueError("Specify either --dist-url or --dist-port, not both.")
    if args.dist_url:
        llm_kwargs["dist_url"] = args.dist_url
    elif args.dist_port:
        llm_kwargs["dist_url"] = f"tcp://127.0.0.1:{args.dist_port}"

    llm = LLM(path, **llm_kwargs)

    # 构造随机 prompt（与 bench.py 一致）
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.fixed_gen_tokens)
        for _ in range(num_seqs)
    ]

    # 打印 prompt 长度信息
    prompt_lens = [len(p) for p in prompt_token_ids]
    avg_prompt_len = sum(prompt_lens) / len(prompt_lens)
    print(f"Prompt lengths: {prompt_lens}")
    print(f"Average prompt length: {avg_prompt_len:.1f}")
    print(f"Batch size: {num_seqs}")
    print(f"Fixed gen tokens: {args.fixed_gen_tokens}")
    print()

    # Warmup
    llm.generate(
        ["Benchmark warmup: "],
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.fixed_gen_tokens),
    )

    # ========== 手动驱动 engine，分离 prefill / decode ==========
    engine = llm  # LLM 继承自 LLMEngine

    engine._step_id = 0

    # 添加请求
    for prompt, sp in zip(prompt_token_ids, sampling_params):
        engine.add_request(prompt, sp)

    prefill_tokens = 0
    prefill_time = 0.0
    decode_tokens = 0
    decode_time = 0.0
    decode_steps = 0

    while not engine.is_finished():
        t_start = perf_counter()
        output, num_tokens = engine.step()
        t_elapsed = perf_counter() - t_start

        if num_tokens > 0:
            # prefill 阶段
            prefill_tokens += num_tokens
            prefill_time += t_elapsed
        else:
            # decode 阶段：num_tokens = -batch_size（每步每个序列生成 1 个 token）
            decode_tokens += (-num_tokens)
            decode_time += t_elapsed
            decode_steps += 1

    # ========== 输出结果 ==========
    print("=" * 60)
    print("Prefill Phase:")
    print(f"  Tokens: {prefill_tokens}")
    print(f"  Time:   {prefill_time:.4f} s")
    if prefill_time > 0:
        print(f"  Throughput: {prefill_tokens / prefill_time:.2f} tok/s")
    print()
    print("Decode Phase:")
    print(f"  Tokens: {decode_tokens}")
    print(f"  Steps:  {decode_steps}")
    print(f"  Time:   {decode_time:.4f} s")
    if decode_time > 0:
        decode_tps = decode_tokens / decode_time
        print(f"  Throughput: {decode_tps:.2f} tok/s")

        # 计算资源利用率
        bytes_per_token = 688_128  # 之前分析的结果
        effective_bw_GiBs = decode_tps * bytes_per_token / (1024**3)
        print()
        print("Bandwidth Analysis:")
        print(f"  Per-token PCIe transfer: {bytes_per_token:,} B ({bytes_per_token/1024:.1f} KiB)")
        print(f"  Effective PCIe bandwidth: {effective_bw_GiBs:.4f} GiB/s")
        print(f"  If PCIe BW = 22.73 GiB/s → utilization = {effective_bw_GiBs/22.73*100:.2f}%")
        print(f"  Theoretical upper bound @ 22.73 GiB/s = {22.73*1024**3/bytes_per_token:.0f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()