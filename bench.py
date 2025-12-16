import argparse
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
from nanovllm.utils.secure import set_security_config


def parse_args():
    parser = argparse.ArgumentParser(description="Nano-VLLM benchmark harness")
    parser.add_argument("--fixed_gen_tokens", type=int, default=256, help="Number of tokens each request must generate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of prompts/sequences per run.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    seed(0)
    num_seqs = args.batch_size
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

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
    llm = LLM(path, enforce_eager=True, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [
        SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.fixed_gen_tokens) for _ in range(num_seqs)
    ]


    llm.generate(["Benchmark: "], SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=args.fixed_gen_tokens))
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
