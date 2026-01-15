import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from nanovllm.utils.secure import set_security_config

def main():
    t0 = time.perf_counter()
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    set_security_config(
        enable_softmax_encrypt=True,
        enable_linear_noise=True,
        encrypt_on_cpu=True,
        decrypt_on_cpu=True,
        tee_strict_mode=True,
        noise_pool_size=16,
        noise_scale=0.05,
        seed=1234,
    )
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    t1 = time.perf_counter()
    print(f"\nTotal time: {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    main()
