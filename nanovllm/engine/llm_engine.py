import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

import torch.cuda.nvtx as nvtx


class LLMEngine:
    #解析配置、启动若干个并行的 worker 进程（用于张量并行 TP）、在主进程创建 0 号 worker、加载 tokenizer、初始化调度器，并注册退出时的清理函数。
    def __init__(self, model, **kwargs):
        #清洗配置参数，只保留 Config 中定义的字段
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        #启动多个子进程用于张量并行
        self.ps = []
        self.events = []
        self._step_id = 0
        ctx = mp.get_context("spawn")   #用 multiprocessing 的 spawn 启动方式

        #根据TP创建子进程，这个子进程启动后会运行 ModelRunner(config, rank=i, event=event)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()     #启动子进程
            self.ps.append(process)
            self.events.append(event)
            
        #在主进程创建 rank=0 的 ModelRunner （rank=0），不需要Process，就在当前进程跑，并且他知道其他worker的events，可以通知它们退出
        self.model_runner = ModelRunner(config, 0, self.events)

        #加载 tokenizer，并将eos_token_id存入config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        #创建调度器scheduler
        
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
    #上面注册的退出时的清理函数，通知所有 worker 进程退出并等待它们结束。
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        with nvtx.range("AddRequest"):
            #如果是字符串提示，则先用 tokenizer 编码成 token ID 列表
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)

            #创建一个 Sequence 对象并添加到调度器中
            seq = Sequence(prompt, sampling_params)

            #将请求添加到调度器
            self.scheduler.add(seq)

    def step(self):
        step_id = self._step_id
        self._step_id += 1

        with nvtx.range(f"Step_{step_id}"):
            with nvtx.range("Scheduler"):
                #区分prefill和decode阶段，获取当前需要处理的序列列表和是否是prefill阶段
                seqs, is_prefill = self.scheduler.schedule()

            with nvtx.range("ModelRunner"):
                #调用 model_runner 进行推理，获取新生成的 token IDs
                token_ids = self.model_runner.call("run", seqs, is_prefill, step_id)

            with nvtx.range("Postprocess"):
                #把 token 写回请求对象里，检查stop条件，更新seq状态，并决定是否结束
                self.scheduler.postprocess(seqs, token_ids)

            #收集本轮已经完成的输出
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
            return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()
    

    # 生成文本的主方法，接受多个提示和采样参数，逐步生成文本直到所有请求完成，并返回生成的文本和对应的 token IDs。
    def generate(
        self,
        prompts: list[str] | list[list[int]], #既能接受字符串提示，也能接受 token ID 列表
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        self._step_id = 0
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        #对采样参数进行处理，如果是单个参数则扩展为与提示数量相同的列表
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        #为每个提示添加生成请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            #一个step中新生成的token以及一共生成的token数量
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            #把某个请求当前累计生成的 token 存起来
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        with nvtx.range("Generate_Finish"):
            #按 seq_id 从小到大，把结果整理成一个 list
            outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
            #将 token IDs 解码成文本
            outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
            if use_tqdm:
                pbar.close()
            return outputs
