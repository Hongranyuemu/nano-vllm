# NanoVLLM TEE 安全推理 — 工作总结

## 一、项目背景

NanoVLLM 是一个基于 vLLM 架构的轻量级推理引擎，核心特点是支持 **TEE（可信执行环境）安全推理**：CPU 可信、GPU 不可信。在 `tee_strict_mode=True` 下，模型的各模块被拆分到 CPU 和 GPU 上执行：

- **GPU 上执行**：QKV 线性层（加密后的矩阵乘法）、FlashAttention（加密后的 Q·K^T 和 S·V）
- **CPU 上执行**：Embedding、LayerNorm、RoPE、QK Norm、正交加密/解密、加性噪声加密/解密、Output Projection、MLP（gate_up_proj + SiLU + down_proj）、LM Head、Sampler

安全机制包括两种：
1. **密文乘明文（加性噪声）**：对 QKV 线性层，CPU 端加噪 c(x)=x+αn_i+βn_j，GPU 做加密后的矩阵乘法，CPU 端减去预计算的补偿 rW 解密
2. **密文乘密文（正交矩阵）**：对 Q·K^T，用正交矩阵 R 加密 Q 和 K，使得 (QR)(KR)^T = QK^T，GPU 无法获取原始 Q/K

模型参数（Qwen3-0.6B）：
- 层数 N_L = 28
- Query 头数 H_Q = 16，KV 头数 H_KV = 8
- Head dim D = 128，hidden_size = 2048
- dtype = bf16（b = 2 bytes）

---

## 二、代码与 inference.pdf 设计方案的对比分析

对照 inference.pdf 的需求，分析了代码的满足情况：

### 已满足
- 非线性操作在 CPU 完成（RMSNorm、SiLU、RoPE 等均在 CPU）
- 密文乘明文的加性噪声加密（NoisePool + α/β 组合 + 预计算 rW）
- 噪声向量旋转混合（每次采样后旋转更新池中向量）
- 密文乘密文的正交矩阵加密 Q·K^T

### 未满足
- softmax 在 GPU 执行（违反非线性必须在 CPU 的要求）
- S·V 未加密（V 以明文送入 GPU）
- rW 旋转更新做了不必要的矩阵乘法（应直接用 rW 的线性组合，O(d) 而非 O(d²)）
- 噪声向量初始化未使用 SVD（代码用随机高斯，PDF 要求用激活值的 SVD 基向量）
- 高效旋转矩阵未实现（PDF 提出分块对角旋转和置换矩阵，代码用完整 d×d 矩阵）

---

## 三、Decode 阶段每 Token PCIe 传输量分析

### 分析方法

逐行追踪代码中所有 `.to("cuda")` 和 `.to("cpu")` 调用，统计每生成一个 token 实际跨 PCIe 传输的数据量。

### 每层有 4 次 PCIe 传输

| # | 方向 | 数据内容 | 形状 | 字节数 |
|---|------|---------|------|--------|
| ① | H2D | 加密输入 → GPU 做 QKV 线性 | [1, 2048] bf16 | 4,096 |
| ② | D2H | QKV 线性输出 → CPU 解密 | [1, 4096] bf16 | 8,192 |
| ③ | H2D | 加密的 q,k,v → GPU 做 Attention | [1,16,128]+[1,8,128]×2 bf16 | 8,192 |
| ④ | D2H | Attention 输出 → CPU | [1, 16, 128] bf16 | 4,096 |
| | | **每层合计** | | **24,576 B** |

### 公式

```
每层传输 = (d_model + (H_Q+2·H_KV)·D + (H_Q+2·H_KV)·D + H_Q·D) × b
         = 2 × (2·H_Q + 2·H_KV) × D × b
         = 2 × 48 × 128 × 2
         = 24,576 B
```

### 28 层总计

```
每 token 传输量 = 28 × 24,576 = 688,128 B ≈ 672 KiB
```

每 token PCIe 传输次数 = 28 × 4 = 112 次

注：Embedding、LM Head、Sampler 均在 CPU，MLP/LayerNorm/RoPE/oProj 也在 CPU，不涉及 PCIe 传输。prepare_decode 的元数据（slot_mapping 等）传输量可忽略。

---

## 四、理论吞吐上限

### 公式

```
理论上限 (tok/s) = PCIe 带宽 (B/s) / 每 token 传输量 (B/tok)
```

### bs=8 实测带宽 22.73 GiB/s 下的计算

```
理论上限 = 22.73 × 1024³ / 688,128 ≈ 35,468 tok/s
```

---

## 五、实测结果（187 服务器，bs=8）

用 bench_decode.py 分离 prefill/decode 阶段：

```
Prefill Phase:
  Tokens: 5157
  Time:   5.8461 s
  Throughput: 882.12 tok/s

Decode Phase:
  Tokens: 2040
  Steps:  255
  Time:   44.6575 s
  Throughput: 45.68 tok/s

Bandwidth Analysis:
  Per-token PCIe transfer: 688,128 B (672.0 KiB)
  Effective PCIe bandwidth: 0.0293 GiB/s
  PCIe utilization: 0.13%
  Theoretical upper bound @ 22.73 GiB/s = 35,467 tok/s
```

### 关键结论

- PCIe 利用率仅 0.13%，PCIe 带宽完全不是瓶颈
- 每步 175ms（44.66s / 255 步），每层约 6.25ms
- 瓶颈在 CPU 计算：MLP、LayerNorm、RoPE、加解密全在 CPU 上

---

## 六、Overlap 优化方案分析

### 思路

把 batch 拆成 K 个微批次（如 bs=8 → 2×4），让 GPU 处理微批次 A 时 CPU 同时处理微批次 B，实现 CPU/GPU 并行。

### 时序对比

当前（串行）：
```
CPU: [8条LN+加密]        [8条解密+RoPE]        [8条oProj+MLP]
GPU:              [8条QKV]              [8条Attn]
```

Overlap（拆成 A/B 交替）：
```
CPU: [A:LN+加密][B:LN+加密][A:解密+RoPE][B:解密+RoPE][A:MLP][B:MLP]
GPU:            [A:QKV]    [B:QKV]      [A:Attn]     [B:Attn]
                ↑ 并行 ↑   ↑ 并行 ↑     ↑ 并行 ↑     ↑ 并行 ↑
```

### 可行性

- A 和 B 是不同序列，没有数据依赖，可以交错
- 需要把 Attention 拆成 3 个子阶段：cpu_pre → gpu → cpu_post
- 需要按微批次切分 Context（slot_mapping, context_lens, block_tables）

### 预期效果

```
没有 overlap：每层耗时 = T_cpu + T_gpu
有 overlap：  每层耗时 ≈ T_cpu（GPU 的工作被隐藏在 CPU 间隙里）
提升比例 = T_gpu / (T_cpu + T_gpu)
```

由于 CPU 占比 >95%，overlap 预计提升 5-10%，不会带来数量级改变。

### 更根本的优化方向

1. **把 MLP 卸载回 GPU** — MLP 是 CPU 最重的计算，需讨论安全模型是否允许
2. **增大 batch size** — 用大显存的卡，摊薄 CPU 每 token 开销
3. **CPU 多线程加速** — torch.set_num_threads() 或 MKL 并行
4. **减少 PCIe 往返次数** — 合并小传输、用 pinned memory

---

## 七、产出文件

| 文件 | 作用 |
|------|------|
| bench_decode.py | 分离 prefill/decode 吞吐的测量脚本 |
| overlap_decode.py | 微批次流水线 overlap 实现（monkey-patch 方式，不修改原始代码） |
| decode_bandwidth_analysis.md | 完整的 PCIe 传输量分析文档 |