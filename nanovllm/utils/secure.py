import math
import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SecurityConfig:
    enable_softmax_encrypt: bool = True
    enable_linear_noise: bool = True
    encrypt_on_cpu: bool = True
    decrypt_on_cpu: bool = True
    tee_strict_mode: bool = True
    noise_pool_size: int = 16
    noise_scale: float = 0.05
    seed: int = 1234


_CONFIG = SecurityConfig()


def get_security_config() -> SecurityConfig:
    return _CONFIG


def set_security_config(**kwargs):
    global _CONFIG
    for k, v in kwargs.items():
        if hasattr(_CONFIG, k):
            setattr(_CONFIG, k, v)


def _manual_seed(seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def orthogonal_matrix(dim: int, dtype: torch.dtype, device: torch.device, seed: Optional[int] = None) -> torch.Tensor:
    """
    生成维度为 dim 的正交矩阵 R，使得 R^{-1} = R^T。
    使用 QR 分解保证数值稳定，R 取 Q。
    """
    g = _manual_seed(_CONFIG.seed if seed is None else seed)
    # 生成随机矩阵并进行 QR 分解，使用cpu进行采样
    a = torch.randn((dim, dim), dtype=torch.float32, device="cpu", generator=g)
    q, r = torch.linalg.qr(a)

    q = q.to(dtype=dtype, device=device)
    return q


class NoisePool:
    """
    噪声池：提供固定数量的输入侧噪声向量 r（形状 [in_features]），
    并在权重确定后预计算补偿项 rW（形状 [out_features]）。
    在前向中使用“加性线性噪声加密”：
      - 从噪声池中随机选取两个向量 r1、r2；
      - 采样两个随机缩放系数 alpha、beta，形成组合噪声 r = alpha*r1 + beta*r2；
      - 在线性输出侧添加对应补偿 rW = alpha*(r1 W^T) + beta*(r2 W^T)。
    同时对被选取的两个向量执行“保持分量的旋转混合”以生成新的两个向量，回填到噪声池中，
    并更新其对应的 rW。
    """

    def __init__(self, in_features: int, out_features: int, pool_size: int, noise_scale: float, seed: int = 1234):
        self.in_features = in_features
        self.out_features = out_features
        self.pool_size = pool_size
        self.noise_scale = noise_scale
        self.seed = seed
        self._rng = _manual_seed(seed)

        # r_pool: [P, in_features] 存在 CPU 上
        self.r_pool_cpu = torch.randn((pool_size, in_features), device="cpu", generator=self._rng) * noise_scale
        self.r_pool_cpu = self.r_pool_cpu.to(dtype=torch.float32, device="cpu")

        # rW_pool: [P, out_features]，在 set_weight 之后计算
        self.rw_pool_cpu: Optional[torch.Tensor] = None
        # 保存权重，便于在旋转更新后仅更新对应行的 rW
        self._w_cpu: Optional[torch.Tensor] = None

    def set_weight(self, weight_shard: torch.Tensor):
        """
        在权重装载/更新后调用，预计算 rW。
        weight_shard: 形状 [out_features, in_features] 的本 rank 权重切片。
        计算：rW = r @ W^T -> [P, out_features]
        """
        assert weight_shard.dim() == 2
        out_features, in_features = weight_shard.shape
        assert out_features == self.out_features and in_features == self.in_features
        # 使用 CPU 计算补偿
        w_cpu = weight_shard.detach().to(dtype=torch.float32, device="cpu")
        self._w_cpu = w_cpu
        self.rw_pool_cpu = self.r_pool_cpu @ w_cpu.T
    def _rotate_and_update(self, idx1: int, idx2: int):
        """
        对 r_pool[idx1], r_pool[idx2] 执行保持分量的旋转混合：
          [r1'; r2'] = [cos -sin; sin cos] @ [r1; r2]
        然后更新 r_pool 与对应的 rW 行。
        """
        # 随机角度 θ ∈ [0, 2π)
        theta = torch.rand((), device="cpu", generator=self._rng).item() * 2.0 * math.pi
        c = math.cos(theta)
        s = math.sin(theta)

        r1 = self.r_pool_cpu[idx1]
        r2 = self.r_pool_cpu[idx2]
        # 逐分量旋转（共享一个 θ）
        r1_new = c * r1 - s * r2
        r2_new = s * r1 + c * r2

        self.r_pool_cpu[idx1] = r1_new
        self.r_pool_cpu[idx2] = r2_new

        if self._w_cpu is not None:
            # 仅更新两行 rW
            self.rw_pool_cpu[idx1] = r1_new @ self._w_cpu.T
            self.rw_pool_cpu[idx2] = r2_new @ self._w_cpu.T

    def sample(self, index: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """
        采样两个噪声向量并进行线性组合：
          r = alpha * r1 + beta * r2
          rW = alpha * rW1 + beta * rW2
        返回 (r_cpu, rW_cpu, (idx1, idx2))，并将 (r1, r2) 经过旋转混合后写回池中。
        """
        assert self.rw_pool_cpu is not None and self._w_cpu is not None, (
            "NoisePool: rW 尚未预计算，请在权重加载后调用 set_weight()。"
        )

        if index is None:
            # 采样两个不相同的索引
            idx1 = int(torch.randint(0, self.pool_size, (1,), device="cpu", generator=self._rng).item())
            idx2 = int(torch.randint(0, self.pool_size - 1, (1,), device="cpu", generator=self._rng).item())
            if idx2 >= idx1:
                idx2 += 1
        else:
            # 当外部指定一个 index 时，第二个索引仍然随机选取且不等于 index
            idx1 = int(index)
            idx2 = int(torch.randint(0, self.pool_size - 1, (1,), device="cpu", generator=self._rng).item())
            if idx2 >= idx1:
                idx2 += 1

        r1 = self.r_pool_cpu[idx1]
        r2 = self.r_pool_cpu[idx2]
        rw1 = self.rw_pool_cpu[idx1]
        rw2 = self.rw_pool_cpu[idx2]

        # 随机缩放系数 alpha, beta（均匀分布于 [0, 1)）
        alpha = torch.rand((), device="cpu", generator=self._rng).to(dtype=torch.float32)
        beta = torch.rand((), device="cpu", generator=self._rng).to(dtype=torch.float32)

        r = alpha * r1 + beta * r2
        rw = alpha * rw1 + beta * rw2

        # 使用旋转混合生成两个新向量并回填池中
        self._rotate_and_update(idx1, idx2)

        return r, rw, (idx1, idx2)
