import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from nanovllm.utils.secure import NoisePool, get_security_config

import torch.cuda.nvtx as nvtx


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

#按列并行划分的线性层
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        enable_mask: bool = True,      # 是否启用加性噪声掩码（x -> x + r）
        mask_scale: float = 0.05,      # 噪声强度
        layer_id: int | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)

        # 计算输出维度
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        
        # 调用父类初始化（列并行，内部会按 TP 分片输出尺寸）
        super().__init__(hidden_size, output_size, bias)

        sec = get_security_config()
        # 掩码/噪声配置
        self.enable_mask = enable_mask and sec.tee_strict_mode
        self.mask_scale = mask_scale
        self.hidden_size = hidden_size
        self.layer_id = layer_id

        self.decrypt_on_cpu = sec.decrypt_on_cpu
        pool_size = sec.noise_pool_size
        # 噪声池（输入维度：hidden_size，输出维度：本层的 out_features）
        if self.enable_mask:
            self._noise_pool = NoisePool(
                in_features=hidden_size,
                out_features=self.weight.shape[0],
                pool_size=pool_size,
                noise_scale=self.mask_scale,
                seed=sec.seed,
            )
        else:
            self._noise_pool = None

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
        # 权重更新后，更新噪声池的 rW 预计算
        if self._noise_pool is not None and param is self.weight:
            # 每次装载一段权重后更新一次 rW，待全部装载完成后会稳定。
            self._noise_pool.set_weight(self.weight.data)

    #前向传播时，先对输入x进行加密处理，然后再进行线性变换，最后解密输出
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 安全配置
        sec = get_security_config()
        cpu_encrypt = sec.encrypt_on_cpu and sec.tee_strict_mode
        cpu_decrypt = sec.decrypt_on_cpu and sec.tee_strict_mode
        # 未启用掩码/噪声：仍需保证设备一致，并在严格TEE下将输出移回CPU
        if not self.enable_mask or self._noise_pool is None:
            if x.device != self.weight.device:
                x = x.to(self.weight.device)
            y = F.linear(x, self.weight, self.bias)
            if sec.tee_strict_mode:
                y = y.to(device="cpu", dtype=y.dtype)
            return y

        with nvtx.range("Sample_Noise(CPU)"):
            # 采样噪声 r 及补偿 rW（均为 CPU 张量，形状：[in_features], [out_features]）
            r_cpu, rw_cpu, _ = self._noise_pool.sample()

        # 在 CPU 进行加密（TEE）：x' = x + r
        if cpu_encrypt:
            with nvtx.range("Encrypt_Linear(CPU)"):
                x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
                r = r_cpu.to(dtype=x_cpu.dtype)
                if x_cpu.dim() == 2:
                    r_b = r.unsqueeze(0)
                else:
                    view_shape = [1] * (x_cpu.dim() - 1) + [r.shape[0]]
                    r_b = r.view(*view_shape)
                x_masked_cpu = x_cpu + r_b
            # 传送给不安全 GPU 做线性
            with nvtx.range("Data_H2D"):
                x_masked = x_masked_cpu.to(device=self.weight.device, dtype=x.dtype)
        else:
            # 在当前设备直接加密（不安全环境），仅用于测试或性能对比
            r = r_cpu.to(device=x.device, dtype=x.dtype)
            if x.dim() == 2:
                r_b = r.unsqueeze(0)
            else:
                view_shape = [1] * (x.dim() - 1) + [r.shape[0]]
                r_b = r.view(*view_shape)
            x_masked = x + r_b

        with nvtx.range("Generate_QKV(GPU)"):
        # 在 GPU 上计算加密后的线性：y' = (x + r) W^T + b
            y_masked = F.linear(x_masked, self.weight, self.bias)

        
        # 解密：减去 rW
        if cpu_decrypt:
            with nvtx.range("Data_D2H"):
            # 将 y' 回传到 CPU，在 CPU 上做补偿，再返回设备
                y_cpu = y_masked.detach().to(device="cpu", dtype=torch.float32)
            
            with nvtx.range("Decrypt_Linear(CPU)"):
                rw = rw_cpu.to(dtype=y_cpu.dtype)
                if y_cpu.dim() == 2:
                    y_cpu = y_cpu - rw.unsqueeze(0)
                else:
                    view_shape = [1] * (y_cpu.dim() - 1) + [rw.shape[0]]
                    y_cpu = y_cpu - rw.view(*view_shape)
                # 严格TEE：QKV线性(GPU)外的后续计算尽量在CPU进行
                if sec.tee_strict_mode:
                    y = y_cpu.to(dtype=y_masked.dtype)
                else:
                    y = y_cpu.to(device=y_masked.device, dtype=y_masked.dtype)
        else:
            # 在 GPU 上完成补偿
            rw = rw_cpu.to(device=y_masked.device, dtype=y_masked.dtype)
            if y_masked.dim() == 2:
                y = y_masked - rw.unsqueeze(0)
            else:
                view_shape = [1] * (y_masked.dim() - 1) + [rw.shape[0]]
                y = y_masked - rw.view(*view_shape)

        # 严格TEE下，保证QKV线性后的输出在CPU，便于后续CPU模块（RMSNorm/Rotary等）
        if sec.tee_strict_mode:
            y = y.to(device="cpu", dtype=y.dtype)

        return y

#按行并行划分的线性层
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
