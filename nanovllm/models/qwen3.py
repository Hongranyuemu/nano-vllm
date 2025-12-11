import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.nvtx import nvtx_range
from nanovllm.utils.secure import get_security_config, orthogonal_matrix


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,

        layer_id: int = 0,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.layer_id = layer_id

        # QK^T加密：使用正交矩阵 R，满足 R^{-1} = R^T
        sec = get_security_config()
        self.encrypt_enabled = sec.enable_softmax_encrypt and sec.tee_strict_mode
        if self.encrypt_enabled:
            # 使用 float32 存储 R，保持正交性，避免 bf16 破坏 R R^T ≈ I
            R = orthogonal_matrix(self.head_dim, dtype=torch.float32, device=torch.device('cpu'))
        else:
            R = torch.eye(self.head_dim, dtype=torch.float32, device='cpu')

        self.encrypt_R = nn.Parameter(R, requires_grad=False)

        # 从安全配置读取线性噪声开关与强度
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            enable_mask=get_security_config().enable_linear_noise,
            mask_scale=get_security_config().noise_scale,
            layer_id=layer_id,
        )
        
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # 严格TEE：除QKV线性与注意力核外，其余尽量在CPU上执行
        if get_security_config().tee_strict_mode:
            self.q_norm.to("cpu")
            self.k_norm.to("cpu")
            self.o_proj.to("cpu")
            self.rotary_emb.to("cpu")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 形成 QKV 矩阵（QKV 线性在GPU，解密后可根据配置保留在CPU）
        with nvtx_range("tee::qkv_project"):
            qkv = self.qkv_proj(hidden_states)
        with nvtx_range("tee::split_qkv"):
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        with nvtx_range("tee::q_norm_cpu"):
            q = self.q_norm(q.view(-1, self.num_heads, self.head_dim))
        with nvtx_range("tee::k_norm_cpu"):
            k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim))
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        with nvtx_range("tee::rotary_cpu"):
            q, k = self.rotary_emb(positions, q, k)

        # 应用正交加密（仅将加密后的 q/k 与 v 送到 GPU 做注意力）
        sec = get_security_config()
        if self.encrypt_enabled:
            use_cpu_encrypt = sec.encrypt_on_cpu and sec.tee_strict_mode
            encrypt_label = "tee::encrypt_qk_cpu" if use_cpu_encrypt else "tee::encrypt_qk"
            with nvtx_range(encrypt_label):
                if use_cpu_encrypt:
                    R_cpu = self.encrypt_R.detach().to(device="cpu", dtype=torch.float32)
                    q = torch.matmul(q.detach().to(device="cpu", dtype=torch.float32), R_cpu)
                    k = torch.matmul(k.detach().to(device="cpu", dtype=torch.float32), R_cpu)
                else:
                    R = self.encrypt_R.to(device=q.device, dtype=q.dtype)
                    q = torch.matmul(q, R)
                    k = torch.matmul(k, R)

        if sec.tee_strict_mode:
            # 加密后的 q/k/v 先转到 GPU；FlashAttention 需要 bf16/fp16。
            target_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            with nvtx_range("tee::cpu_to_cuda_qkv"):
                q_gpu = q.to(device="cuda", dtype=target_dtype)
                k_gpu = k.to(device="cuda", dtype=target_dtype)
                v_gpu = v.to(device="cuda", dtype=target_dtype)
            # 在 GPU 上计算注意力（包含 s = softmax(QK^T) 和 s @ V）
            o_gpu = self.attn(q_gpu, k_gpu, v_gpu)
            # 将注意力输出带回 CPU，后续输出投影在 CPU 进行
            with nvtx_range("tee::cuda_to_cpu_attn"):
                o_cpu = o_gpu.to(device="cpu", dtype=o_gpu.dtype)
            with nvtx_range("tee::o_proj_cpu"):
                output = self.o_proj(o_cpu.flatten(1, -1))
            return output
        else:
            o = self.attn(q, k, v)
            with nvtx_range("tee::o_proj_gpu"):
                output = self.o_proj(o.flatten(1, -1))
            return output

class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
            layer_id=layer_id,
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 使layernorm和mlp在CPU上执行
        if get_security_config().tee_strict_mode:
            self.input_layernorm.to("cpu")
            self.post_attention_layernorm.to("cpu")
            self.mlp.to("cpu")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with nvtx_range("tee::input_layernorm"):
            if residual is None:
                hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        with nvtx_range("tee::post_attention_layernorm"):
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        with nvtx_range("tee::mlp_cpu"):
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        from nanovllm.utils.secure import get_security_config
        if get_security_config().tee_strict_mode:
            # 严格 TEE 下将 embedding 常驻 CPU，避免输入 token 在 GPU 上暴露
            self.embed_tokens.to("cpu")
            self.norm.to("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        sec = get_security_config()
        if sec.tee_strict_mode:
            # 输入 ID / 位置编码先迁移到可信 CPU，保证嵌入阶段不触碰 GPU
            with nvtx_range("tee::embed_inputs_to_cpu"):
                input_ids = input_ids.to("cpu")
                positions = positions.to("cpu")
        embed_label = "tee::embed_tokens_cpu" if sec.tee_strict_mode else "embed_tokens"
        with nvtx_range(embed_label):
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        for idx, layer in enumerate(self.layers):
            range_name = f"tee::decoder_layer_{idx}"
            with nvtx_range(range_name):
                hidden_states, residual = layer(positions, hidden_states, residual)
        with nvtx_range("tee::final_norm"):
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        from nanovllm.utils.secure import get_security_config
        if get_security_config().tee_strict_mode:
            self.lm_head.to("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        with nvtx_range("tee::lm_head"):
            return self.lm_head(hidden_states)
