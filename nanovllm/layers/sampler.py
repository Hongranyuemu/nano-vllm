import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if logits.device != temperatures.device:
            logits = logits.to(temperatures.device)
        greedy_mask = temperatures <= 1e-10
        temps = temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        logits = logits.float().div_(temps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        if greedy_mask.any():
            greedy_tokens = logits.argmax(dim=-1)
            sample_tokens = sample_tokens.clone()
            sample_tokens[greedy_mask] = greedy_tokens[greedy_mask]
        return sample_tokens
