import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append("../model")
from model import GPT, LayerNorm, Block

class Time_Embedding(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.dim = in_features

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        device = inputs.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1) 
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = inputs[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TDiffusion(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.alpha = config.alpha
        self.total_timestep = config.total_timestep
        self.time_steps = Time_Embedding(config.time_dim)
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) \
                        for _ in range(config.n_layer*2)])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.noise_rate = []
        rate_sum = 0.0
        for i in range(10):
            self.noise_rate.append(rate_sum)
            rate_sum += 1/2**(i+1)

    # Not still implementation
    # Calculate the value of how much noise is added to each word vector
    def get_noise_rate(self, latents, t, time_step):
        return None, None

    def add_noise(self, latents, noise_rate):
        noise = torch.randn_like(latents)
        return (1- noise_rate) * latents + noise_rate * noise

    def forward(self, idx, targets=None, time_step=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        noise_rate, noise_step_emb = self.get_noise_rate(tok_emb, t, time_step)
        noised_tok_emb, noise = self.add_noise(tok_emb, noise_rate)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(noised_tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x) + noise_step_emb
        x = self.transformer.ln_f(x)
        noise_pred = x + noised_tok_emb # Residual structure for easier learning

        # denoise
        x = self.step(noised_tok_emb, noise_pred, t, time_step)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = self.alpha * F.mse_loss(noise_pred, noise)
            loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
GPT = TDiffusion