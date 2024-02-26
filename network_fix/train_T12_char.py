import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append("../model")
from model import GPT

class T12(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.lm_heads = nn.ModuleList([nn.Linear(config.n_embd, config.vocab_size, bias=False) \
                        for _ in range(config.n_layer*2)])

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        x_keeper = []
        for block in self.transformer.h:
            x = block(x)
            x_keeper.append(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            loss = 0
            # if we are given some desired targets also calculate the loss
            for i in range(len(x_keeper)):
                if len(x_keeper)-1 <= i:
                    logits = self.lm_heads[2*i](x_keeper[i])
                    loss += F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                    break
                logits1 = self.lm_heads[2*i](x_keeper[i])
                logits2 = self.lm_heads[2*i+1](x_keeper[i])
                loss += F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1), ignore_index=-1)
                logits2 = logits2[:,1:,:]
                targets_shift = targets[:,1:]
                loss += F.cross_entropy(logits2.reshape(-1, logits2.size(-1)), targets_shift.reshape(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
GPT = T12