"""Small explicit decoder; every component is constructed in lesson 13."""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, width, head_size, context):
        super().__init__()
        self.query = nn.Linear(width, head_size, bias=False)
        self.key = nn.Linear(width, head_size, bias=False)
        self.value = nn.Linear(width, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(context, context)).bool())

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** .5)
        length = x.shape[1]
        scores = scores.masked_fill(~self.mask[:length, :length], float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return weights @ v


class MultiHead(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        assert width % heads == 0
        self.heads = nn.ModuleList([AttentionHead(width, width // heads, context) for _ in range(heads)])
        self.projection = nn.Linear(width, width)

    def forward(self, x):
        pieces = [head(x) for head in self.heads]
        return self.projection(torch.cat(pieces, dim=-1))


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention, self.ff = MultiHead(width, heads, context), FeedForward(width)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ff(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, context=16, width=24, heads=2, depth=1):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*[Block(width, heads, context) for _ in range(depth)])
        self.norm, self.output = nn.LayerNorm(width), nn.Linear(width, vocab_size)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        return self.output(self.norm(self.blocks(x)))

    @torch.no_grad()
    def generate(self, tokens, steps, temperature=1.):
        for _ in range(steps):
            logits = self(tokens[:, -self.context:])[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
