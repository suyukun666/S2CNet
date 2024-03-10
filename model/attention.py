import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
        

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, x_ma, attn_bias):
        b, n, _, h = *x.shape, self.heads
        kv = self.to_kv(x).chunk(2, dim = -1)
        q = self.to_q(x_ma)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), kv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots + attn_bias.view(b, h, n, n)
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class GraphAttention(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        self.fc_gcn = nn.Linear(dim, dim, bias=False)
        self.nl_gcn = nn.LayerNorm([dim])

    def forward(self, x, attn_bias):
        add_attn = attn_bias[0] + attn_bias[1]
        for attn, ff in self.layers:
            x_ma = self.aggregation(x, attn_bias)
            x = attn(x, x_ma=x_ma, attn_bias=add_attn)
            x = ff(x)
        return x
    
    def aggregation(self, x, attn_bias):
        A, S = attn_bias[0], attn_bias[1]
        A = torch.mean(A, dim=1)
        S = torch.mean(S, dim=1)
        G = torch.softmax(A * S, dim=2)
        x_ma = self.fc_gcn(torch.matmul(G, x))
        x_ma = self.nl_gcn(x_ma)
        x_ma = F.relu(x_ma)
        return x_ma

