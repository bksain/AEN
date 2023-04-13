import torch
from einops import rearrange, repeat
from torch import nn, einsum
import math


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


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
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                                              Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TFormer(nn.Module):
    def __init__(self, num_patches=16, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = x.contiguous().view(-1, 16, 512)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n+1)]
        x = self.spatial_transformer(x)
        x = x[:, 0]

        return x
    
class TFormer_SP(nn.Module):
    def __init__(self, num_patches=16, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        dim1 = 784
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dim1))
        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches + 1, dim1))
        self.spatial_transformer1 = Transformer(dim1, depth, heads, dim_head, mlp_dim, dropout)

        
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim1))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches + 1, dim1))
        self.spatial_transformer2 = Transformer(dim1, depth, heads, dim_head, mlp_dim, dropout)

        
        self.cls_token3 = nn.Parameter(torch.randn(1, 1, dim1))
        self.pos_embedding3 = nn.Parameter(torch.randn(1, num_patches + 1, dim1))
        self.spatial_transformer3 = Transformer(dim1, depth, heads, dim_head, mlp_dim, dropout)
        

    def forward(self, x):
        x0 = x[0].contiguous().view(-1, 16, 512)
        b, n, _ = x0.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x0 = torch.cat((cls_tokens, x0), dim=1)
        x0 = x0 + self.pos_embedding[:, :(n+1)]
        x0 = self.spatial_transformer(x0)
        x0 = x0[:, 0]


        x1 = x[1].contiguous().view(-1, 16, 784)
        b, n, _ = x1.shape
        cls_tokens = repeat(self.cls_token1, '() n d -> b n d', b=b)
        x1 = torch.cat((cls_tokens, x1), dim=1)
        x1 = x1 + self.pos_embedding1[:, :(n + 1)]
        x1 = self.spatial_transformer1(x1)
        x1 = x1[:, 0]


        x2 = x[2].contiguous().view(-1, 16, 784)
        b, n, _ = x2.shape
        cls_tokens = repeat(self.cls_token2, '() n d -> b n d', b=b)
        x2 = torch.cat((cls_tokens, x2), dim=1)
        x2 = x2 + self.pos_embedding2[:, :(n + 1)]
        x2 = self.spatial_transformer2(x2)
        x2 = x2[:, 0]


        x3 = x[3].contiguous().view(-1, 16, 784)
        b, n, _ = x3.shape
        cls_tokens = repeat(self.cls_token3, '() n d -> b n d', b=b)
        x3 = torch.cat((cls_tokens, x3), dim=1)
        x3 = x3 + self.pos_embedding3[:, :(n + 1)]
        x3 = self.spatial_transformer3(x3)
        x3 = x3[:, 0]

        return [x0, x1, x2, x3]

class Dual_TFormer(nn.Module):
    def __init__(self, num_patches=16, dim=512, depth=3, heads=8, mlp_dim=1024, dim_head=64, dropout=0.0):
        super().__init__()

        self.dim_con = 256
        self.cls_token_con = nn.Parameter(torch.randn(1, 1, self.dim_con))
        self.pos_embedding_con = nn.Parameter(torch.randn(1, num_patches + 1, self.dim_con))
        self.spatial_transformer_con = Transformer(self.dim_con, depth, heads, dim_head, mlp_dim, dropout)

        self.dim_former = 256
        self.cls_token_former = nn.Parameter(torch.randn(1, 1, self.dim_former))
        self.pos_embedding_former = nn.Parameter(torch.randn(1, num_patches+1, self.dim_former))
        self.spatial_transformer_former = Transformer(self.dim_former, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):

        x_con = x[2].contiguous().view(-1, 16, self.dim_con)

        b, n, _ = x_con.shape
        cls_tokens = repeat(self.cls_token_con, '() n d -> b n d', b=b)
        x_con = torch.cat((cls_tokens, x_con), dim=1)
        x_con = x_con + self.pos_embedding_con[:, :(n + 1)]
        x_con = self.spatial_transformer_con(x_con)
        x_con = x_con[:, 0]


        x_former = x[3].contiguous().view(-1, 16, self.dim_former)

        b, n, _ = x_former.shape
        cls_tokens = repeat(self.cls_token_former, '() n d -> b n d', b=b)
        x_former = torch.cat((cls_tokens, x_former), dim=1)
        x_former = x_former + self.pos_embedding_former[:, :(n+1)]
        x_former = self.spatial_transformer_former(x_former)
        x_former = x_former[:, 0]

        x = torch.cat([x_con , x_former], dim=1)

        return x


def temporal_transformer():
    return TFormer_SP() #TFormer() # Dual_TFormer() #


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = temporal_transformer()
    model(img)
