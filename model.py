
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from mamba_ssm.modules.mamba_simple import Mamba, Block
from collections import namedtuple
import numpy as np

Prediction = namedtuple("Prediction", ("denoised"))

class SinusoidalEmbedding(nn.Module):
    def __init__(self, min_frequency =1.0, max_frequency =1000.0, embedding_dims=32):
        super(SinusoidalEmbedding, self).__init__()

        min_f = np.log(min_frequency)
        max_f = np.log(max_frequency)
        frequencies = torch.exp(torch.linspace(min_f, max_f, embedding_dims // 2))
        self.register_buffer("angular_speeds", 2.0 * torch.pi * frequencies)

    def forward(self, x):
        return torch.cat(
            [torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, is_causal=False, dropout=0.0, n_heads=8):
        super().__init__()
        self.is_causal = is_causal
        self.dropout = dropout
        self.n_heads = n_heads
        self.qkv_rearrange = Rearrange("bs n (h d) -> bs h n d", h=self.n_heads)
        self.out_rearrange = Rearrange("bs h n d -> bs n (h d)", h=self.n_heads)

    def forward(self, q, k, v, attn_mask=None):
        assert q.size(-1) == k.size(-1)
        assert k.size(-2) == v.size(-2)

        q, k, v = [self.qkv_rearrange(x) for x in [q, k, v]]

        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.is_causal,
            dropout_p=self.dropout if self.training else 0)

        out = self.out_rearrange(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout=0.0, n_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.attention = MultiHeadAttention(is_causal, dropout, n_heads)

    def forward(self, x):
        q, k, v = self.qkv_proj(x).split(self.embed_dim, dim=2)
        return self.attention(q, k, v)


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, is_causal=False, dropout=0, n_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attention = MultiHeadAttention(is_causal, dropout, n_heads)

    def forward(self, x, y):
        q = self.q_proj(x)
        k, v = self.kv_proj(y).split(self.embed_dim, dim=2)
        return self.attention(q, k, v)
        


class MLP(nn.Module):
    def __init__(self, embed_dim, multiplier=4, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, multiplier * embed_dim, kernel_size=1, padding="same"),
            nn.Conv2d(
                multiplier * embed_dim,
                multiplier * embed_dim,
                kernel_size=3,
                padding="same",
                groups=multiplier * embed_dim,
            ),  # <- depthwise conv
            nn.GELU(),
            nn.Conv2d(multiplier * embed_dim, embed_dim, kernel_size=1, padding="same"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        w = h = int(np.sqrt(x.size(1)))  # only square images for now
        x = rearrange(x, "bs (h w) d -> bs d h w", h=h, w=w)
        x = self.mlp(x)
        x = rearrange(x, "bs d h w -> bs (h w) d")
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        is_causal: bool,
        mlp_multiplier: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = SelfAttention(embed_dim, is_causal, dropout, n_heads=embed_dim // 32)
        self.cross_attention = CrossAttention(
            embed_dim, is_causal=False, dropout=0, n_heads=embed_dim // 32
        )
        self.mlp = MLP(embed_dim, mlp_multiplier, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # assert x != None, "err, x is none"
        x = self.self_attention(self.norm1(x)) + x
        x = self.cross_attention(self.norm2(x), y) + x
        x = self.mlp(x)
        return x


class DenoiserTransBlock(nn.Module):
    def __init__(
        self,
        patch_size: int,
        img_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.mlp_multiplier = mlp_multiplier

        seq_len = int((self.img_size / self.patch_size) * (self.img_size / self.patch_size))
        patch_dim = self.n_channels * self.patch_size * self.patch_size

        self.patchify_and_embed = nn.Sequential(
            nn.Conv2d(
                self.n_channels,
                patch_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),
            Rearrange("bs d h w -> bs (h w) d"),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        ).to(torch.float32)

        self.rearrange2 = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=int(self.img_size / self.patch_size),
            p1=self.patch_size,
            p2=self.patch_size,
        )

        self.pos_embed = nn.Embedding(seq_len, self.embed_dim)
        self.register_buffer("precomputed_pos_enc", torch.arange(0, seq_len).long())

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=self.embed_dim,
                    mlp_multiplier=self.mlp_multiplier,
                    # note that this is a non-causal block since we are
                    # denoising the entire image no need for masking
                    is_causal=False,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.out_proj = nn.Sequential(nn.Linear(self.embed_dim, patch_dim), self.rearrange2)

    def forward(self, x, cond):
        self = self.to(torch.float32)
        x = self.patchify_and_embed(x)
        pos_enc = self.precomputed_pos_enc[: x.size(1)].expand(x.size(0), -1)
        x = x + self.pos_embed(pos_enc)
        
        for i,block in enumerate(self.decoder_blocks):
            x = block(x, cond)
        x = self.out_proj(x)
        return x



class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        image_size: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        mlp_multiplier: int = 4,
        n_channels: int = 4
    ):
        super().__init__()

        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels

        self.fourier_feats = nn.Sequential(
            SinusoidalEmbedding(embedding_dims=noise_embed_dims),
            nn.Linear(noise_embed_dims, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.denoiser_trans_block = DenoiserTransBlock(patch_size, image_size, embed_dim, dropout, n_layers, mlp_multiplier, n_channels)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, noise_level):
        
        noise_level = self.fourier_feats(noise_level)
        noise_level = noise_level.view(-1, 1, self.embed_dim)
        label = torch.zeros((noise_level.shape), dtype=torch.float32).to(Config.device)
        
        noise_label_emb = torch.cat([noise_level, label], dim=1)  # bs, 2, d
        noise_label_emb = self.norm(noise_label_emb).to(torch.float32)
        x = self.denoiser_trans_block(x, noise_label_emb)
        return Prediction(x)


class MambaDenoiser(nn.Module):
    def __init__(
        self,
        image_size: int,
        noise_embed_dims: int,
        patch_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        n_channels: int = 4
    ):
        super().__init__()
        assert n_layers % 8 == 0, f"Number of layers must be division by 8"

        
        self.image_size = image_size
        self.noise_embed_dims = noise_embed_dims
        self.embed_dim = embed_dim
        self.n_channels = n_channels
        self.patch_size = patch_size
        
        patch_dim = self.n_channels * self.patch_size * self.patch_size
        
        n_io=3
        n_f=embed_dim
        n_b=n_layers
        self.patchify_and_embed = nn.Sequential(nn.Conv2d(n_io + 1, n_f, 1), nn.ReLU(), nn.Conv2d(n_f, n_f, 1, bias=False), nn.PixelUnshuffle(2))
        self.mid = nn.ModuleList(Block(n_f * 4, Mamba) for _ in range(n_b))
        self.out_proj = nn.Sequential(nn.Conv2d(n_f * 12, n_f * 4, 1), nn.ReLU(), nn.PixelShuffle(2), nn.Conv2d(n_f, n_io, 1))


    def transpose_xy(self, *args):
        # swap x/y axes of an N[XY]C tensor
        return [a.view(a.shape[0], int(a.shape[1]**0.5), int(a.shape[1]**0.5), a.shape[2]).transpose(1, 2).reshape(a.shape) for a in args]

    def flip_s(self, *args):
        # reverse sequence axis of an NSE tensor
        return [a.flip(1) for a in args]

    def forward(self, x, noise_level):

        
        x = self.patchify_and_embed(torch.cat([x, noise_level.expand(x[:, :1].shape)], 1))
        y = x.flatten(2).transpose(-2, -1)
        z = None
        for i, mid in enumerate(self.mid):
            y, z = mid(y, z)
            
            y, z = self.transpose_xy(y, z)
            if (i + 1) % 4 == 0:
                y, z = self.flip_s(y, z)
                
        y, z = y.transpose(-2, -1).view(x.shape), z.transpose(-2, -1).view(x.shape)
        out = self.out_proj(torch.cat([x, y, z], 1))
        
        return Prediction(out)

