"""
Binary segmentation: U-Net-style CNN with a Transformer encoder+decoder bottleneck
and a PatchGAN-style discriminator (image + mask).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvGN(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.gn = nn.GroupNorm(min(8, c_out), c_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class HybridUNetTransformerBinary(nn.Module):
    """
    CNN encoder (/8), spatial TransformerEncoder + TransformerDecoder bottleneck,
    CNN decoder with skips. Output: (B, 1, H, W) logits (binary seg).
    H and W must be divisible by 8.
    """

    def __init__(
        self,
        base_ch: int = 32,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 512,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.1,
        use_transformer: bool = True,
    ):
        super().__init__()
        self.use_transformer = use_transformer
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.enc1 = nn.Sequential(
            _ConvGN(1, c1),
            _ConvGN(c1, c1),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            _ConvGN(c1, c2),
            _ConvGN(c2, c2),
        )
        self.enc3 = nn.Sequential(
            _ConvGN(c2, c3),
            _ConvGN(c3, c3),
        )
        self.proj_in = nn.Linear(c3, d_model)
        self.proj_out = nn.Linear(d_model, c3)
        self.tgt_in = nn.Linear(d_model, d_model, bias=False)

        el = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu"
        )
        self.trans_enc = nn.TransformerEncoder(el, enc_layers)
        dl = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu"
        )
        self.trans_dec = nn.TransformerDecoder(dl, dec_layers)

        self.max_seq = 4096
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seq, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        ch = c3 + c2
        self.up1 = nn.Sequential(
            _ConvGN(ch, c2),
            _ConvGN(c2, c2),
        )
        ch2 = c2 + c1
        self.up2 = nn.Sequential(
            _ConvGN(ch2, c1),
            _ConvGN(c1, c1),
        )
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("Expected (B, 1, H, W)")
        h, w = x.shape[2], x.shape[3]
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"H,W must be divisible by 8, got {h}x{w}")

        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        b, c, hh, ww = p3.shape

        if self.use_transformer:
            seq = hh * ww
            if seq > self.max_seq:
                raise ValueError(f"Spatial tokens {seq} exceed max_seq={self.max_seq}")

            t = p3.flatten(2).transpose(1, 2)
            t = self.proj_in(t)
            t = t + self.pos_emb[:, :seq, :]
            mem = self.trans_enc(t)
            tgt = self.tgt_in(mem)
            out = self.trans_dec(tgt, mem)
            out = self.proj_out(out)
            out = out.transpose(1, 2).reshape(b, c, hh, ww)
        else:
            out = p3

        u = F.interpolate(out, size=e2.shape[2:], mode="bilinear", align_corners=False)
        u = torch.cat([u, e2], dim=1)
        u = self.up1(u)
        u = F.interpolate(u, size=e1.shape[2:], mode="bilinear", align_corners=False)
        u = torch.cat([u, e1], dim=1)
        u = self.up2(u)
        return self.head(u)


class PatchDiscriminator(nn.Module):
    """
    Markovian / PatchGAN discriminator: (image, mask) -> patch logits (B,1,H',W').
    mask can be soft [0,1] or logits (caller chooses).
    """

    def __init__(self, in_ch: int = 2, base: int = 48):
        super().__init__()
        def cblk(i: int, o: int, s: int):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(i, o, 4, s, 1, bias=False)),
                nn.InstanceNorm2d(o, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            cblk(in_ch, base, 2),
            cblk(base, base * 2, 2),
            cblk(base * 2, base * 4, 2),
            nn.utils.spectral_norm(nn.Conv2d(base * 4, 1, 4, 1, 1)),
        )

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image, mask], dim=1)
        return self.net(x)


class PretrainedHybridGAN(nn.Module):
    """
    Leverages a heavily pre-trained ResNet-34 ImageNet encoder dynamically spliced with 
    our custom 4096-token spatial Transformer bottleneck to feed the adversarial critic.
    """
    def __init__(self, use_transformer=True):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.unet = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1)
        self.use_transformer = use_transformer
        
        d_model = 512
        nhead = 8
        dim_ff = 512
        enc_layers = 2
        
        el = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1, batch_first=True, activation="gelu")
        self.trans_enc = nn.TransformerEncoder(el, enc_layers)
        
        self.max_seq = 4096
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_seq, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x):
        features = self.unet.encoder(x)
        features = list(features)
        
        if self.use_transformer:
            f_in = features[-1]
            b, c, hh, ww = f_in.shape
            seq = hh * ww
            if seq > self.max_seq:
                raise ValueError(f"Max seq {self.max_seq} exceeded by {seq}")
                
            t = f_in.flatten(2).transpose(1, 2)
            t = t + self.pos_emb[:, :seq, :]
            mem = self.trans_enc(t)
            out = mem.transpose(1, 2).reshape(b, c, hh, ww)
            features[-1] = out
            
        decoder_output = self.unet.decoder(features)
        return self.unet.segmentation_head(decoder_output)
