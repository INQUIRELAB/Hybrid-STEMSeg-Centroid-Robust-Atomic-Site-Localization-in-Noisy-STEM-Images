"""Spatial Transformer bottleneck (encoder + decoder) for CNN U-Net feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialTransformerBottleneck(nn.Module):
    """
    Replace a conv bottleneck with TransformerEncoder + TransformerDecoder on
    flattened spatial tokens, matching HybridUNetTransformerBinary.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int = 256,
        nhead: int = 8,
        dim_ff: int = 512,
        enc_layers: int = 2,
        dec_layers: int = 2,
        dropout: float = 0.1,
        max_seq: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.max_seq = max_seq

        self.proj_in = nn.Linear(in_channels, d_model)
        self.proj_out = nn.Linear(d_model, out_channels)
        self.tgt_in = nn.Linear(d_model, d_model, bias=False)

        el = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu"
        )
        self.trans_enc = nn.TransformerEncoder(el, enc_layers)
        dl = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, activation="gelu"
        )
        self.trans_dec = nn.TransformerDecoder(dl, dec_layers)

        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, hh, ww = x.shape
        seq = hh * ww
        if seq > self.max_seq:
            raise ValueError(f"Spatial tokens {seq} exceed max_seq={self.max_seq}")

        t = x.flatten(2).transpose(1, 2)
        t = self.proj_in(t)
        t = t + self.pos_emb[:, :seq, :]
        mem = self.trans_enc(t)
        tgt = self.tgt_in(mem)
        out = self.trans_dec(tgt, mem)
        out = self.proj_out(out)
        return out.transpose(1, 2).reshape(b, self.out_channels, hh, ww)
