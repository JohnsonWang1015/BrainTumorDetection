from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(
        self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32
    ) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8, 2, stride=2
        )
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    @staticmethod
    def _align_spatial(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        src_h, src_w = src.shape[-2], src.shape[-1]
        ref_h, ref_w = ref.shape[-2], ref.shape[-1]

        if src_h < ref_h or src_w < ref_w:
            pad_h = max(ref_h - src_h, 0)
            pad_w = max(ref_w - src_w, 0)
            src = F.pad(
                src,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )
            src_h, src_w = src.shape[-2], src.shape[-1]

        if src_h > ref_h or src_w > ref_w:
            crop_h = max(src_h - ref_h, 0)
            crop_w = max(src_w - ref_w, 0)
            src = src[
                :,
                :,
                crop_h // 2 : src_h - (crop_h - crop_h // 2),
                crop_w // 2 : src_w - (crop_w - crop_w // 2),
            ]

        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._align_spatial(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self._align_spatial(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self._align_spatial(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self._align_spatial(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)
