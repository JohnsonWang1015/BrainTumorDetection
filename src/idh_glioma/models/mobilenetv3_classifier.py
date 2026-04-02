from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def build_mobilenetv3_binary(num_input_channels: int = 3) -> nn.Module:
    model = mobilenet_v3_small(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        num_input_channels,
        first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model
