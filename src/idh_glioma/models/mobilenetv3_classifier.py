from __future__ import annotations

import torch.nn as nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small


def build_mobilenetv3_binary(
    num_input_channels: int = 3,
    pretrained: bool = True,
    variant: str = "small",
) -> nn.Module:
    """Build a MobileNetV3 binary classifier.

    ``variant="small"`` keeps backwards compatibility with the existing CT/MRI
    checkpoint (`mobilenetv3_ct_best.pt`). ``variant="large"`` is preferred for
    the IDH task on the small TCGA-LGG cohort because the bigger ImageNet stem
    transfers better when training data is limited.
    """
    weights = "DEFAULT" if pretrained else None
    if variant == "small":
        model = mobilenet_v3_small(weights=weights)
    elif variant == "large":
        model = mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Expected 'small' or 'large'.")

    if num_input_channels != 3:
        # Replace stem only when channel count differs (loses pretrained stem weights).
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
