import math

from torch import nn


def init_layers(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LSTMCell):
        nn.init.orthogonal_(m.weight_hh, gain=math.sqrt(2.0))
        nn.init.orthogonal_(m.weight_ih, gain=math.sqrt(2.0))

        if m.bias is not None:
            nn.init.zeros_(m.bias_hh)
            nn.init.zeros_(m.bias_ih)

    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.GroupNorm):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
