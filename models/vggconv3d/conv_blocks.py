"""VGG-style 3D convolutional stack, Sec. 3.3: "several convolutional
layers ... each followed by leaky ReLU activation", with CBAM inserted
after each block. Xavier initialisation is applied to all conv/linear
weights (Sec. 3.3.2).

The paper does not give exact layer counts, channel widths, or kernel
sizes -- this stack (4 blocks, 64->128->256->512, spatial-only downsampling
via 3D maxpool) is a standard VGG-3D configuration, not a value read off
the paper.
"""

import torch
import torch.nn as nn

from models.vggconv3d.cbam import CBAM


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.cbam = CBAM(out_channels)
        # spatial-only downsampling: keep the temporal axis intact until
        # the dedicated temporal max pool at the end of the stack.
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv(x))
        x = self.cbam(x)
        return self.pool(x)


class VGGConv3DBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, channels=(64, 128, 256, 512)):
        super().__init__()
        blocks = []
        c_in = in_channels
        for c_out in channels:
            blocks.append(Conv3DBlock(c_in, c_out))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = c_in
        self._init_xavier()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)  # (B, C_out, T, H', W')
