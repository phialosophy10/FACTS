"""2D SRGAN and ESRGAN networks.

Since the low-resolution input is already on the same voxel grid as the
high-resolution target (clinical CT re-sliced to the micro-CT grid, and the
synthetic data upsampled back to full size), the generators do not upsample:
they map an image to a sharpened image of the same size.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------- SRGAN ---
class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters),
            nn.PReLU(),
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters),
        )

    def forward(self, x):
        return x + self.block(x)


class SRGenerator(nn.Module):
    """SRResNet generator (Ledig et al., 2017) without the upsampling layers."""

    def __init__(self, channels=1, filters=64, n_blocks=16, global_residual=True):
        super().__init__()
        self.global_residual = global_residual
        self.conv1 = nn.Sequential(nn.Conv2d(channels, filters, 9, 1, 4), nn.PReLU())
        self.res_blocks = nn.Sequential(*[ResidualBlock(filters) for _ in range(n_blocks)])
        self.conv2 = nn.Sequential(nn.Conv2d(filters, filters, 3, 1, 1), nn.BatchNorm2d(filters))
        self.conv3 = nn.Conv2d(filters, channels, 9, 1, 4)

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(self.res_blocks(out1)) + out1
        out = self.conv3(out)
        return x + out if self.global_residual else out


# --------------------------------------------------------------- ESRGAN ---
class DenseResidualBlock(nn.Module):
    """Residual dense block (5 densely connected convolutions)."""

    def __init__(self, filters, growth=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.convs = nn.ModuleList(
            [nn.Conv2d(filters + i * growth, growth if i < 4 else filters, 3, 1, 1) for i in range(5)]
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feats = [x]
        for i, conv in enumerate(self.convs):
            out = conv(torch.cat(feats, 1))
            if i < 4:
                out = self.act(out)
                feats.append(out)
        return x + self.res_scale * out


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, filters, growth=32, res_scale=0.2):
        super().__init__()
        self.res_scale = res_scale
        self.blocks = nn.Sequential(*[DenseResidualBlock(filters, growth, res_scale) for _ in range(3)])

    def forward(self, x):
        return x + self.res_scale * self.blocks(x)


class ESRGenerator(nn.Module):
    """RRDBNet generator (Wang et al., 2018) without the upsampling layers."""

    def __init__(self, channels=1, filters=64, n_blocks=8, growth=32, global_residual=True):
        super().__init__()
        self.global_residual = global_residual
        self.conv1 = nn.Conv2d(channels, filters, 3, 1, 1)
        self.res_blocks = nn.Sequential(*[RRDB(filters, growth) for _ in range(n_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, channels, 3, 1, 1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.conv2(self.res_blocks(out1)) + out1
        out = self.conv3(out)
        return x + out if self.global_residual else out


# -------------------------------------------------------- Discriminator ---
class Discriminator(nn.Module):
    """VGG-style discriminator. Fully convolutional: returns a map of logits
    (one per 16x16 receptive-field region), so it works for any patch size."""

    def __init__(self, channels=1, filters=(64, 128, 256, 512)):
        super().__init__()

        def block(in_f, out_f, first=False):
            layers = [nn.Conv2d(in_f, out_f, 3, 1, 1)]
            if not first:
                layers.append(nn.BatchNorm2d(out_f))
            layers += [nn.LeakyReLU(0.2, inplace=True),
                       nn.Conv2d(out_f, out_f, 3, 2, 1),
                       nn.BatchNorm2d(out_f),
                       nn.LeakyReLU(0.2, inplace=True)]
            return layers

        layers, in_f = [], channels
        for i, out_f in enumerate(filters):
            layers += block(in_f, out_f, first=(i == 0))
            in_f = out_f
        layers.append(nn.Conv2d(in_f, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def build_generator(model, n_blocks=None):
    model = model.upper()
    if model == "SRGAN":
        return SRGenerator(n_blocks=n_blocks or 16)
    if model == "ESRGAN":
        return ESRGenerator(n_blocks=n_blocks or 8)
    raise ValueError(f"Unknown model {model}")
