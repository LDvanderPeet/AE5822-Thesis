import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPool + DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upscaling block: ConvTranspose + skip connection +DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(x,
                  [
                      diff_y // 2,
                      diff_x - diff_x // 2,
                      diff_y // 2,
                      diff_y - diff_y // 2]
                  )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for SAR reconstruction
    """
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, depth: int = 4):
        super().__init__()

        self.depth = depth

        self.inc = DoubleConv(in_channels, base_channels)

        self.downs = nn.ModuleList()
        channels = base_channels
        for _ in range(depth):
            self.downs.append(Down(channels, channels * 2))
            channels *= 2

        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(channels, channels // 2))
            channels //= 2

        self.outc = OutConv(channels, out_channels)

    def forward(self, x):
        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip)

        return self.outc(x)

#base_channels = base_channels class VAE(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(784, 400),
#             nn.ReLU(),
#             nn.Linear(400, 20)  # 10 for mean and 10 for log-variance
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(10, 400),
#             nn.ReLU(),
#             nn.Linear(400, 784),
#             nn.Sigmoid()
#         )
#
# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super().__init__()
#
#         def block(feat_in, feat_out, normalize=True):
#             layers = [nn.Linear(feat_in, feat_out)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(feat_out, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *block(latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )
