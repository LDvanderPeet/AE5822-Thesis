import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """
    Encodes scalar timesteps into sinusoidal vector embeddings.

    Uses the standard Transformer-style positional encoding to map discrete timesteps $t$ to a high-dimensional space,
    allowing the U-Net to be conditioned on the diffusion noise level.

    Parameters
    ----------
    dim : int
        The dimensionality of the resulting embedding vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DoubleConv(nn.Module):
    """
    Standard block containing two successive 3x3 convolutions.

    If `time_emb_dim` is provided, a linear projection is used to add temporal information into the feature maps between
    the two convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_emb_dim : int, optional
        The dimension of the time embedding vector for diffusion conditioning.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        else:
            self.time_mlp = None

    def forward(self, x, t, time_emb=None):
        x = self.relu(self.conv1(x))

        if time_emb is not None and self.time_mlp is not None:
            time_shift = self.time_mlp(time_emb).view(time_emb.shape[0], -1, 1, 1)
            x = x + time_shift

        x = self.relu(self.conv2(x))
        return x


class Down(nn.Module):
    """
    Downscaling block that reduces spatial resolution.

    Applies MaxPool2d followed by a `DoubleConv` block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_emb_dim : int, optional
        Dimension for time embedding injection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)


    def forward(self, x, time_emb=None):
        x = self.pool(x)
        return self.conv(x, time_emb)


class Up(nn.Module):
    """
    Upscaling block that restores spatial resolution using skip connections.

    Performs a ConvTranspose2d, pads the result to match the skip connection dimensions (if necessary), and concatenates
    feature maps before the `DoubleConv`.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    time_emb_dim : int, optional
        Dimension for time embedding injection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, time_emb=None):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(x,
                  [
                      diff_x // 2,
                      diff_x - diff_x // 2,
                      diff_y // 2,
                      diff_y - diff_y // 2]
                  )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x, time_emb)


class OutConv(nn.Module):
    """
    Final 1x1 convolution layer

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Flexible U-Net architecture for SAR reconstruction and diffusion.

    The model dynamically adjusts its input layer based on whether it is operating in 'standard' mode (predicting $y$
    from $x$) or 'diffusion' mode (predicting noise $\epsilon$ from a concatenated $x$ and $y_t$).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    base_channels : int, default=64
        The number of feature channels in the first layer.
    depth : int, default=4
        The number of downsampling and upsampling levels.
    is_diffusion : bool
        If True, initializes time-embedding layers and adjusts the input layer to accept concatenated $x$ and $y_t$.
    """
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 64, depth: int = 4, is_diffusion: bool = False):
        super().__init__()
        self.depth = depth
        self.is_diffusion = is_diffusion

        if is_diffusion:
            self.time_dim = base_channels * 4
            self.time_mlp = nn.Sequential(
                TimeEmbedding(base_channels),
                nn.Linear(base_channels, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim),
            )
            actual_in_channels = in_channels + out_channels
        else:
            self.time_dim = None
            actual_in_channels = in_channels


        self.inc = DoubleConv(actual_in_channels, base_channels, self.time_dim)
        self.downs = nn.ModuleList()
        channels = base_channels
        for _ in range(depth):
            self.downs.append(Down(channels, channels * 2, self.time_dim))
            channels *= 2

        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(Up(channels, channels // 2, self.time_dim))
            channels //= 2

        self.outc = OutConv(channels, out_channels)

    def forward(self, x, t=None):
        """
        Forward pass for the U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        t : torch.Tensor, optional
            A tensor of timesteps (B,) used only if `is_diffusion` is True.

        Returns
        -------
        torch.Tensor
            The reconstructed SAR tensor or the predicted noise.
        """
        time_emb = self.time_mlp(t) if  (self.is_diffusion and t is not None) else None
        skips = []

        x = self.inc(x, time_emb)
        skips.append(x)

        for down in self.downs:
            x = down(x, time_emb)
            skips.append(x)

        skips = skips[:-1][::-1]

        for up, skip in zip(self.ups, skips):
            x = up(x, skip, time_emb)

        return self.outc(x)


if __name__ == "__main__":
    model = UNet(in_channels=22, out_channels=2, is_diffusion=True)
    dummy_input = torch.randn(1, 22, 128, 128)
    dummy_target_noise = torch.randn(1, 2, 128, 128)
    t = torch.randint(0, 1000, (1,))

    # Concatenate condition (22) + noisy target (2) = 24 channels
    model_input = torch.cat([dummy_input, dummy_target_noise], dim=1)
    output = model(model_input, t)
    print(output.shape)  # Should be [1, 2, 128, 128]


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
