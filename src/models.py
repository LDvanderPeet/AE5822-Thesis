import torch
import torch.nn as nn
import numpy as np
from diffusers import UNet2DConditionModel


class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20)  # 10 for mean and 10 for log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()

        def block(feat_in, feat_out, normalize=True):
            layers = [nn.Linear(feat_in, feat_out)]
            if normalize:
                layers.append(nn.BatchNorm1d(feat_out, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
