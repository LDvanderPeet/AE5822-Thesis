import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__innit__()

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
