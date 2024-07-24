import torch
from torch import nn
from torch.nn import functional as F
# from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def _init_(self):
        super().__init__(
            # (Batch_Size,Channel, Height, Width) -> (Batch_size, 128, Height,Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1)

        )
