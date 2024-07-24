import torch
from torch import nn
from torch.nn import functional as F
# from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
