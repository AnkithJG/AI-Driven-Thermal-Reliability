import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = (1 / (in_channels * out_channels))
        #Hermitian Symmetry
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        pass