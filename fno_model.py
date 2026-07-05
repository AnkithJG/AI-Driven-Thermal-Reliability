import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(x_ft[:, :, :self.modes, :self.modes], self.weights1)
        out_ft[:, :, -self.modes:, :self.modes] = self.compl_mul2d(x_ft[:, :, -self.modes:, :self.modes], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    


class FNO2d(nn.Module):
    def __init__(self, modes, width):
        super(FNO2d, self).__init__()
        self.modes = modes
        self.width = width

        self.p = nn.Linear(1, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.q1 = nn.Linear(self.width, 128)
        self.q2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.q1(x)
        x = F.gelu(x)
        x = self.q2(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
