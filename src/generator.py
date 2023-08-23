import torch 
import torch.nn as nn

class Generator(nn.Module):
    '''
    生成機Gのクラス
    input size: (batch_size, 100, 1, 1)
    output size: (batch_size, 4, 256, 256)
    '''
    def __init__(self, nz=100, nch_g=32, net_g=32, nch=3):
        super().__init__()
        self.nz = nz
        self.net_g = net_g
        self.nch = nch

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(in_channels=nz, out_channels=nch_g*16, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(nch_g*16),
                nn.ReLU()
            ), #(B, 512, 4, 4)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(nch_g*16, nch_g*8, 4, 2, 1),
                nn.BatchNorm2d(nch_g*8),
                nn.ReLU()
            ), # (B, 256, 8, 8)
            'layer2': nn.Sequential(
                nn.ConvTranspose2d(nch_g*8, nch_g*4, 4, 2, 1),
                nn.BatchNorm2d(nch_g*4),
                nn.ReLU()
            ), # (B, 128, 16, 16)
            'layer3': nn.Sequential(
                nn.ConvTranspose2d(nch_g*4, nch_g*2, 4, 2, 1),
                nn.BatchNorm2d(nch_g*2),
                nn.ReLU()
            ), # (B, 64, 32, 32)
            'layer4': nn.Sequential(
                nn.ConvTranspose2d(nch_g*2, nch_g, 4, 2, 1),
                nn.BatchNorm2d(nch_g),
                nn.ReLU()
            ), # (B, 32, 64, 64)
            'layer5': nn.Sequential(
                nn.ConvTranspose2d(nch_g, nch, 4, 2, 1),
                nn.BatchNorm2d(nch),
                nn.Tanh()
            ), # (B, 32, 128, 128)
        })


    def forward(self, z):
        for layer in self.layers.values():
            z = layer(z)
        return z