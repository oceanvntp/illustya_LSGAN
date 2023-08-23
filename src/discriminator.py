import torch.nn as nn

class Discriminator(nn.Module):
    """
    識別機のクラス
    input size: (batch size, 4, 256, 256)
    """
    def __init__(self, nch=3, nch_d=32):
        super().__init__()
        self.nch = nch
        self.nch_d = nch_d

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nch, nch_d, 4, 2, 1),
                nn.LeakyReLU(negative_slope=0.2)
            ), #(B, 32, 64, 64)
            'layer1': nn.Sequential(
                nn.Conv2d(nch_d, nch_d*2, 4, 2, 1),
                nn.BatchNorm2d(nch_d*2),
                nn.LeakyReLU(negative_slope=0.2)
            ), #(B, 64, 32, 32)
            'layer2': nn.Sequential(
                nn.Conv2d(nch_d*2, nch_d*4, 4, 2, 1),
                nn.BatchNorm2d(nch_d*4),
                nn.LeakyReLU(negative_slope=0.2)
            ), #(B, 128, 16, 16)
            'layer3': nn.Sequential(
                nn.Conv2d(nch_d*4, nch_d*8, 4, 2, 1),
                nn.BatchNorm2d(nch_d*8),
                nn.LeakyReLU(negative_slope=0.2)
            ), #(B, 256, 8, 8)
            'layer4': nn.Sequential(
                nn.Conv2d(nch_d*8, nch_d*16, 4, 2, 1),
                nn.BatchNorm2d(nch_d*16),
                nn.LeakyReLU(negative_slope=0.2)
            ), #(B, 512, 4, 4)
            'classifire': nn.Sequential(
                nn.Conv2d(nch_d*16, 1, 4, 1, 0)
            )
            # 'layer5': nn.Sequential(
            #     nn.Conv2d(nch_d*16, nch_d*32, 4, 2, 1),
            #     nn.BatchNorm2d(nch_d*32),
            #     nn.LeakyReLU(negative_slope=0.2)
            # ), #(B, 1024, 4, 4)
            # 'layer6':
            #     nn.Conv2d(nch_d*32, 1, 4, 1, 0)
        })


    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x.squeeze()