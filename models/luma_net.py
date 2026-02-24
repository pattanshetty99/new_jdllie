import torch.nn as nn

class LumaNet(nn.Module):
    def __init__(self):
        super(LumaNet, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.resblocks = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64,64,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,64,3,padding=1)
            ) for _ in range(5)]
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        res = self.resblocks(feat)
        out = feat + res
        return self.decoder(out)
