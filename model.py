import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lz = nn.Linear(64, 128 * 4 * 4)
        self.lc = nn.Sequential(
            nn.BatchNorm2d(128), # b 128 4 4
            nn.Upsample(scale_factor=2), # b 128 8 8
            nn.Conv2d(128, 64, 3, stride=1, padding=1), # b 128 11 11 -> b 64 8 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # b 64 16 16
            nn.Conv2d(64, 32, 3, stride=1, padding=1), # b 64 18 18 -> b 32 16 16
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # b 32 32 32
            nn.Conv2d(32, 16, 3, stride=1, padding=1), # b 32 34 34 -> b 16 32 32
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), # b 16 64 64
            nn.Conv2d(16, 8, 3, stride=1, padding=1), # b 16 66 66 -> b 8 64 64
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 1, 3, stride=1, padding=1), # b 1 64 64
            nn.Sigmoid()
        )
    
    def forward(self, z):
        out = self.lz(z)
        out = out.view(z.shape[0], 128, 4, 4)
        out = self.lc(out)
        return out # b 1 64 64


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lc = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), # b 1 66 66 -> b 16 64 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # b 16 32 32
            nn.Dropout(0.25),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, 1), # b 16 34 34 -> b 32 32 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # b 32 16 16
            nn.Dropout(0.25),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1), # b 32 18 18 -> b 64 16 16
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # b 64 8 8
            nn.Dropout(0.25),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1), # b 64 11 11 -> b 128 8 8
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2), # b 128 4 4
            nn.BatchNorm2d(128)
        )
        self.ld = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, prob=True):
        return self.ld(self.lc(x).view(x.shape[0], 128 * 4 * 4))
