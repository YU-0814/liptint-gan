"""U-Net generator, 70x70 PatchGAN discriminator, VGG19 perceptual loss (Isola et al., 2017).
"""
import torch
import torch.nn as nn
from torchvision import models


def down(cin, cout, norm=True):
    layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    return nn.Sequential(*layers, nn.LeakyReLU(0.2, inplace=True))


def up(cin, cout, dropout=False):
    layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """8 down, 8 up, skip connections."""

    def __init__(self, in_ch=4, out_ch=3, nf=64):
        super().__init__()
        self.downs = nn.ModuleList([down(in_ch, nf, norm=False), down(nf, nf * 2), down(nf * 2, nf * 4), down(nf * 4, nf * 8),
                                    down(nf * 8, nf * 8), down(nf * 8, nf * 8), down(nf * 8, nf * 8), down(nf * 8, nf * 8, norm=False)])
        self.ups = nn.ModuleList([up(nf * 8, nf * 8, dropout=True), up(nf * 16, nf * 8, dropout=True), up(nf * 16, nf * 8, dropout=True),
                                  up(nf * 16, nf * 8), up(nf * 16, nf * 4), up(nf * 8, nf * 2), up(nf * 4, nf)])
        self.out = nn.Sequential(nn.ConvTranspose2d(nf * 2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        skips = []
        for layer in self.downs:
            x = layer(x)
            skips.append(x)
        x = self.ups[0](skips[-1])
        for layer, skip in zip(self.ups[1:], reversed(skips[1:-1]), strict=True):
            x = layer(torch.cat([x, skip], 1))
        return self.out(torch.cat([x, skips[0]], 1))


class PatchDiscriminator(nn.Module):
    """Real/fake score per 70x70 patch (30x30 output)."""

    def __init__(self, in_ch=4, out_ch=3, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, 4, 2, 1), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, 4, 1, 1), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, 1, 4, 1, 1))

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))


class PerceptualLoss(nn.Module):
    """L1 between VGG19 features at relu1_2, relu2_2, relu3_2."""

    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:14].eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.taps = {3, 8, 13}

    def forward(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.taps:
                loss = loss + nn.functional.l1_loss(x, y)
        return loss
