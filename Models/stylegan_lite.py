import torch
import torch.nn as nn

# StyleGAN-Lite inspired Generator (simplified)
class StyleGenerator(nn.Module):
    def __init__(self, noise_dim):
        super(StyleGenerator, self).__init__()

        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(True)
            )

        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 512 * 4 * 4),
            nn.Unflatten(1, (512, 4, 4)),
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)

# StyleGAN-Lite inspired Discriminator (simplified)
class StyleDiscriminator(nn.Module):
    def __init__(self):
        super(StyleDiscriminator, self).__init__()

        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, 4, 2, 1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.disc = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1)