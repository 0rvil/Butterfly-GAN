import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=3, feature_map_g=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, feature_map_g * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_g * 8 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (feature_map_g * 8, 4, 4)),  # Output shape: (B, 512, 4, 4)
            nn.ConvTranspose2d(feature_map_g * 8, feature_map_g * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_g * 4, feature_map_g * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_g * 2, feature_map_g, 4, 2, 1),
            nn.BatchNorm2d(feature_map_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_g, img_channels, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_map_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, feature_map_d, 4, 2, 1),  # 64x64 → 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_d, feature_map_d * 2, 4, 2, 1),
            nn.BatchNorm2d(feature_map_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_d * 2, feature_map_d * 4, 4, 2, 1),
            nn.BatchNorm2d(feature_map_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_d * 4, feature_map_d * 8, 4, 2, 1),
            nn.BatchNorm2d(feature_map_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_d * 8, 1, 4, 1, 0),  # Final output 1×1×1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)