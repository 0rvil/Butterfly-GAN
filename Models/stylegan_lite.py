import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=100, w_dim=512, num_layers=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2)]
            z_dim = w_dim
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        z = z / z.norm(dim=1, keepdim=True)
        return self.mapping(z)



class AdaIN(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)
        self.fc.bias.data[:channels] = 1 # Scale, variance
        self.fc.bias.data[channels:] = 0 # Bias, bias


    def forward(self, x, w):
        style = self.fc(w)
        scale, bias = style.chunk(2, dim=1)
        scale = scale.unsqueeze(2).unsqueeze(3)
        bias = bias.unsqueeze(2).unsqueeze(3)
        x = F.instance_norm(x)
        return scale * x + bias



class StyledConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise_strength = nn.Parameter(torch.full((1,), 0.05))
        self.adain = AdaIN(style_dim, out_ch)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise=None):
        x = self.conv(x)
        if noise is None:
            noise = torch.randn_like(x)
        x = x + self.noise_strength * noise
        x = self.adain(x, w)
        return self.lrelu(x)

        
class StyleGANGenerator(nn.Module):
    def __init__(self, z_dim=100, w_dim=512):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.input_const = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.blocks = nn.ModuleList([
            StyledConvBlock(512, 512, w_dim),
            StyledConvBlock(512, 256, w_dim),
            StyledConvBlock(256, 128, w_dim),
            StyledConvBlock(128, 64, w_dim),
            nn.Conv2d(64, 3, 1)
        ])

    def synthesis(self, w):
        x = self.input_const.repeat(w.size(0), 1, 1, 1)
        for block in self.blocks[:-1]:
            x = block(x, w)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return torch.tanh(self.blocks[-1](x))

    def forward(self, z):
        w = self.mapping(z)
        return self.synthesis(w)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.down = nn.AvgPool2d(2)
        self.lrelu = nn.LeakyReLU(0.2)


    def forward(self,x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.down(x)
        return x


class StyleGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_rgb = nn.Conv2d(3, 64, 1)
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512)
        ])
        self.final_conv = nn.Conv2d(512, 512, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(513 * 4 * 4, 1)
        )


    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        x = self.lrelu(self.final_conv(x))
        x = self.avgpool(x)
        stddev = torch.std(x, dim=0, keepdim=True)
        stddev = stddev.mean().expand(x.size(0), 1, x.size(2), x.size(3))
        x = torch.cat([x, stddev], 1)
        return self.fc(x)


